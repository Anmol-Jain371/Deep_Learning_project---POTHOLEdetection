"""
========================================================================================
RoadVision AI -- Phase 4A.1: Leakage-Controlled Pothole440 YOLO-Seg Dataset Builder
========================================================================================
Strictly read-only with respect to ./Pothole440-main/Pothole440-main/.
Builds derived dataset `dataset_pothole440_seg_grouped/` with zero base-group leakage:
  1. Groups all 440 files into 98 base capture IDs (YYYY_MM_DD_pothole_NN).
  2. Stratified deterministic split (~70% train / 15% val / 15% test = 68 / 15 / 15)
     with seed=42, keeping original non-augmented images for val and test, and
     excluding val/test augmentations from training.
  3. Quarantines empty-mask augmented crops (6 files) after inspection.
  4. Converts binary masks to normalized YOLO polygon annotations (class 0) via
     cv2.RETR_EXTERNAL + eps=0.0005*arcLength simplification.
  5. Reconstructs masks from written YOLO labels and measures IoU & Dice per sample.
  6. Generates 24 RGB+polygon training overlays under
     runs/benchmark/pothole440_seg_preparation/.
  7. Verifies zero cross-split base-group/SHA256 leakage and freezes manifests.
========================================================================================
"""

import os
import re
import csv
import json
import yaml
import shutil
import random
import hashlib
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(BASE_DIR, "Pothole440-main", "Pothole440-main")
OUT_DATASET = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped")
PREP_REPORT_DIR = os.path.join(BASE_DIR, "runs", "benchmark", "pothole440_seg_preparation")
OVERLAY_DIR = os.path.join(PREP_REPORT_DIR, "train_overlays")
os.makedirs(PREP_REPORT_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

SEED = 42
EPS_FACTOR = 0.0005  # Polygon Douglas-Peucker simplification factor * arcLength
MIN_CONTOUR_AREA_PX = 16.0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_64(img_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = r[:, 1:] > r[:, :-1]
    v = 0
    for b in diff.flatten():
        v = (v << 1) | int(b)
    return v


def phash_64(img_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(r)[:8, :8]
    med = np.median(dct.flatten()[1:])
    v = 0
    for b in (dct > med).flatten():
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def parse_base_id(stem: str) -> Tuple[str, str, str, bool]:
    m = re.match(r"^(\d{4}_\d{2}_\d{2}_pothole_\d+)(.*)$", stem)
    if not m:
        return stem, "", "unknown", True
    base_id = m.group(1)
    suffix = m.group(2)
    session = base_id[:10]
    is_original = (suffix == "")
    return base_id, suffix, session, is_original


def mask_to_yolo_polygons(mask_2d: np.ndarray) -> Tuple[List[List[float]], Dict[str, Any]]:
    H, W = mask_2d.shape[:2]
    fg = (mask_2d > 0).astype(np.uint8)

    # Check topology including interior holes for diagnostics
    _, hier = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    n_holes = int(sum(1 for h in hier[0] if h[3] >= 0)) if hier is not None else 0

    n_labels, _, _, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    n_components = max(0, n_labels - 1)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    vertex_counts: List[int] = []
    dropped_tiny = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA_PX and len(contours) > 1:
            # Preserve small pothole components unless sub-noise (<16 px^2) with another main contour;
            # wait: never silently drop valid potholes—keep >= 4 px^2!
            if area < 4.0:
                dropped_tiny += 1
                continue
        peri = cv2.arcLength(cnt, True)
        eps = EPS_FACTOR * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            approx = cnt
        if len(approx) < 3:
            dropped_tiny += 1
            continue

        pts = approx.reshape(-1, 2)
        norm_coords: List[float] = []
        for x_px, y_px in pts:
            xn = float(np.clip(x_px / float(W), 0.0, 1.0))
            yn = float(np.clip(y_px / float(H), 0.0, 1.0))
            norm_coords.extend([xn, yn])

        polygons.append(norm_coords)
        vertex_counts.append(len(pts))

    meta = {
        "connected_components": n_components,
        "interior_holes": n_holes,
        "polygons_exported": len(polygons),
        "vertex_counts": vertex_counts,
        "dropped_degenerate_contours": dropped_tiny,
    }
    return polygons, meta


def reconstruct_mask_from_yolo(label_path: str, H: int, W: int) -> np.ndarray:
    canvas = np.zeros((H, W), dtype=np.uint8)
    if not os.path.exists(label_path):
        return canvas
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = [float(v) for v in parts[1:]]
            pts = np.array(
                [[int(round(coords[i] * W)), int(round(coords[i + 1] * H))] for i in range(0, len(coords) - 1, 2)],
                dtype=np.int32,
            )
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            cv2.fillPoly(canvas, [pts], 255)
    return canvas


def compute_iou_dice(mask_gt: np.ndarray, mask_rec: np.ndarray) -> Tuple[float, float]:
    a = mask_gt > 0
    b = mask_rec > 0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    sum_ab = int(a.sum() + b.sum())
    if union == 0:
        return 1.0, 1.0
    iou = inter / float(union)
    dice = (2.0 * inter) / float(sum_ab) if sum_ab > 0 else 1.0
    return iou, dice


def main():
    rng = random.Random(SEED)

    # Clean output dataset dir before rebuilding deterministically
    if os.path.exists(OUT_DATASET):
        shutil.rmtree(OUT_DATASET)

    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT_DATASET, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT_DATASET, "labels", sp), exist_ok=True)
    os.makedirs(os.path.join(OUT_DATASET, "manifests"), exist_ok=True)

    # 1. Scan all 440 source triplets
    all_items = []
    groups = defaultdict(list)

    for orig_sp in ("train", "val", "test"):
        rgb_dir = os.path.join(SRC_ROOT, orig_sp, "rgb")
        mask_dir = os.path.join(SRC_ROOT, orig_sp, "mask")
        for fn in sorted(os.listdir(rgb_dir)):
            if not fn.lower().endswith(".png"):
                continue
            stem = os.path.splitext(fn)[0]
            base_id, suffix, session, is_orig = parse_base_id(stem)
            rp = os.path.join(rgb_dir, fn)
            mp = os.path.join(mask_dir, fn)
            mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            fg_px = int(np.sum(mask > 0))
            item = {
                "orig_split": orig_sp,
                "filename": fn,
                "stem": stem,
                "base_id": base_id,
                "suffix": suffix,
                "session": session,
                "is_original": is_orig,
                "rgb_path": rp,
                "mask_path": mp,
                "rgb_sha256": sha256_file(rp),
                "fg_px": fg_px,
                "is_empty_mask": (fg_px == 0),
            }
            all_items.append(item)
            groups[base_id].append(item)

    # 2. Stratified base-group split (~70/15/15 => 68 / 15 / 15) across sessions
    # Force groups lacking an unaugmented original (only 2022_06_03_pothole_39) into train
    by_session = defaultdict(list)
    forced_train = []
    for bid in sorted(groups):
        has_orig = any(x["is_original"] and not x["is_empty_mask"] for x in groups[bid])
        if not has_orig:
            forced_train.append(bid)
        else:
            by_session[bid[:10]].append(bid)

    # Sort sessions and compute quota proportional to session size targeting 15 val, 15 test
    eligible_bids = []
    for s in sorted(by_session):
        s_bids = sorted(by_session[s])
        rng.shuffle(s_bids)
        eligible_bids.append((s, s_bids))

    # Proportional allocation using largest-remainder method for 15 val & 15 test across sessions
    total_elig = sum(len(x[1]) for x in eligible_bids)
    val_quotas = {s: len(bids) * 15.0 / total_elig for s, bids in eligible_bids}
    test_quotas = {s: len(bids) * 15.0 / total_elig for s, bids in eligible_bids}

    def apportion(quotas: Dict[str, float], target: int) -> Dict[str, int]:
        floors = {k: int(v) for k, v in quotas.items()}
        rem = target - sum(floors.values())
        order = sorted(quotas.keys(), key=lambda k: (quotas[k] - floors[k], quotas[k]), reverse=True)
        for k in order[:rem]:
            floors[k] += 1
        return floors

    val_counts = apportion(val_quotas, 15)
    # Apportion test out of remaining groups cleanly
    rem_quotas = {s: (len(bids) - val_counts[s]) * 15.0 / (total_elig - 15) for s, bids in eligible_bids}
    test_counts = apportion(rem_quotas, 15)

    group_to_split: Dict[str, str] = {}
    for bid in forced_train:
        group_to_split[bid] = "train"

    for s, bids in eligible_bids:
        nv = val_counts[s]
        nt = test_counts[s]
        for bid in bids[:nv]:
            group_to_split[bid] = "val"
        for bid in bids[nv:nv + nt]:
            group_to_split[bid] = "test"
        for bid in bids[nv + nt:]:
            group_to_split[bid] = "train"

    # 3. Export images, convert masks to YOLO-Seg polygons, verify IoU & Dice
    manifest_rows = []
    split_records = defaultdict(list)
    conversion_stats = []
    train_candidates_for_overlay = []

    for item in all_items:
        target_sp = group_to_split[item["base_id"]]
        disposition = None
        label_rel = ""
        img_rel = ""
        iou_val = None
        dice_val = None
        poly_meta = {"connected_components": 0, "interior_holes": 0, "polygons_exported": 0, "vertex_counts": [], "dropped_degenerate_contours": 0}

        if item["is_empty_mask"]:
            disposition = "QUARANTINED_EMPTY_MASK"
        elif target_sp in ("val", "test") and not item["is_original"]:
            disposition = f"EXCLUDED_{target_sp.upper()}_AUGMENTATION"
        else:
            disposition = f"INCLUDED_{target_sp.upper()}"
            img_dst = os.path.join(OUT_DATASET, "images", target_sp, item["filename"])
            lbl_dst = os.path.join(OUT_DATASET, "labels", target_sp, item["stem"] + ".txt")
            img_rel = f"images/{target_sp}/{item['filename']}"
            lbl_rel = f"labels/{target_sp}/{item['stem']}.txt"

            shutil.copy2(item["rgb_path"], img_dst)

            mask = cv2.imread(item["mask_path"], cv2.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            H, W = mask.shape[:2]
            polygons, poly_meta = mask_to_yolo_polygons(mask)

            with open(lbl_dst, "w", encoding="utf-8") as lf:
                for poly in polygons:
                    coords_str = " ".join(f"{v:.6f}" for v in poly)
                    lf.write(f"0 {coords_str}\n")

            # Reconstruct and score IoU/Dice
            rec_mask = reconstruct_mask_from_yolo(lbl_dst, H, W)
            iou_val, dice_val = compute_iou_dice((mask > 0).astype(np.uint8) * 255, rec_mask)
            conversion_stats.append({
                "split": target_sp,
                "filename": item["filename"],
                "base_id": item["base_id"],
                "iou": iou_val,
                "dice": dice_val,
                "connected_components": poly_meta["connected_components"],
                "interior_holes": poly_meta["interior_holes"],
                "polygons_exported": poly_meta["polygons_exported"],
                "total_vertices": sum(poly_meta["vertex_counts"]),
                "img_dst": img_dst,
                "lbl_dst": lbl_dst,
                "mask_path": item["mask_path"],
            })

            rgb_img = cv2.imread(img_dst)
            split_records[target_sp].append({
                "filename": item["filename"],
                "base_id": item["base_id"],
                "session": item["session"],
                "is_original": item["is_original"],
                "rgb_sha256": item["rgb_sha256"],
                "dhash": dhash_64(rgb_img),
                "phash": phash_64(rgb_img),
            })

            if target_sp == "train":
                train_candidates_for_overlay.append(conversion_stats[-1])

        manifest_rows.append({
            "filename": item["filename"],
            "base_id": item["base_id"],
            "session": item["session"],
            "orig_pothole440_split": item["orig_split"],
            "assigned_group_split": target_sp,
            "is_original": item["is_original"],
            "is_empty_mask": item["is_empty_mask"],
            "disposition": disposition,
            "img_rel_path": img_rel,
            "lbl_rel_path": lbl_rel,
            "rgb_sha256": item["rgb_sha256"],
            "connected_components": poly_meta["connected_components"],
            "interior_holes": poly_meta["interior_holes"],
            "polygons_exported": poly_meta["polygons_exported"],
            "conversion_iou": round(iou_val, 6) if iou_val is not None else "",
            "conversion_dice": round(dice_val, 6) if dice_val is not None else "",
        })

    # 4. Write split_manifest.csv
    csv_path = os.path.join(OUT_DATASET, "manifests", "split_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 5. Render 24 diverse training overlays (4 grids of 6 + individual inspection metrics)
    # Include multi-component, hole-containing, lowest-IoU, and diverse-session train samples
    sorted_tr = sorted(train_candidates_for_overlay, key=lambda x: x["iou"])
    chosen_overlays = []
    # 6 lowest-IoU cases + 4 multi-component cases + 14 evenly spaced across remaining
    for c in sorted_tr[:6]:
        if c not in chosen_overlays:
            chosen_overlays.append(c)
    for c in sorted_tr:
        if c["connected_components"] >= 2 and c not in chosen_overlays:
            chosen_overlays.append(c)
            if len(chosen_overlays) >= 10:
                break
    remaining = [c for c in sorted_tr if c not in chosen_overlays]
    step = max(1, len(remaining) // (24 - len(chosen_overlays)))
    for c in remaining[::step]:
        if len(chosen_overlays) >= 24:
            break
        chosen_overlays.append(c)

    # Render individual overlay images AND 4 summary grids (6 per grid)
    tiles = []
    for idx, c in enumerate(chosen_overlays, 1):
        img = cv2.imread(c["img_dst"])
        H, W = img.shape[:2]
        ov = img.copy()
        with open(c["lbl_dst"], "r", encoding="utf-8") as lf:
            for line in lf:
                parts = line.strip().split()
                coords = [float(v) for v in parts[1:]]
                pts = np.array(
                    [[int(round(coords[i] * W)), int(round(coords[i + 1] * H))] for i in range(0, len(coords) - 1, 2)],
                    dtype=np.int32,
                )
                cv2.fillPoly(ov, [pts], (0, 220, 120))
                cv2.polylines(img, [pts], True, (0, 255, 255), 3)
        blended = cv2.addWeighted(img, 0.72, ov, 0.28, 0)
        # Re-draw crisp yellow polygon border
        with open(c["lbl_dst"], "r", encoding="utf-8") as lf:
            for line in lf:
                parts = line.strip().split()
                coords = [float(v) for v in parts[1:]]
                pts = np.array(
                    [[int(round(coords[i] * W)), int(round(coords[i + 1] * H))] for i in range(0, len(coords) - 1, 2)],
                    dtype=np.int32,
                )
                cv2.polylines(blended, [pts], True, (0, 255, 255), 3)

        indiv_path = os.path.join(OVERLAY_DIR, f"overlay_{idx:02d}_{os.path.splitext(c['filename'])[0]}.jpg")
        cv2.putText(
            blended,
            f"{c['filename']} | IoU={c['iou']:.4f} Dice={c['dice']:.4f} | polys={c['polygons_exported']} holes={c['interior_holes']}",
            (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2
        )
        cv2.imwrite(indiv_path, cv2.resize(blended, (900, 600)), [cv2.IMWRITE_JPEG_QUALITY, 90])

        tile = cv2.resize(blended, (480, 320))
        tiles.append(tile)

    for g_idx in range(4):
        chunk = tiles[g_idx * 6:(g_idx + 1) * 6]
        if not chunk:
            continue
        while len(chunk) < 6:
            chunk.append(np.zeros_like(chunk[0]))
        grid = np.vstack([np.hstack(chunk[:3]), np.hstack(chunk[3:6])])
        cv2.imwrite(os.path.join(PREP_REPORT_DIR, f"train_overlays_grid_{g_idx + 1}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 6. Write portable data.yaml & dataset README.md
    data_yaml = {
        "path": "dataset_pothole440_seg_grouped",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
        "nc": 1,
    }
    yaml_path = os.path.join(OUT_DATASET, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=False)

    readme_text = (
        "# dataset_pothole440_seg_grouped\n\n"
        "Leakage-controlled YOLO instance/region segmentation benchmark reconstructed from `./Pothole440-main/`.\n\n"
        "- **Split Policy:** Deterministic `seed=42` split at the `98`-group base capture ID level (~70/15/15: 68 train / 15 val / 15 test).\n"
        "- **Leakage Control:** Zero base-group overlap across splits; validation and test contain strictly unaugmented original captures (`15` + `15`); all `val`/`test` augmentations excluded.\n"
        "- **Mask Caveat:** Source Pothole440 labels are binary semantic masks (without instance IDs or interior hole encoding in YOLO-Seg).\n"
        "- **Usage & Citation:** Strictly for non-commercial academic research. Cite Ali et al. (2026), *IEEE Transactions on Industrial Informatics*, DOI `10.1109/TII.2026.3706348`.\n"
    )
    with open(os.path.join(OUT_DATASET, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_text)

    # 7. Leakage Verification & Polygon Metrics Summary
    split_groups = {sp: set(r["base_id"] for r in split_records[sp]) for sp in ("train", "val", "test")}
    split_shas = {sp: set(r["rgb_sha256"] for r in split_records[sp]) for sp in ("train", "val", "test")}
    split_sessions = {sp: dict(sorted(Counter(r["session"] for r in split_records[sp]).items())) for sp in ("train", "val", "test")}

    cross_group_overlap = {
        "train_val": sorted(list(split_groups["train"] & split_groups["val"])),
        "train_test": sorted(list(split_groups["train"] & split_groups["test"])),
        "val_test": sorted(list(split_groups["val"] & split_groups["test"])),
    }
    cross_sha_overlap = {
        "train_val": len(split_shas["train"] & split_shas["val"]),
        "train_test": len(split_shas["train"] & split_shas["test"]),
        "val_test": len(split_shas["val"] & split_shas["test"]),
    }

    # Cross-split perceptual similarity check on final included images
    all_included = [r for sp in ("train", "val", "test") for r in split_records[sp]]
    perceptual_cross = []
    for sp1, sp2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        for a in split_records[sp1]:
            for b in split_records[sp2]:
                dd = hamming(a["dhash"], b["dhash"])
                pd = hamming(a["phash"], b["phash"])
                if dd <= 6 and pd <= 6:
                    perceptual_cross.append({
                        "a": f"{sp1}/{a['filename']}",
                        "b": f"{sp2}/{b['filename']}",
                        "dhash_dist": dd,
                        "phash_dist": pd,
                    })

    # Image-label 1-to-1 and coordinate [0,1] verification
    coord_errors = []
    pair_mismatches = []
    for sp in ("train", "val", "test"):
        imgs = sorted(os.listdir(os.path.join(OUT_DATASET, "images", sp)))
        lbls = sorted(os.listdir(os.path.join(OUT_DATASET, "labels", sp)))
        if [os.path.splitext(x)[0] for x in imgs] != [os.path.splitext(x)[0] for x in lbls]:
            pair_mismatches.append(sp)
        for lf in lbls:
            with open(os.path.join(OUT_DATASET, "labels", sp, lf), "r", encoding="utf-8") as fp:
                for ln, line in enumerate(fp, 1):
                    parts = line.strip().split()
                    if len(parts) < 7 or int(parts[0]) != 0 or len(parts[1:]) % 2 != 0:
                        coord_errors.append((sp, lf, ln, "malformed"))
                        continue
                    vals = [float(v) for v in parts[1:]]
                    if any(v < 0.0 or v > 1.0 for v in vals):
                        coord_errors.append((sp, lf, ln, "out_of_range"))

    ious = np.array([c["iou"] for c in conversion_stats], dtype=np.float64)
    dices = np.array([c["dice"] for c in conversion_stats], dtype=np.float64)
    flagged_poor = [
        {"split": c["split"], "filename": c["filename"], "iou": round(c["iou"], 5), "dice": round(c["dice"], 5), "holes": c["interior_holes"], "components": c["connected_components"]}
        for c in sorted(conversion_stats, key=lambda x: x["iou"]) if c["iou"] < 0.96
    ]

    disp_counts = Counter(r["disposition"] for r in manifest_rows)

    dataset_manifest = {
        "manifest_version": "dataset_pothole440_seg_grouped_v1_frozen",
        "seed": SEED,
        "polygon_conversion_settings": {
            "contour_mode": "cv2.RETR_EXTERNAL",
            "chain_approx": "cv2.CHAIN_APPROX_SIMPLE",
            "approx_poly_dp_epsilon_factor": EPS_FACTOR,
            "class_id": 0,
            "class_name": "pothole",
        },
        "summary_counts": {
            "source_total_triplets": len(all_items),
            "unique_base_groups": len(groups),
            "dispositions": dict(disp_counts),
            "splits": {
                sp: {
                    "base_groups": len(split_groups[sp]),
                    "images": len(split_records[sp]),
                    "originals": sum(1 for r in split_records[sp] if r["is_original"]),
                    "augmented": sum(1 for r in split_records[sp] if not r["is_original"]),
                    "polygons_total": sum(c["polygons_exported"] for c in conversion_stats if c["split"] == sp),
                    "sessions": split_sessions[sp],
                }
                for sp in ("train", "val", "test")
            },
        },
        "polygon_conversion_quality": {
            "total_masks_converted": len(conversion_stats),
            "total_polygons_written": int(sum(c["polygons_exported"] for c in conversion_stats)),
            "iou": {
                "mean": round(float(np.mean(ious)), 5),
                "median": round(float(np.median(ious)), 5),
                "p05": round(float(np.percentile(ious, 5)), 5),
                "p10": round(float(np.percentile(ious, 10)), 5),
                "min": round(float(np.min(ious)), 5),
                "max": round(float(np.max(ious)), 5),
            },
            "dice": {
                "mean": round(float(np.mean(dices)), 5),
                "median": round(float(np.median(dices)), 5),
                "p05": round(float(np.percentile(dices, 5)), 5),
                "p10": round(float(np.percentile(dices, 10)), 5),
                "min": round(float(np.min(dices)), 5),
                "max": round(float(np.max(dices)), 5),
            },
            "flagged_iou_lt_0_96_count": len(flagged_poor),
            "flagged_iou_lt_0_96": flagged_poor,
        },
        "leakage_and_integrity_verification": {
            "cross_split_base_group_overlap": cross_group_overlap,
            "cross_split_exact_rgb_sha256_overlap": cross_sha_overlap,
            "cross_split_perceptual_matches_dhash_phash_le_6": perceptual_cross,
            "image_label_pair_mismatches": pair_mismatches,
            "coordinate_integrity_errors": coord_errors,
        },
    }

    manifest_json_path = os.path.join(OUT_DATASET, "manifests", "dataset_manifest.json")
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    sha_path = os.path.join(OUT_DATASET, "manifests", "dataset_manifest.sha256")
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{sha256_file(manifest_json_path)}  dataset_manifest.json\n")
        f.write(f"{sha256_file(csv_path)}  split_manifest.csv\n")
        f.write(f"{sha256_file(yaml_path)}  data.yaml\n")

    print("=== PHASE 4A.1 BUILD SUMMARY ===")
    print(json.dumps(dataset_manifest["summary_counts"], indent=2))
    print("Conversion Quality:", json.dumps(dataset_manifest["polygon_conversion_quality"], indent=2))
    print("Leakage Check:", json.dumps(dataset_manifest["leakage_and_integrity_verification"], indent=2))
    print("Manifest SHA256:", sha256_file(manifest_json_path))


if __name__ == "__main__":
    main()
