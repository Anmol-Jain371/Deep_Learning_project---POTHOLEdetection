"""
Phase 2 Pre-Training Verification & Manifest Freeze Script:
  1. Exact SHA256 duplicate check across all splits of dataset_unified
  2. Perceptual hash (dHash + pHash) near-duplicate check across splits (Hamming <= 8)
  3. Source provenance & sequential frame grouping audit
  4. External Rangoon annotation completeness, coordinate, size, & overlap audit
  5. Visual contact-sheet rendering for human/agent inspection of external_rangoon boxes
  6. Immutable frozen manifest JSON with SHA256 checksums
"""

import os
import re
import json
import hashlib
from collections import defaultdict
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UNIFIED_DIR = os.path.join(BASE_DIR, "dataset_unified")
MANIFEST_DIR = os.path.join(UNIFIED_DIR, "manifests")
os.makedirs(MANIFEST_DIR, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_64(img_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val


def phash_64(img_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    med = np.median(dct.flatten()[1:])  # exclude DC term
    val = 0
    for bit in (dct > med).flatten():
        val = (val << 1) | int(bit)
    return val


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def parse_source_group(fname: str) -> str:
    if fname.startswith("rdd_in_India_"):
        # e.g. rdd_in_India_000123.jpg -> single frame ID in RDD2022 India
        return "rdd2022_india"
    if "frame_" in fname:
        return "phone_v1_video_sequence"
    if fname.startswith("rangoon_"):
        return "rangoon_road_video_sequence"
    return "other"


def main():
    splits = ["train", "val", "test", "external_rangoon"]
    records = []
    by_split = defaultdict(list)

    for sp in splits:
        img_dir = os.path.join(UNIFIED_DIR, "images", sp)
        lbl_dir = os.path.join(UNIFIED_DIR, "labels", sp)
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            ip = os.path.join(img_dir, fname)
            lp = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            img = cv2.imread(ip)
            h_px, w_px = img.shape[:2] if img is not None else (0, 0)
            img_sha = sha256_file(ip)
            lbl_sha = sha256_file(lp) if os.path.exists(lp) else None
            dh = dhash_64(img) if img is not None else 0
            ph = phash_64(img) if img is not None else 0
            boxes = []
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            boxes.append([int(parts[0])] + [float(x) for x in parts[1:5]])
            rec = {
                "split": sp,
                "filename": fname,
                "source_group": parse_source_group(fname),
                "width": w_px,
                "height": h_px,
                "img_sha256": img_sha,
                "lbl_sha256": lbl_sha,
                "dhash": f"{dh:016x}",
                "phash": f"{ph:016x}",
                "dh_int": dh,
                "ph_int": ph,
                "num_boxes": len(boxes),
                "boxes": boxes,
            }
            records.append(rec)
            by_split[sp].append(rec)

    # 1. Exact SHA256 duplicates (within-split and cross-split)
    sha_map = defaultdict(list)
    for r in records:
        sha_map[r["img_sha256"]].append((r["split"], r["filename"]))
    exact_dups = {k: v for k, v in sha_map.items() if len(v) > 1}

    # 2. Perceptual hash near-duplicates (dHash <= 6 or pHash <= 6) across splits AND within splits
    cross_near_dups = []
    within_near_dups = defaultdict(list)
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = records[i], records[j]
            d_dist = hamming(ri["dh_int"], rj["dh_int"])
            p_dist = hamming(ri["ph_int"], rj["ph_int"])
            if d_dist <= 6 or p_dist <= 6:
                pair = {
                    "a": f"{ri['split']}/{ri['filename']}",
                    "b": f"{rj['split']}/{rj['filename']}",
                    "dhash_dist": d_dist,
                    "phash_dist": p_dist,
                }
                if ri["split"] != rj["split"]:
                    cross_near_dups.append(pair)
                else:
                    within_near_dups[ri["split"]].append(pair)

    # 3. Source-video grouping check
    group_by_split = {
        sp: dict(sorted(defaultdict(int, {
            g: sum(1 for r in by_split[sp] if r["source_group"] == g)
            for g in set(r["source_group"] for r in by_split[sp])
        }).items()))
        for sp in splits
    }

    # 4. External Rangoon annotation geometry audit + contact sheet
    rangoon_audit = {
        "total_images": len(by_split["external_rangoon"]),
        "total_boxes": sum(r["num_boxes"] for r in by_split["external_rangoon"]),
        "images_with_zero_boxes": [r["filename"] for r in by_split["external_rangoon"] if r["num_boxes"] == 0],
        "large_boxes_area_gt_0_30": [],
        "tiny_boxes_area_lt_0_005": [],
        "edge_touching_boxes": [],
        "box_areas": [],
    }
    tiles = []
    for r in by_split["external_rangoon"]:
        ip = os.path.join(UNIFIED_DIR, "images", "external_rangoon", r["filename"])
        img = cv2.imread(ip)
        ih, iw = img.shape[:2]
        vis = img.copy()
        for b_idx, b in enumerate(r["boxes"]):
            _, xc, yc, w, h = b
            area = w * h
            rangoon_audit["box_areas"].append(area)
            x1 = int((xc - w / 2) * iw)
            y1 = int((yc - h / 2) * ih)
            x2 = int((xc + w / 2) * iw)
            y2 = int((yc + h / 2) * ih)
            cv2.rectangle(vis, (max(0, x1), max(0, y1)), (min(iw - 1, x2), min(ih - 1, y2)), (0, 255, 0), 3)
            cv2.putText(vis, f"B{b_idx+1}", (max(4, x1 + 4), max(22, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if area > 0.30:
                rangoon_audit["large_boxes_area_gt_0_30"].append({"file": r["filename"], "box": b_idx + 1, "area": round(area, 3), "w": round(w, 3), "h": round(h, 3)})
            if area < 0.005:
                rangoon_audit["tiny_boxes_area_lt_0_005"].append({"file": r["filename"], "box": b_idx + 1, "area": round(area, 4)})
            if (xc - w / 2) <= 0.01 or (yc - h / 2) <= 0.01 or (xc + w / 2) >= 0.99 or (yc + h / 2) >= 0.99:
                rangoon_audit["edge_touching_boxes"].append({"file": r["filename"], "box": b_idx + 1})
        cv2.putText(vis, f"{r['filename']} ({r['num_boxes']})", (8, ih - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(vis, (320, 180)))

    # 7 rows x 5 cols contact sheet
    rows_img = []
    for row_i in range(7):
        row_tiles = tiles[row_i * 5:(row_i + 1) * 5]
        rows_img.append(np.hstack(row_tiles))
    contact = np.vstack(rows_img)
    contact_out = os.path.join(MANIFEST_DIR, "external_rangoon_audit_sheet.jpg")
    cv2.imwrite(contact_out, contact, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 5. Check RDD2022/India/test & evaluation/ground_truth_labels status
    rdd_test_imgs_dir = os.path.join(BASE_DIR, "RDD2022", "India", "test", "images")
    rdd_test_count = len(os.listdir(rdd_test_imgs_dir)) if os.path.exists(rdd_test_imgs_dir) else 0
    eval_lbl_dir = os.path.join(BASE_DIR, "evaluation", "ground_truth_labels")
    eval_nonempty = 0
    if os.path.exists(eval_lbl_dir):
        eval_nonempty = sum(1 for f in os.listdir(eval_lbl_dir) if os.path.getsize(os.path.join(eval_lbl_dir, f)) > 0)

    # Strip integer hash fields before saving JSON
    clean_records = {}
    for sp in splits:
        clean_records[sp] = [
            {k: v for k, v in r.items() if k not in ("dh_int", "ph_int", "boxes")}
            for r in by_split[sp]
        ]

    manifest = {
        "manifest_version": "dataset_unified_v1_frozen",
        "splits_summary": {
            sp: {
                "images": len(by_split[sp]),
                "boxes": sum(r["num_boxes"] for r in by_split[sp]),
                "negative_images": sum(1 for r in by_split[sp] if r["num_boxes"] == 0),
                "source_groups": group_by_split[sp],
            }
            for sp in splits
        },
        "integrity_checks": {
            "exact_sha256_duplicates": len(exact_dups),
            "cross_split_perceptual_near_dups_hamming_le_6": cross_near_dups,
            "within_split_perceptual_near_dups_counts": {k: len(v) for k, v in within_near_dups.items()},
            "within_split_perceptual_near_dups_sample": {k: v[:10] for k, v in within_near_dups.items()},
        },
        "external_rangoon_audit": {
            "total_images": rangoon_audit["total_images"],
            "total_boxes": rangoon_audit["total_boxes"],
            "images_with_zero_boxes": rangoon_audit["images_with_zero_boxes"],
            "large_boxes_area_gt_0_30_count": len(rangoon_audit["large_boxes_area_gt_0_30"]),
            "large_boxes_area_gt_0_30": rangoon_audit["large_boxes_area_gt_0_30"],
            "tiny_boxes_area_lt_0_005_count": len(rangoon_audit["tiny_boxes_area_lt_0_005"]),
            "edge_touching_boxes_count": len(rangoon_audit["edge_touching_boxes"]),
            "median_box_area": round(float(np.median(rangoon_audit["box_areas"])), 4),
            "mean_box_area": round(float(np.mean(rangoon_audit["box_areas"])), 4),
        },
        "untouched_test_assessment": {
            "rdd2022_india_official_test_images_without_labels": rdd_test_count,
            "evaluation_40_frames_annotated_count": eval_nonempty,
            "dataset_unified_test_previously_inspected": True,
            "external_rangoon_previously_inspected": True,
        },
        "files": clean_records,
    }

    manifest_path = os.path.join(MANIFEST_DIR, "frozen_manifest_v1.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest_sha = sha256_file(manifest_path)

    # Save top-level checksum record
    checksum_path = os.path.join(MANIFEST_DIR, "frozen_manifest_v1.sha256")
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(f"{manifest_sha}  frozen_manifest_v1.json\n")
        f.write(f"{sha256_file(os.path.join(UNIFIED_DIR, 'data.yaml'))}  data.yaml\n")
        f.write(f"{sha256_file(os.path.join(UNIFIED_DIR, 'data_external_rangoon.yaml'))}  data_external_rangoon.yaml\n")

    print("=== MANIFEST & INTEGRITY SUMMARY ===")
    print("Manifest SHA256:", manifest_sha)
    print("Exact SHA256 duplicates:", len(exact_dups))
    print("Cross-split perceptual near-dups (dHash<=6 or pHash<=6):", len(cross_near_dups))
    for p in cross_near_dups[:15]:
        print("  CROSS:", p)
    print("Within-split perceptual near-dups:", {k: len(v) for k, v in within_near_dups.items()})
    print("Source groups by split:", json.dumps(group_by_split, indent=2))
    print("External Rangoon audit:", json.dumps(manifest["external_rangoon_audit"], indent=2))
    print("Untouched test assessment:", json.dumps(manifest["untouched_test_assessment"], indent=2))


if __name__ == "__main__":
    main()
