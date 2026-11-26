"""
========================================================================================
RoadVision AI -- Phase 3B: Pothole440 Dataset & Depth Encoding Inspection Suite
========================================================================================
Strictly read-only with respect to ./Pothole440-main/.
Preserves official test split strictly untouched for calibration/model selection
(structural, format, pairing, and hash-leakage checks only on test; exploratory depth
statistics and visualizations are derived exclusively from train/val).

Outputs:
  - runs/benchmark/pothole440_inspection/dataset_inventory.json
  - runs/benchmark/pothole440_inspection/sample_XX_<stem>.png (12 diverse train samples)
========================================================================================
"""

import os
import re
import json
import hashlib
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = os.path.join(BASE_DIR, "Pothole440-main", "Pothole440-main")
OUT_DIR = os.path.join(BASE_DIR, "runs", "benchmark", "pothole440_inspection")
os.makedirs(OUT_DIR, exist_ok=True)


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
    med = np.median(dct.flatten()[1:])
    val = 0
    for bit in (dct > med).flatten():
        val = (val << 1) | int(bit)
    return val


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_session_date(fname: str) -> str:
    m = re.match(r"^(\d{4}_\d{2}_\d{2})_", fname)
    return m.group(1) if m else "unknown"


def audit_split(split: str) -> Dict[str, Any]:
    sp_dir = os.path.join(DATASET_ROOT, split)
    rgb_dir = os.path.join(sp_dir, "rgb")
    mask_dir = os.path.join(sp_dir, "mask")
    depth_dir = os.path.join(sp_dir, "depth")

    rgb_files = sorted(os.listdir(rgb_dir)) if os.path.exists(rgb_dir) else []
    mask_files = sorted(os.listdir(mask_dir)) if os.path.exists(mask_dir) else []
    depth_files = sorted(os.listdir(depth_dir)) if os.path.exists(depth_dir) else []

    rgb_stems = {os.path.splitext(f)[0]: f for f in rgb_files}
    mask_stems = {os.path.splitext(f)[0]: f for f in mask_files}
    depth_stems = {os.path.splitext(f)[0]: f for f in depth_files}

    all_stems = sorted(set(rgb_stems) | set(mask_stems) | set(depth_stems))
    unmatched = []
    corrupted = []
    records = []

    resolutions = Counter()
    rgb_dtypes = Counter()
    mask_dtypes = Counter()
    mask_unique_vals = set()
    depth_dtypes = Counter()
    depth_channels = Counter()
    png_info_keys = set()
    sessions = Counter()

    for stem in all_stems:
        rf = rgb_stems.get(stem)
        mf = mask_stems.get(stem)
        df = depth_stems.get(stem)
        if not (rf and mf and df):
            unmatched.append({"stem": stem, "rgb": rf, "mask": mf, "depth": df})
            continue

        rp = os.path.join(rgb_dir, rf)
        mp = os.path.join(mask_dir, mf)
        dp = os.path.join(depth_dir, df)

        rgb = cv2.imread(rp, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED)

        if rgb is None or mask is None or depth is None:
            corrupted.append({"stem": stem, "rgb_ok": rgb is not None, "mask_ok": mask is not None, "depth_ok": depth is not None})
            continue

        # PNG metadata check
        try:
            with Image.open(dp) as pimg:
                for k in pimg.info.keys():
                    png_info_keys.add(str(k))
        except Exception:
            pass

        rh, rw = rgb.shape[:2]
        mh, mw = mask.shape[:2]
        dh_px, dw_px = depth.shape[:2]
        aligned = (rh == mh == dh_px) and (rw == mw == dw_px)

        resolutions[f"{rw}x{rh}"] += 1
        rgb_dtypes[str(rgb.dtype)] += 1
        mask_dtypes[str(mask.dtype)] += 1
        depth_dtypes[str(depth.dtype)] += 1
        depth_channels[1 if depth.ndim == 2 else depth.shape[2]] += 1

        # Normalize 2D mask if 3-channel
        mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
        uvals = np.unique(mask_2d)
        for uv in uvals[:20]:
            mask_unique_vals.add(int(uv))

        fg = mask_2d > 0
        fg_frac = float(np.mean(fg))
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        n_components = max(0, int(n_labels) - 1)
        contours, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = sum(1 for c in contours if len(c) >= 3 and cv2.contourArea(c) >= 16)

        # Depth single channel check
        depth_2d = depth if depth.ndim == 2 else depth[:, :, 0]
        channels_identical = True if depth.ndim == 2 else (
            np.array_equal(depth[:, :, 0], depth[:, :, 1]) and np.array_equal(depth[:, :, 1], depth[:, :, 2])
        )

        sess = extract_session_date(rf)
        sessions[sess] += 1

        rec = {
            "split": split,
            "stem": stem,
            "filename": rf,
            "session_date": sess,
            "rgb_shape": list(rgb.shape),
            "mask_shape": list(mask.shape),
            "depth_shape": list(depth.shape),
            "aligned": aligned,
            "rgb_sha256": sha256_file(rp),
            "mask_sha256": sha256_file(mp),
            "depth_sha256": sha256_file(dp),
            "dhash": dhash_64(rgb),
            "phash": phash_64(rgb),
            "fg_coverage_pct": round(fg_frac * 100.0, 3),
            "connected_components": n_components,
            "yolo_valid_contours": valid_contours,
            "depth_min": int(np.min(depth_2d)),
            "depth_max": int(np.max(depth_2d)),
            "depth_channels_identical": bool(channels_identical),
        }

        # Detailed depth interior/exterior analysis ONLY on train & val (preserving test untouched)
        if split in ("train", "val"):
            in_vals = depth_2d[fg]
            out_vals = depth_2d[~fg]
            # Dilated annular rim around mask (25px)
            kernel = np.ones((51, 51), np.uint8)
            dilated = cv2.dilate(fg.astype(np.uint8), kernel, iterations=1) > 0
            rim_mask = dilated & (~fg)
            rim_vals = depth_2d[rim_mask]

            out_nonzero = out_vals[out_vals > 0]
            rec["depth_stats"] = {
                "inside_mean": round(float(np.mean(in_vals)), 2) if in_vals.size else None,
                "inside_median": round(float(np.median(in_vals)), 2) if in_vals.size else None,
                "inside_min": int(np.min(in_vals)) if in_vals.size else None,
                "inside_max": int(np.max(in_vals)) if in_vals.size else None,
                "inside_zero_frac": round(float(np.mean(in_vals == 0)), 4) if in_vals.size else None,
                "outside_mean": round(float(np.mean(out_vals)), 2) if out_vals.size else None,
                "outside_median": round(float(np.median(out_vals)), 2) if out_vals.size else None,
                "outside_zero_frac": round(float(np.mean(out_vals == 0)), 4) if out_vals.size else None,
                "outside_nonzero_median": round(float(np.median(out_nonzero)), 2) if out_nonzero.size else None,
                "annular_rim_mean": round(float(np.mean(rim_vals)), 2) if rim_vals.size else None,
                "annular_rim_median": round(float(np.median(rim_vals)), 2) if rim_vals.size else None,
                "centroid": [round(float(centroids[1][0]), 1), round(float(centroids[1][1]), 1)] if n_components >= 1 else None,
            }

        records.append(rec)

    return {
        "split": split,
        "rgb_count": len(rgb_files),
        "mask_count": len(mask_files),
        "depth_count": len(depth_files),
        "matched_triplets": len(records),
        "unmatched": unmatched,
        "corrupted": corrupted,
        "aligned_count": sum(1 for r in records if r["aligned"]),
        "resolutions": dict(resolutions),
        "rgb_dtypes": dict(rgb_dtypes),
        "mask_dtypes": dict(mask_dtypes),
        "mask_unique_values": sorted(list(mask_unique_vals)),
        "depth_dtypes": dict(depth_dtypes),
        "depth_channels": {str(k): v for k, v in depth_channels.items()},
        "depth_all_channels_identical": all(r["depth_channels_identical"] for r in records),
        "depth_png_info_keys": sorted(list(png_info_keys)),
        "sessions": dict(sorted(sessions.items())),
        "records": records,
    }


def render_sample_visualization(rec: Dict[str, Any], idx: int):
    stem = rec["stem"]
    rp = os.path.join(DATASET_ROOT, "train", "rgb", rec["filename"])
    mp = os.path.join(DATASET_ROOT, "train", "mask", rec["filename"])
    dp = os.path.join(DATASET_ROOT, "train", "depth", rec["filename"])

    rgb_bgr = cv2.imread(rp, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    fg = mask > 0
    # Overlay RGB + Mask
    overlay = rgb.copy()
    tint = np.zeros_like(rgb)
    tint[fg] = [255, 60, 60]
    overlay = cv2.addWeighted(overlay, 0.72, tint, 0.28, 0)
    contours, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 180), 2)

    # Centroid for cross-section profiles
    ys, xs = np.where(fg)
    cy = int(np.mean(ys)) if len(ys) else depth.shape[0] // 2
    cx = int(np.mean(xs)) if len(xs) else depth.shape[1] // 2

    fig = plt.figure(figsize=(16.5, 9.5), dpi=140)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1.15], hspace=0.30, wspace=0.25)

    # 1. Original RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title(f"RGB ({rgb.shape[1]}x{rgb.shape[0]})", fontsize=10)
    ax1.axis("off")

    # 2. Segmentation Mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap="gray")
    ax2.set_title(f"Mask (Cov: {rec['fg_coverage_pct']}%, Comp: {rec['connected_components']})", fontsize=10)
    ax2.axis("off")

    # 3. RGB-Mask Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ax3.axhline(cy, color="cyan", linestyle="--", linewidth=0.9, alpha=0.8)
    ax3.axvline(cx, color="yellow", linestyle="--", linewidth=0.9, alpha=0.8)
    ax3.set_title("RGB + Mask Overlay & Profile Axes", fontsize=10)
    ax3.axis("off")

    # 4. Colorized Depth Map
    ax4 = fig.add_subplot(gs[0, 3])
    im = ax4.imshow(depth, cmap="turbo")
    ax4.axhline(cy, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.axvline(cx, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.set_title(f"Depth ({depth.dtype}, [{depth.min()}, {depth.max()}])", fontsize=10)
    ax4.axis("off")
    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.03)
    cbar.set_label("Native Pixel Unit", fontsize=8)

    # 5. Depth Histogram (Inside vs Outside Mask)
    ax5 = fig.add_subplot(gs[1, 0])
    in_vals = depth[fg]
    out_vals = depth[~fg]
    bins = np.linspace(0, 255, 48)
    if out_vals.size > 0:
        ax5.hist(out_vals, bins=bins, density=True, alpha=0.55, color="#64748b", label="Outside Mask")
    if in_vals.size > 0:
        ax5.hist(in_vals, bins=bins, density=True, alpha=0.65, color="#dc2626", label="Inside Mask")
    ax5.set_xlabel("Depth Value (Native uint8 [0-255])", fontsize=9)
    ax5.set_ylabel("Density", fontsize=9)
    ax5.set_title("Depth Distribution (In vs Out)", fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.3)

    # 6. Horizontal Profile (along y = cy)
    ax6 = fig.add_subplot(gs[1, 1])
    h_prof = depth[cy, :]
    h_mask = fg[cy, :]
    x_coords = np.arange(len(h_prof))
    ax6.plot(x_coords, h_prof, color="#0284c7", linewidth=1.3, label=f"Row y={cy}")
    if np.any(h_mask):
        ax6.fill_between(x_coords, 0, h_prof, where=h_mask, color="#dc2626", alpha=0.25, label="Mask Span")
    ax6.set_xlabel("X Pixel", fontsize=9)
    ax6.set_ylabel("Native Depth Intensity", fontsize=9)
    ax6.set_title("Horizontal Depth Profile", fontsize=10)
    ax6.set_ylim(-5, max(260, int(depth.max()) + 10))
    ax6.legend(fontsize=8)
    ax6.grid(True, linestyle="--", alpha=0.3)

    # 7. Vertical Profile (along x = cx)
    ax7 = fig.add_subplot(gs[1, 2])
    v_prof = depth[:, cx]
    v_mask = fg[:, cx]
    y_coords = np.arange(len(v_prof))
    ax7.plot(y_coords, v_prof, color="#d97706", linewidth=1.3, label=f"Col x={cx}")
    if np.any(v_mask):
        ax7.fill_between(y_coords, 0, v_prof, where=v_mask, color="#dc2626", alpha=0.25, label="Mask Span")
    ax7.set_xlabel("Y Pixel", fontsize=9)
    ax7.set_ylabel("Native Depth Intensity", fontsize=9)
    ax7.set_title("Vertical Depth Profile", fontsize=10)
    ax7.set_ylim(-5, max(260, int(depth.max()) + 10))
    ax7.legend(fontsize=8)
    ax7.grid(True, linestyle="--", alpha=0.3)

    # 8. 3D Depth Surface Visualization in native units
    ax8 = fig.add_subplot(gs[1, 3], projection="3d")
    ds_h, ds_w = 80, 120
    d_small = cv2.resize(depth, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
    X_grid, Y_grid = np.meshgrid(np.linspace(0, depth.shape[1], ds_w), np.linspace(0, depth.shape[0], ds_h))
    surf = ax8.plot_surface(X_grid, Y_grid, d_small, cmap="turbo", linewidth=0, antialiased=True)
    ax8.set_xlabel("X (px)", fontsize=7)
    ax8.set_ylabel("Y (px)", fontsize=7)
    ax8.set_zlabel("Native Unit", fontsize=7)
    ax8.set_title("3D Surface (Native Units)", fontsize=10)
    ax8.view_init(elev=38, azim=-55)

    fig.suptitle(f"Pothole440 Train Sample #{idx:02d}: {stem} (Session {rec['session_date']})", fontsize=12.5, fontweight="bold")
    out_path = os.path.join(OUT_DIR, f"sample_{idx:02d}_{stem}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def select_diverse_train_samples(train_records: List[Dict[str, Any]], count: int = 12) -> List[Dict[str, Any]]:
    # Group by session date and sort within each by fg_coverage_pct so we get diverse dates, sizes, & depths
    by_sess = defaultdict(list)
    for r in train_records:
        by_sess[r["session_date"]].append(r)

    selected = []
    sess_keys = sorted(by_sess.keys())
    # First pick one from each session date (alternating small, median, large coverage)
    for idx, sk in enumerate(sess_keys):
        items = sorted(by_sess[sk], key=lambda x: x["fg_coverage_pct"])
        if idx % 3 == 0:
            pick = items[0]
        elif idx % 3 == 1:
            pick = items[len(items) // 2]
        else:
            pick = items[-1]
        if pick not in selected:
            selected.append(pick)
        if len(selected) >= count:
            break

    # Fill remaining up to `count` from stratified coverage percentiles across remaining records
    if len(selected) < count:
        used_stems = {r["stem"] for r in selected}
        rem = sorted([r for r in train_records if r["stem"] not in used_stems], key=lambda x: (x["connected_components"], x["fg_coverage_pct"]))
        step = max(1, len(rem) // (count - len(selected)))
        for r in rem[::step]:
            if len(selected) >= count:
                break
            selected.append(r)

    return selected[:count]


def main():
    print("Auditing Pothole440 dataset at:", DATASET_ROOT)
    top_entries = sorted(os.listdir(DATASET_ROOT))
    has_annotations_dir = os.path.exists(os.path.join(DATASET_ROOT, "annotations"))

    split_data = {}
    all_records = []
    for sp in ("train", "val", "test"):
        res = audit_split(sp)
        split_data[sp] = res
        all_records.extend(res["records"])
        print(f"[{sp}] rgb={res['rgb_count']}, mask={res['mask_count']}, depth={res['depth_count']}, matched={res['matched_triplets']}, unmatched={len(res['unmatched'])}, corrupted={len(res['corrupted'])}, aligned={res['aligned_count']}")

    # Duplicate & Cross-split Leakage Audit
    rgb_sha_map = defaultdict(list)
    mask_sha_map = defaultdict(list)
    depth_sha_map = defaultdict(list)
    for r in all_records:
        rgb_sha_map[r["rgb_sha256"]].append(f"{r['split']}/{r['filename']}")
        mask_sha_map[r["mask_sha256"]].append(f"{r['split']}/{r['filename']}")
        depth_sha_map[r["depth_sha256"]].append(f"{r['split']}/{r['filename']}")

    exact_rgb_dups = [v for v in rgb_sha_map.values() if len(v) > 1]
    exact_mask_dups = [v for v in mask_sha_map.values() if len(v) > 1]
    exact_depth_dups = [v for v in depth_sha_map.values() if len(v) > 1]

    cross_near_dups = []
    within_near_dups = defaultdict(list)
    n = len(all_records)
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = all_records[i], all_records[j]
            dd = hamming(ri["dhash"], rj["dhash"])
            pd = hamming(ri["phash"], rj["phash"])
            if dd <= 6 or pd <= 6:
                pair = {
                    "a": f"{ri['split']}/{ri['filename']}",
                    "b": f"{rj['split']}/{rj['filename']}",
                    "dhash_dist": dd,
                    "phash_dist": pd,
                    "both_le_6": (dd <= 6 and pd <= 6),
                }
                if ri["split"] != rj["split"]:
                    cross_near_dups.append(pair)
                else:
                    within_near_dups[ri["split"]].append(pair)

    # Train/Val depth statistics summary (excluding test)
    tv_records = [r for r in all_records if r["split"] in ("train", "val")]
    tv_nonempty = [r for r in tv_records if r["depth_stats"]["inside_mean"] is not None]
    outside_zero_fracs = [r["depth_stats"]["outside_zero_frac"] for r in tv_records]
    inside_zero_fracs = [r["depth_stats"]["inside_zero_frac"] for r in tv_nonempty]
    inside_means = [r["depth_stats"]["inside_mean"] for r in tv_nonempty]
    outside_means = [r["depth_stats"]["outside_mean"] for r in tv_records]
    rim_medians = [r["depth_stats"]["annular_rim_median"] for r in tv_nonempty]
    inside_gt_outside = sum(1 for r in tv_nonempty if r["depth_stats"]["inside_mean"] > r["depth_stats"]["outside_mean"])
    inside_lt_outside = sum(1 for r in tv_nonempty if r["depth_stats"]["inside_mean"] < r["depth_stats"]["outside_mean"])
    outside_always_zero = sum(1 for r in tv_records if r["depth_stats"]["outside_zero_frac"] >= 0.99)
    outside_mostly_nonzero = sum(1 for r in tv_records if r["depth_stats"]["outside_zero_frac"] < 0.05)

    # Generate 12 diverse train visualizations (from non-empty masks so profiles/centroids exist)
    chosen = select_diverse_train_samples([r for r in split_data["train"]["records"] if r["connected_components"] > 0], count=12)
    vis_paths = []
    for idx, rec in enumerate(chosen, 1):
        vp = render_sample_visualization(rec, idx)
        vis_paths.append(vp)
    print(f"Rendered {len(vis_paths)} train sample visualizations to {OUT_DIR}")

    # Strip per-record heavy hash ints before saving JSON
    clean_splits = {}
    for sp, sdata in split_data.items():
        clean_splits[sp] = {k: v for k, v in sdata.items() if k != "records"}
        clean_splits[sp]["sample_count"] = len(sdata["records"])
        clean_splits[sp]["fg_coverage_pct_stats"] = {
            "min": round(min(r["fg_coverage_pct"] for r in sdata["records"]), 3),
            "median": round(float(np.median([r["fg_coverage_pct"] for r in sdata["records"]])), 3),
            "mean": round(float(np.mean([r["fg_coverage_pct"] for r in sdata["records"]])), 3),
            "max": round(max(r["fg_coverage_pct"] for r in sdata["records"]), 3),
        }
        clean_splits[sp]["connected_components_hist"] = dict(sorted(Counter(r["connected_components"] for r in sdata["records"]).items()))
        clean_splits[sp]["yolo_valid_contour_images"] = sum(1 for r in sdata["records"] if r["yolo_valid_contours"] > 0)
        clean_splits[sp]["depth_min_across_split"] = min(r["depth_min"] for r in sdata["records"])
        clean_splits[sp]["depth_max_across_split"] = max(r["depth_max"] for r in sdata["records"])

    # Compute session overlap across splits
    sess_train = set(split_data["train"]["sessions"].keys())
    sess_val = set(split_data["val"]["sessions"].keys())
    sess_test = set(split_data["test"]["sessions"].keys())

    inventory = {
        "dataset_root": DATASET_ROOT,
        "top_level_entries": top_entries,
        "annotations_dir_present": has_annotations_dir,
        "total_triplets": len(all_records),
        "splits": clean_splits,
        "depth_encoding_summary_train_val": {
            "images_analyzed": len(tv_records),
            "images_with_nonempty_mask": len(tv_nonempty),
            "images_with_empty_mask": [f"{r['split']}/{r['filename']}" for r in tv_records if r["depth_stats"]["inside_mean"] is None],
            "outside_zero_frac_min": round(min(outside_zero_fracs), 4),
            "outside_zero_frac_median": round(float(np.median(outside_zero_fracs)), 4),
            "outside_zero_frac_mean": round(float(np.mean(outside_zero_fracs)), 4),
            "outside_zero_frac_max": round(max(outside_zero_fracs), 4),
            "inside_zero_frac_median": round(float(np.median(inside_zero_fracs)), 4),
            "inside_mean_range": [round(min(inside_means), 2), round(max(inside_means), 2)],
            "inside_mean_median": round(float(np.median(inside_means)), 2),
            "outside_mean_range": [round(min(outside_means), 2), round(max(outside_means), 2)],
            "outside_mean_median": round(float(np.median(outside_means)), 2),
            "annular_rim_median_range": [round(min(rim_medians), 2), round(max(rim_medians), 2)],
            "count_inside_gt_outside": inside_gt_outside,
            "count_inside_lt_outside": inside_lt_outside,
            "count_outside_all_zero_ge99pct": outside_always_zero,
            "count_outside_mostly_nonzero_lt5pct_zero": outside_mostly_nonzero,
        },
        "quality_and_leakage_audit": {
            "exact_rgb_sha256_duplicates": exact_rgb_dups,
            "exact_mask_sha256_duplicates": exact_mask_dups,
            "exact_depth_sha256_duplicates": exact_depth_dups,
            "cross_split_perceptual_near_dups_count": len(cross_near_dups),
            "cross_split_both_dhash_phash_le_6": [p for p in cross_near_dups if p["both_le_6"]],
            "cross_split_perceptual_near_dups_sample": cross_near_dups[:20],
            "within_split_perceptual_near_dups_counts": {k: len(v) for k, v in within_near_dups.items()},
            "session_dates_train": sorted(list(sess_train)),
            "session_dates_val": sorted(list(sess_val)),
            "session_dates_test": sorted(list(sess_test)),
            "session_overlap_train_val": sorted(list(sess_train & sess_val)),
            "session_overlap_train_test": sorted(list(sess_train & sess_test)),
        },
        "visualized_train_samples": [
            {
                "index": i + 1,
                "stem": r["stem"],
                "session_date": r["session_date"],
                "fg_coverage_pct": r["fg_coverage_pct"],
                "connected_components": r["connected_components"],
                "depth_range": [r["depth_min"], r["depth_max"]],
                "depth_stats": r["depth_stats"],
                "plot_path": vis_paths[i],
            }
            for i, r in enumerate(chosen)
        ],
    }

    inv_path = os.path.join(OUT_DIR, "dataset_inventory.json")
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print("Saved inventory JSON:", inv_path)
    print("Depth encoding summary (train+val):", json.dumps(inventory["depth_encoding_summary_train_val"], indent=2))
    print("Quality & leakage:", json.dumps({
        "exact_rgb_dups": len(exact_rgb_dups),
        "exact_mask_dups": len(exact_mask_dups),
        "exact_depth_dups": len(exact_depth_dups),
        "cross_split_near_dups": len(cross_near_dups),
        "cross_split_both_le6": len(inventory["quality_and_leakage_audit"]["cross_split_both_dhash_phash_le_6"]),
        "within_split_near_dups": inventory["quality_and_leakage_audit"]["within_split_perceptual_near_dups_counts"],
        "session_overlap_train_val": inventory["quality_and_leakage_audit"]["session_overlap_train_val"],
        "session_overlap_train_test": inventory["quality_and_leakage_audit"]["session_overlap_train_test"],
    }, indent=2))


if __name__ == "__main__":
    main()
