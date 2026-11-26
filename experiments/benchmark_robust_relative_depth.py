"""
========================================================================================
RoadVision AI -- Phase 5A: Dashcam Dev, Temporal, Pothole440 & Performance Benchmark
========================================================================================
Evaluates all 5 relative-depth reference-surface estimation strategies:
  A. median_baseline   B. existing_annular   C. ols_plane   D. ransac_plane   E. huber_irls_plane

Using identical cached Phase 2 YOLOv8n proposals and MiDaS_small depth predictions:
  1. 36 Phase 4A.4 development frames (98 candidate regions): validity, rejection reasons,
     inlier ratios, reference & cavity residual statistics, and extraction timing.
  2. Sensitivity analyses: annular margin {0.25, 0.45, 0.70}, box perturbation (±8%),
     and synthetic outlier contamination injected onto real MiDaS frames.
  3. Within-track temporal relative-depth stability on `ruralRoad_potHoles.mp4`
     (disclaimer: lower variance != higher physical accuracy).
  4. Pothole440 permitted `val` (15 originals) Go!SCAN valid-region & ordinal depth audit.
  5. RTX 2050 stage-by-stage latency, VRAM, and proposal-scaling benchmark.
  6. 9-panel diagnostic visualizations (both valid and rejected/invalid examples).
========================================================================================
"""

import os
import sys
import csv
import json
import time
from collections import Counter
from typing import List, Dict, Any, Tuple

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.roi_guided_segmentation import DEFAULT_DET_WEIGHTS
from experiments.benchmark_roi_segmentation import (
    build_dev_frame_catalog,
    load_frame,
    compute_latency_stats,
)
from experiments.robust_relative_depth import (
    ALL_METHODS,
    RobustDepthConfig,
    MiDaSDepthRunner,
    extract_relative_depth_for_box,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "robust_depth_viz")


def render_9panel_diagnostic(
    save_name: str,
    title_prefix: str,
    img_bgr: np.ndarray,
    depth_map: np.ndarray,
    box: Tuple[float, float, float, float],
    neighbors: List[Tuple[float, float, float, float]],
    res_dict: Dict[str, Any],
) -> str:
    arr = res_dict["arrays"]
    reg = arr["region"]
    fit = arr["fit_out"]
    res_map = arr["signed_residual_map"]

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = reg["box_xyxy"]
    rx1, ry1, rx2, ry2 = reg["outer_xyxy"]
    h_out, w_out = ry2 - ry1, rx2 - rx1

    # 2. Boxes panel
    vis_boxes = rgb.copy()
    for nb in neighbors:
        nx1, ny1, nx2, ny2 = [int(round(v)) for v in nb]
        cv2.rectangle(vis_boxes, (max(0, nx1), max(0, ny1)), (min(W - 1, nx2), min(H - 1, ny2)), (249, 115, 22), 2)
    cv2.rectangle(vis_boxes, (rx1, ry1), (min(W - 1, rx2), min(H - 1, ry2)), (56, 189, 248), 2)
    cv2.rectangle(vis_boxes, (x1, y1), (min(W - 1, x2), min(H - 1, y2)), (34, 197, 94), 2)

    # 4 & 5: Annular sampling & inliers/outliers overlay on cropped outer RGB
    outer_rgb = rgb[ry1:ry2, rx1:rx2].copy()
    ann_vis = (outer_rgb * 0.45).astype(np.uint8)
    ann_vis[reg["valid_ring_mask"]] = [56, 189, 248]

    inl_vis = (outer_rgb * 0.45).astype(np.uint8)
    ys, xs = reg["ys_local"], reg["xs_local"]
    inl = fit["inlier_mask"]
    if len(ys) > 0 and len(inl) == len(ys):
        inl_vis[ys[inl], xs[inl]] = [34, 197, 94]     # Green inliers
        inl_vis[ys[~inl], xs[~inl]] = [239, 68, 68]   # Red rejected outliers

    # 6. Reconstruct reference plane across full outer patch
    from experiments.robust_relative_depth import _normalize_coords
    coeffs = fit["coeffs"]
    if res_dict["method"] in ("ols_plane", "ransac_plane", "huber_irls_plane"):
        oy, ox = np.indices((h_out, w_out))
        yn, xn = _normalize_coords(oy, ox, h_out, w_out)
        ref_outer = coeffs[0] * yn + coeffs[1] * xn + coeffs[2]
    elif res_dict["method"] == "existing_annular":
        oy, ox = np.indices((h_out, w_out))
        ref_outer = coeffs[0] * oy + coeffs[1] * ox + coeffs[2]
    else:
        ref_outer = np.full((h_out, w_out), coeffs[2], dtype=np.float64)

    fig = plt.figure(figsize=(16.5, 11.2), dpi=130)
    gs = GridSpec(3, 3, figure=fig, hspace=0.28, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title("1. Original Road Frame", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(vis_boxes)
    ax2.set_title("2. Target Box (Green), Outer (Cyan), Neighbors (Orange)", fontsize=9.2)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(depth_map, cmap="inferno")
    ax3.set_title("3. MiDaS Rel. Inverse Depth (Higher=Closer)", fontsize=9.5)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(ann_vis)
    ax4.set_title(f"4. Annular Ring (n={reg['valid_ref_samples']}, Q={reg['populated_quadrants']}/4)", fontsize=9.5)
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(inl_vis)
    ax5.set_title(f"5. Ref Samples (Green=Inlier {fit['inlier_ratio']*100:.1f}%, Red=Outlier)", fontsize=9.2)
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(ref_outer, cmap="inferno")
    ax6.set_title("6. Fitted Ref Surface (Rel. Inv-Depth)", fontsize=9.5)
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.03)
    ax6.axis("off")

    ax7 = fig.add_subplot(gs[2, 0])
    vmax = max(1e-2, float(np.nanpercentile(np.abs(res_map), 98))) if np.isfinite(res_map).any() else 1.0
    im7 = ax7.imshow(res_map, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax7.set_title("7. Signed Residual (Ref - Depth; + = Deeper)", fontsize=9.2)
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.03)
    ax7.axis("off")

    ax8 = fig.add_subplot(gs[2, 1])
    fin = res_map[np.isfinite(res_map)]
    if len(fin) > 0:
        ax8.hist(fin.ravel(), bins=28, color="#3b82f6", alpha=0.85, edgecolor="white")
        ax8.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="0 Ref")
        ax8.axvline(res_dict["relative_cavity_stats"]["raw_diagnostic_p75"], color="#ef4444", linewidth=1.6, label="P75")
        ax8.legend(fontsize=8)
    ax8.set_title("8. Cavity Signed Residual Distribution (Rel. Units)", fontsize=9.2)
    ax8.set_xlabel("Relative Inverse-Depth Residual", fontsize=8.5)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    st = res_dict["relative_cavity_stats"]
    rd = res_dict["ref_diagnostics"]
    verdict = "VALID" if res_dict["is_valid"] else "INVALID"
    color_v = "#16a34a" if res_dict["is_valid"] else "#dc2626"
    lines = [
        f"Method: {res_dict['method']}",
        f"Status: {verdict} (Heur Score: {res_dict['heuristic_reliability_score']:.3f})",
        f"Water Suspected: {res_dict['is_water_suspected']}",
        f"Failure Reasons: {', '.join(res_dict['failure_reasons']) or 'None'}",
        f"Warnings: {', '.join(res_dict['warnings']) or 'None'}",
        "",
        f"Ref Samples: {rd['valid_ref_samples']} (Inlier: {rd['inlier_ratio']:.3f})",
        f"Ref MAD / RMSE: {rd['ref_fit_mad']:.3f} / {rd['ref_fit_rmse']:.3f}",
        f"Diag Median / IQR: {st['raw_diagnostic_median']:.3f} / {st['iqr_residual'] if st['iqr_residual'] is not None else 'N/A (Invalid)'}",
        f"Diag P75: {st['raw_diagnostic_p75']:.3f} | PosFrac: {st['positive_depression_frac']:.3f}",
        "(Note: Relative MiDaS units ONLY; not cm)",
    ]
    ax9.text(0.03, 0.95, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor=color_v, linewidth=2))
    ax9.set_title("9. Quality Verdict & Diagnostics", fontsize=9.5)

    fig.suptitle(title_prefix, fontsize=11.5, fontweight="bold")
    out_p = os.path.join(VIZ_DIR, save_name)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def run_temporal_stability_check(det_model: YOLO, midas: MiDaSDepthRunner, cfg: RobustDepthConfig) -> Dict[str, Any]:
    """Measure within-track relative-depth CV across ByteTrack IDs on ruralRoad_potHoles.mp4 (first 120 frames)."""
    vpath = os.path.join(BASE_DIR, "uploads", "upload_1794847200_ruralRoad_potHoles.mp4")
    cap = cv2.VideoCapture(vpath)
    tracks: Dict[int, Dict[str, List[float]]] = {}
    frames_checked = 0
    try:
        while frames_checked < 120:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            res = det_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.5, imgsz=640, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0 and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                tids = res.boxes.id.int().cpu().tolist()
                dmap, _ = midas.predict(frame)
                for idx, (box, tid) in enumerate(zip(boxes, tids)):
                    nbrs = [tuple(float(v) for v in boxes[j]) for j in range(len(boxes)) if j != idx]
                    if tid not in tracks:
                        tracks[tid] = {m: [] for m in ALL_METHODS}
                    for m in ALL_METHODS:
                        out = extract_relative_depth_for_box(
                            dmap, tuple(float(v) for v in box), method=m, neighbor_boxes=nbrs, img_bgr=frame, cfg=cfg
                        )
                        if out["is_valid"] and out["relative_cavity_stats"]["p75_residual"] is not None:
                            tracks[tid][m].append(out["relative_cavity_stats"]["p75_residual"])
            frames_checked += 1
    finally:
        cap.release()

    method_cvs = {m: [] for m in ALL_METHODS}
    method_valid_obs = {m: 0 for m in ALL_METHODS}
    for tid, mdict in tracks.items():
        for m, vals in mdict.items():
            method_valid_obs[m] += len(vals)
            if len(vals) >= 4 and np.mean(vals) > 1e-4:
                cv = float(np.std(vals, ddof=1) / np.mean(vals))
                method_cvs[m].append(cv)

    return {
        "video": "uploads/upload_1794847200_ruralRoad_potHoles.mp4 (frames 0..119)",
        "unique_track_ids_seen": len(tracks),
        "note": "Lower within-track CV reflects temporal consistency of relative P75 residuals ONLY, NOT physical depth accuracy.",
        "methods": {
            m: {
                "valid_observations": method_valid_obs[m],
                "tracks_with_ge4_valid_obs": len(method_cvs[m]),
                "mean_within_track_cv": round(float(np.mean(method_cvs[m])), 4) if method_cvs[m] else None,
                "median_within_track_cv": round(float(np.median(method_cvs[m])), 4) if method_cvs[m] else None,
            }
            for m in ALL_METHODS
        },
    }


def run_pothole440_feasibility_check(midas: MiDaSDepthRunner) -> Dict[str, Any]:
    """
    Task 8: Investigate permitted Pothole440 `val` (15 originals, NEVER `test`) Go!SCAN valid-region
    identification and strictly bounded within-image ordinal checks.
    """
    split_csv = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "manifests", "split_manifest.csv")
    src_root = os.path.join(BASE_DIR, "Pothole440-main", "Pothole440-main")
    rows = [r for r in csv.DictReader(open(split_csv, encoding="utf-8")) if r["disposition"] == "INCLUDED_VAL"]

    per_sample = []
    spearman_rhos = []
    valid_polarity_agree = 0
    severely_cropped_annular = 0

    for r in rows:
        fn = r["filename"]
        sp = r["orig_pothole440_split"]
        rgb_bgr = cv2.imread(os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "images", "val", fn))
        d_u8 = cv2.imread(os.path.join(src_root, sp, "depth", fn), cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.imread(os.path.join(src_root, sp, "mask", fn), cv2.IMREAD_UNCHANGED) > 0

        # Border-connected dominant constant padding (preserves genuine interior scanner pixels sharing dom_val)
        vals, cnts = np.unique(d_u8, return_counts=True)
        dom_val = int(vals[np.argmax(cnts)])
        is_dom = (d_u8 == dom_val).astype(np.uint8)
        _, lbls, _, _ = cv2.connectedComponentsWithStats(is_dom, connectivity=8)
        border_lbls = set(np.unique(np.concatenate([lbls[0, :], lbls[-1, :], lbls[:, 0], lbls[:, -1]]))) - {0}
        pad_mask = np.isin(lbls, list(border_lbls))
        valid_scan = ~pad_mask

        ys, xs = np.where(gt_mask)
        y1, y2, x1, x2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        bw, bh = x2 - x1, y2 - y1
        my, mx = max(20, int(0.35 * bh)), max(20, int(0.35 * bw))
        ry1, ry2 = max(0, y1 - my), min(d_u8.shape[0], y2 + my)
        rx1, rx2 = max(0, x1 - mx), min(d_u8.shape[1], x2 + mx)
        ann_mask = np.zeros_like(gt_mask, dtype=bool)
        ann_mask[ry1:ry2, rx1:rx2] = True
        ann_mask[y1:y2, x1:x2] = False

        ann_in_pad_pct = float(pad_mask[ann_mask].mean() * 100.0) if ann_mask.sum() > 0 else 100.0
        if ann_in_pad_pct > 25.0:
            severely_cropped_annular += 1

        midas_map, _ = midas.predict(rgb_bgr)
        # Inside valid_scan: compare midas inverse depth vs negated Go!SCAN (-d_u8, since higher uint8 = deeper)
        valid_crater = gt_mask & valid_scan
        valid_road = ann_mask & valid_scan

        rho = None
        if valid_scan.sum() >= 100:
            # Subsample uniformly for fast deterministic tie-aware Spearman correlation
            vy, vx = np.where(valid_scan)
            step = max(1, len(vy) // 5000)
            xs = midas_map[vy[::step], vx[::step]].astype(np.float64)
            ys = -d_u8[vy[::step], vx[::step]].astype(np.float64)

            def _avg_ranks(arr: np.ndarray) -> np.ndarray:
                order = np.argsort(arr, kind="mergesort")
                sorted_vals = arr[order]
                n = len(arr)
                ranks = np.empty(n, dtype=np.float64)
                i = 0
                while i < n:
                    j = i + 1
                    while j < n and sorted_vals[j] == sorted_vals[i]:
                        j += 1
                    avg_r = 0.5 * (i + j - 1) + 1.0
                    ranks[order[i:j]] = avg_r
                    i = j
                return ranks

            rx = _avg_ranks(xs) - 0.5 * (len(xs) + 1.0)
            ry = _avg_ranks(ys) - 0.5 * (len(ys) + 1.0)
            denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
            r_val = float(np.sum(rx * ry) / denom) if denom > 1e-12 else float("nan")
            if np.isfinite(r_val):
                rho = round(float(r_val), 4)
                spearman_rhos.append(rho)

        # Signed ordering: Go!SCAN valid_crater deeper than valid_road vs MiDaS valid_crater deeper than valid_road
        gs_deeper = None
        md_deeper = None
        agree = False
        if valid_crater.sum() >= 20 and valid_road.sum() >= 20:
            gs_deeper = bool(np.median(d_u8[valid_crater]) > np.median(d_u8[valid_road]))
            md_deeper = bool(np.median(midas_map[valid_crater]) < np.median(midas_map[valid_road]))
            agree = (gs_deeper == md_deeper)
            valid_polarity_agree += int(agree)

        per_sample.append({
            "filename": fn,
            "dom_pad_val": dom_val,
            "border_pad_pct": round(float(pad_mask.mean() * 100.0), 2),
            "interior_matching_dom_val_pct": round(float(((d_u8 == dom_val) & valid_scan).mean() * 100.0), 2),
            "annular_ring_in_pad_pct": round(ann_in_pad_pct, 2),
            "valid_road_ring_px": int(valid_road.sum()),
            "spearman_rho_valid_scan": rho,
            "goscan_cavity_deeper_than_valid_ring": gs_deeper,
            "midas_cavity_deeper_than_valid_ring": md_deeper,
            "ordinal_ordering_agree": agree,
        })

    return {
        "split_inspected": "dataset_pothole440_seg_grouped/val (15 unaugmented originals; test NEVER touched)",
        "samples_count": len(rows),
        "samples_with_gt25pct_annular_in_pad": severely_cropped_annular,
        "valid_ordinal_ordering_agreement": f"{valid_polarity_agree}/{len(rows)}",
        "mean_spearman_rho_valid_scan": round(float(np.mean(spearman_rhos)), 4) if spearman_rhos else None,
        "median_spearman_rho_valid_scan": round(float(np.median(spearman_rhos)), 4) if spearman_rhos else None,
        "physical_residual_comparison_status": "BLOCKED",
        "blocking_reasons": [
            "8-bit quantized Go!SCAN maps lack per-scan min/max physical scaling metadata and camera intrinsics.",
            f"{severely_cropped_annular}/15 validation samples have >25% of the surrounding road ring missing inside constant padding.",
            "Border-connected flood fill separates outer padding without erasing valid interior pixels sharing dom_val, but cannot reconstruct missing surrounding pavement outside the scan crop.",
        ],
        "per_sample": per_sample,
    }


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = YOLO(DEFAULT_DET_WEIGHTS)
    midas = MiDaSDepthRunner(device=device_str)

    # Warmup
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    _ = det_model.predict(dummy, imgsz=640, conf=0.25, iou=0.70, device=0 if device_str == "cuda" else "cpu", verbose=False)
    midas.warmup(360, 640)

    catalog = build_dev_frame_catalog()
    cfg = RobustDepthConfig()

    # Cache identical detector proposals + MiDaS depth maps across all 36 frames (3 timing loops for det & midas)
    cached_frames = []
    det_times_ms = []
    midas_times_ms = []

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)

    for loop_i in range(3):
        for entry in catalog:
            img_bgr, ts = load_frame(entry)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            td0 = time.perf_counter()
            det_res = det_model.predict(
                img_bgr, imgsz=640, conf=0.25, iou=0.70, device=0 if device_str == "cuda" else "cpu", verbose=False
            )[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            det_ms = (time.perf_counter() - td0) * 1000.0
            det_times_ms.append(det_ms)

            boxes = det_res.boxes.xyxy.cpu().numpy().tolist() if (det_res.boxes is not None and len(det_res.boxes) > 0) else []
            confs = det_res.boxes.conf.cpu().numpy().tolist() if (det_res.boxes is not None and len(det_res.boxes) > 0) else []

            if len(boxes) > 0:
                dmap, m_ms = midas.predict(img_bgr)
                midas_times_ms.append(m_ms)
            else:
                dmap, m_ms = None, 0.0
                midas_times_ms.append(0.0)

            if loop_i == 0:
                cached_frames.append({
                    "entry": entry,
                    "ts": ts,
                    "img_bgr": img_bgr,
                    "boxes": boxes,
                    "confs": confs,
                    "depth_map": dmap,
                    "det_ms": det_ms,
                    "midas_ms": m_ms,
                })

    peak_backbone_alloc_mb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0
    peak_backbone_res_mb = round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0

    # Evaluate all 5 reference strategies on cached frames (3 extraction passes for stable CPU timing)
    method_summaries = {}
    per_region_csv_rows = []
    scaling_by_prop = {m: {"1": [], "2": [], "3": [], "4+": []} for m in ALL_METHODS}

    # Select representative visualization cases dynamically after pass 1
    viz_candidates = {}

    for m in ALL_METHODS:
        valid_cnt = 0
        total_cnt = 0
        fail_counter = Counter()
        ref_mads, ref_rmses, inl_ratios = [], [], []
        cav_meds, cav_p75s, cav_p90s, cav_iqrs, cav_mads, pos_fracs = [], [], [], [], [], []
        stage_ann, stage_fit, stage_res, stage_qa, stage_tot = [], [], [], [], []
        frame_ext_times = { "1": [], "2": [], "3": [], "4+": [] }

        for loop_i in range(3):
            for cf in cached_frames:
                boxes = cf["boxes"]
                if not boxes or cf["depth_map"] is None:
                    continue
                dmap = cf["depth_map"]
                img_bgr = cf["img_bgr"]
                frame_t = 0.0
                for idx, (box, dconf) in enumerate(zip(boxes, cf["confs"])):
                    nbrs = [tuple(boxes[j]) for j in range(len(boxes)) if j != idx]
                    out = extract_relative_depth_for_box(
                        dmap, tuple(box), method=m, neighbor_boxes=nbrs, img_bgr=img_bgr, cfg=cfg,
                        return_arrays=(loop_i == 0 and m in ("ransac_plane", "huber_irls_plane", "existing_annular"))
                    )
                    tm = out["timing_ms"]
                    stage_ann.append(tm["annular_region_ms"])
                    stage_fit.append(tm["ref_fit_ms"])
                    stage_res.append(tm["residual_compute_ms"])
                    stage_qa.append(tm["quality_eval_ms"])
                    stage_tot.append(tm["extraction_total_ms"])
                    frame_t += tm["extraction_total_ms"]

                    if loop_i == 0:
                        total_cnt += 1
                        rd = out["ref_diagnostics"]
                        st = out["relative_cavity_stats"]
                        ref_mads.append(rd["ref_fit_mad"])
                        ref_rmses.append(rd["ref_fit_rmse"])
                        inl_ratios.append(rd["inlier_ratio"])
                        pos_fracs.append(st["positive_depression_frac"])
                        if out["is_valid"]:
                            valid_cnt += 1
                            cav_meds.append(st["median_residual"])
                            cav_p75s.append(st["p75_residual"])
                            cav_p90s.append(st["p90_residual"])
                            cav_iqrs.append(st["iqr_residual"])
                            cav_mads.append(st["mad_residual"])
                        else:
                            for r in out["failure_reasons"]:
                                fail_counter[r] += 1

                        per_region_csv_rows.append({
                            "frame_id": cf["entry"]["frame_id"],
                            "source_video": cf["entry"]["source_video"],
                            "frame_number": cf["entry"]["frame_number"],
                            "region_idx": idx,
                            "det_conf": round(dconf, 4),
                            "method": m,
                            "is_valid": out["is_valid"],
                            "failure_reasons": "|".join(out["failure_reasons"]),
                            "warnings": "|".join(out["warnings"]),
                            "heuristic_reliability": out["heuristic_reliability_score"],
                            "is_water_suspected": out["is_water_suspected"],
                            "valid_ref_samples": rd["valid_ref_samples"],
                            "populated_quadrants": rd["populated_quadrants"],
                            "neighbor_excluded_px": rd["neighbor_excluded_px"],
                            "inlier_ratio": rd["inlier_ratio"],
                            "ref_fit_mad": rd["ref_fit_mad"],
                            "ref_fit_rmse": rd["ref_fit_rmse"],
                            "p75_residual_valid": st["p75_residual"],
                            "p90_residual_valid": st["p90_residual"],
                            "median_residual_valid": st["median_residual"],
                            "iqr_residual_valid": st["iqr_residual"],
                            "mad_residual_valid": st["mad_residual"],
                            "raw_diag_p75": st["raw_diagnostic_p75"],
                            "positive_depression_frac": st["positive_depression_frac"],
                            "extraction_ms": tm["extraction_total_ms"],
                        })

                        if m == "ransac_plane":
                            key = (cf["entry"]["frame_id"], idx)
                            viz_candidates[key] = (cf, idx, tuple(box), nbrs, out)

                bkey = "4+" if len(boxes) >= 4 else str(len(boxes))
                frame_ext_times[bkey].append(frame_t)

        method_summaries[m] = {
            "total_candidate_regions": total_cnt,
            "valid_count": valid_cnt,
            "valid_coverage_pct": round(100.0 * valid_cnt / max(1, total_cnt), 2),
            "invalid_count": total_cnt - valid_cnt,
            "failure_reason_counts": dict(fail_counter),
            "ref_fit_mad_mean": round(float(np.mean(ref_mads)), 4),
            "ref_fit_mad_median": round(float(np.median(ref_mads)), 4),
            "ref_fit_rmse_mean": round(float(np.mean(ref_rmses)), 4),
            "inlier_ratio_mean": round(float(np.mean(inl_ratios)), 4),
            "inlier_ratio_median": round(float(np.median(inl_ratios)), 4),
            "valid_cavity_p75_median": round(float(np.median(cav_p75s)), 4) if cav_p75s else None,
            "valid_cavity_p75_iqr": round(float(np.percentile(cav_p75s, 75) - np.percentile(cav_p75s, 25)), 4) if cav_p75s else None,
            "valid_cavity_p90_median": round(float(np.median(cav_p90s)), 4) if cav_p90s else None,
            "valid_cavity_med_median": round(float(np.median(cav_meds)), 4) if cav_meds else None,
            "positive_depress_frac_mean": round(float(np.mean(pos_fracs)), 4),
            "per_region_timing_ms": {
                "annular_region": compute_latency_stats(stage_ann),
                "ref_fit": compute_latency_stats(stage_fit),
                "residual_compute": compute_latency_stats(stage_res),
                "quality_eval": compute_latency_stats(stage_qa),
                "extraction_total": compute_latency_stats(stage_tot),
            },
            "frame_extraction_scaling_ms": {
                k: compute_latency_stats(v) for k, v in frame_ext_times.items() if v
            },
        }

    # Sensitivity experiments on the 36 cached frames (margin, box perturbation, reference contamination)
    sensitivity = {"annular_margin": {}, "box_perturbation_8pct": {}, "synthetic_ring_contamination": {}}
    for m in ("ols_plane", "ransac_plane", "huber_irls_plane"):
        # Baseline P75 map
        base_vals = {}
        for cf in cached_frames:
            if cf["depth_map"] is None: continue
            for idx, box in enumerate(cf["boxes"]):
                nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
                o = extract_relative_depth_for_box(cf["depth_map"], tuple(box), method=m, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=cfg)
                base_vals[(cf["entry"]["frame_id"], idx)] = o

        # 1. Margin sweep (0.25 vs 0.70 vs 0.45)
        m_sens = {}
        for mr in (0.25, 0.70):
            cfg_m = RobustDepthConfig(outer_margin_ratio=mr)
            rel_diffs = []
            both_valid = 0
            new_valid = 0
            for cf in cached_frames:
                if cf["depth_map"] is None: continue
                for idx, box in enumerate(cf["boxes"]):
                    nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
                    o = extract_relative_depth_for_box(cf["depth_map"], tuple(box), method=m, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=cfg_m)
                    if o["is_valid"]: new_valid += 1
                    ob = base_vals[(cf["entry"]["frame_id"], idx)]
                    if ob["is_valid"] and o["is_valid"] and abs(ob["relative_cavity_stats"]["p75_residual"]) > 1e-3:
                        both_valid += 1
                        rel_diffs.append(abs(o["relative_cavity_stats"]["p75_residual"] - ob["relative_cavity_stats"]["p75_residual"]) / abs(ob["relative_cavity_stats"]["p75_residual"]))
            m_sens[f"margin_{mr}"] = {
                "valid_count": new_valid,
                "median_rel_p75_shift_pct": round(100.0 * float(np.median(rel_diffs)), 2) if rel_diffs else None,
            }
        sensitivity["annular_margin"][m] = m_sens

        # 2. Bounding-box perturbation (±8% shifts/scales)
        rng_p = np.random.default_rng(42)
        p_diffs = []
        p_valid_retained = 0
        p_base_valid = 0
        for cf in cached_frames:
            if cf["depth_map"] is None: continue
            H, W = cf["depth_map"].shape[:2]
            for idx, box in enumerate(cf["boxes"]):
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                dx1, dx2 = rng_p.uniform(-0.08 * bw, 0.08 * bw, size=2)
                dy1, dy2 = rng_p.uniform(-0.08 * bh, 0.08 * bh, size=2)
                pbox = (max(0, box[0] + dx1), max(0, box[1] + dy1), min(W, box[2] + dx2), min(H, box[3] + dy2))
                nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
                o = extract_relative_depth_for_box(cf["depth_map"], pbox, method=m, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=cfg)
                ob = base_vals[(cf["entry"]["frame_id"], idx)]
                if ob["is_valid"]:
                    p_base_valid += 1
                    if o["is_valid"]:
                        p_valid_retained += 1
                        if abs(ob["relative_cavity_stats"]["p75_residual"]) > 1e-3:
                            p_diffs.append(abs(o["relative_cavity_stats"]["p75_residual"] - ob["relative_cavity_stats"]["p75_residual"]) / abs(ob["relative_cavity_stats"]["p75_residual"]))
        sensitivity["box_perturbation_8pct"][m] = {
            "base_valid": p_base_valid,
            "perturbed_valid_retained": p_valid_retained,
            "median_rel_p75_shift_pct": round(100.0 * float(np.median(p_diffs)), 2) if p_diffs else None,
        }

        # 3. Real-MiDaS reference contamination (+200 spike strip on 25% of top outer border)
        c_diffs = []
        c_valid = 0
        for cf in cached_frames:
            if cf["depth_map"] is None: continue
            for idx, box in enumerate(cf["boxes"]):
                ob = base_vals[(cf["entry"]["frame_id"], idx)]
                if not ob["is_valid"]: continue
                d_cor = cf["depth_map"].copy()
                ox1, oy1, ox2, oy2 = ob["outer_xyxy"]
                bx1, by1, bx2, by2 = ob["box_xyxy"]
                top_h = max(0, by1 - oy1)
                if top_h >= 4:
                    d_cor[oy1 : oy1 + max(2, int(0.8 * top_h)), ox1:ox2] += float(np.nanstd(d_cor) * 4.0 + 50.0)
                nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
                o = extract_relative_depth_for_box(d_cor, tuple(box), method=m, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=cfg)
                if o["is_valid"]: c_valid += 1
                if abs(ob["relative_cavity_stats"]["p75_residual"]) > 1e-3:
                    c_diffs.append(abs(o["relative_cavity_stats"]["raw_diagnostic_p75"] - ob["relative_cavity_stats"]["p75_residual"]) / abs(ob["relative_cavity_stats"]["p75_residual"]))
        sensitivity["synthetic_ring_contamination"][m] = {
            "valid_retained_out_of": len(c_diffs),
            "valid_retained": c_valid,
            "median_rel_p75_error_pct": round(100.0 * float(np.median(c_diffs)), 2) if c_diffs else None,
        }

    # Write per-region CSV
    csv_path = os.path.join(BENCH_DIR, "robust_relative_depth_per_region.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_region_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(per_region_csv_rows)

    # Render 8 diverse 9-panel diagnostic visualizations (both valid and invalid failures)
    selected_viz_keys = []
    valid_keys = [k for k, v in viz_candidates.items() if v[4]["is_valid"]]
    water_keys = [k for k, v in viz_candidates.items() if "suspected_water_reflection" in v[4]["failure_reasons"]]
    pol_keys = [k for k, v in viz_candidates.items() if "unexpected_residual_polarity" in v[4]["failure_reasons"]]
    nbr_keys = [k for k, v in viz_candidates.items() if v[4]["ref_diagnostics"]["neighbor_excluded_px"] > 0 and v[4]["is_valid"]]

    for lst, take in [(valid_keys, 3), (nbr_keys, 1), (water_keys, 2), (pol_keys, 2)]:
        for k in lst:
            if k not in selected_viz_keys:
                selected_viz_keys.append(k)
                take -= 1
                if take <= 0:
                    break

    viz_paths = []
    for k in selected_viz_keys[:8]:
        cf, idx, box, nbrs, out = viz_candidates[k]
        fid = cf["entry"]["frame_id"]
        status_tag = "valid" if out["is_valid"] else "invalid"
        vp = render_9panel_diagnostic(
            save_name=f"{fid}_r{idx}_{status_tag}.png",
            title_prefix=f"Phase 5A Diagnostic | {fid} (Region #{idx}) | {cf['entry']['source_video']} | {status_tag.upper()}",
            img_bgr=cf["img_bgr"],
            depth_map=cf["depth_map"],
            box=box,
            neighbors=nbrs,
            res_dict=out,
        )
        viz_paths.append(vp)

    # Temporal & Pothole440 checks
    temporal_res = run_temporal_stability_check(det_model, midas, cfg)
    p440_res = run_pothole440_feasibility_check(midas)

    # Full report JSON
    midas_active_times = [t for t in midas_times_ms if t > 0]
    results_json = {
        "experiment": "Phase 5A Robust Relative Depth Extraction & Adaptive Reference Benchmark",
        "startup_midas_load_ms_separate": midas.startup_load_ms,
        "config": vars(cfg),
        "dev_frames_count": len(cached_frames),
        "backbone_performance_rtx2050": {
            "detector_640_ms": compute_latency_stats(det_times_ms),
            "midas_small_active_ms": compute_latency_stats(midas_active_times),
            "midas_small_amortized_all_frames_ms": compute_latency_stats(midas_times_ms),
            "peak_gpu_allocated_mb": peak_backbone_alloc_mb,
            "peak_gpu_reserved_mb": peak_backbone_res_mb,
        },
        "method_comparison_dev_36_frames": method_summaries,
        "sensitivity_analyses": sensitivity,
        "temporal_stability_rural_video": temporal_res,
        "pothole440_val_feasibility": p440_res,
        "diagnostic_viz_paths": viz_paths,
    }

    json_path = os.path.join(BENCH_DIR, "robust_relative_depth_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print("=== PHASE 5A BENCHMARK COMPLETE ===")
    print("Backbone det:", results_json["backbone_performance_rtx2050"]["detector_640_ms"])
    print("Backbone midas active:", results_json["backbone_performance_rtx2050"]["midas_small_active_ms"])
    for m, s in method_summaries.items():
        print(f"  {m:18s} | valid={s['valid_count']}/{s['total_candidate_regions']} ({s['valid_coverage_pct']:5.1f}%) | mad={s['ref_fit_mad_median']:.4f} | inl={s['inlier_ratio_median']:.3f} | reg_ms={s['per_region_timing_ms']['extraction_total']['mean_ms']:.2f}")
    print("Pothole440 val feasibility:", {k: p440_res[k] for k in ("samples_with_gt25pct_annular_in_pad", "valid_ordinal_ordering_agreement", "mean_spearman_rho_valid_scan", "physical_residual_comparison_status")})


if __name__ == "__main__":
    main()
