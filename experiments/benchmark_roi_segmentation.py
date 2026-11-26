"""
========================================================================================
RoadVision AI -- Phase 4A.4: 3-Way Inference Comparison & RTX 2050 Performance Benchmark
========================================================================================
Compares three inference strategies across 36 diverse development frames (prioritizing
unseen dashcam/road video segments outside detector training):
  - Strategy A: Phase 2 Controlled YOLOv8n Detector Only
  - Strategy B: Phase 4A.2 Full-Frame YOLOv8n-seg Only
  - Strategy C: Phase 4A.4 Detector-Guided ROI Segmentation with Quality-Aware Fallback

Also benchmarks:
  - Sequential vs Batched ROI inference on RTX 2050 (4 GB VRAM)
  - Stage-by-stage latency breakdown (detector, ROI extract, seg ROI, reconstruct, QA)
  - Latency & throughput scaling versus number of detected potholes per frame
  - Verification detection comparison on existing bbox GT sets (Rangoon 35 & Phone val 14)

STRICT SCIENTIFIC RULES:
  - Does NOT touch or tune on Pothole440 grouped test split.
  - Does NOT compute mask IoU/Dice on dashcam frames lacking genuine pixel GT.
  - Does NOT equate mask acceptance rate with segmentation accuracy.
========================================================================================
"""

import os
import sys
import csv
import json
import time
import glob
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.roi_guided_segmentation import (
    ROISegConfig,
    ROIGuidedSegmenter,
    DEFAULT_DET_WEIGHTS,
    DEFAULT_SEG_WEIGHTS,
)
from experiments.evaluate_dashcam_transfer import eval_bbox_dataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "roi_seg_comparison_viz")

# 35 frame indices previously inspected in dataset_v2_candidates (avoided for unseen Rangoon dev set)
RANGOON_V2_INDICES = {
    0, 40, 61, 100, 120, 143, 182, 211, 255, 284, 313, 321, 370, 401, 422, 454, 481, 488,
    522, 549, 588, 625, 641, 676, 715, 721, 760, 803, 806, 844, 883, 892, 922, 963, 1002
}


def build_dev_frame_catalog() -> List[Dict[str, Any]]:
    """Curate 36 diverse development frames with full video/provenance metadata."""
    catalog = []

    # 1. Unseen Rural Dashcam Video (12 frames) — ZERO exposure in detector or seg training
    rural_frames = [
        (0, "Approach — single foreground depression"),
        (15, "Near road edge pothole + vegetation"),
        (30, "Multiple mid-field water puddles"),
        (45, "Sunlit gravel & muddy edge potholes"),
        (60, "Rural dashcam mid-range potholes & trees"),
        (75, "Close foreground water-filled cavities"),
        (90, "Dense multi-pothole rural damage cluster"),
        (105, "Small/distant & mid-field multi-potholes"),
        (120, "Roadside puddles + cracked mud surface"),
        (135, "Dense pothole cluster across lane width"),
        (150, "Sunlit water-filled potholes & horizon"),
        (180, "Far road damage cluster & shoulder grass"),
    ]
    for fnum, notes in rural_frames:
        catalog.append({
            "frame_id": f"dev_rural_f{fnum:04d}",
            "source_video": "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
            "session_id": "rural_road_unseen",
            "frame_number": fnum,
            "prior_detector_exposure": "UNSEEN",
            "prior_seg_exposure": "UNSEEN",
            "notes": notes,
        })

    # 2. Unseen Temporal Indices from Rangoon Road Video (12 frames strictly outside dataset_v2_candidates)
    rangoon_unseen = [
        (20, "Muddy water cavity & concrete patch contrast"),
        (80, "Multiple rough cavities & aggregate debris"),
        (160, "Severe mud/water pothole cluster"),
        (230, "Shadowed curb-side potholes & vehicle"),
        (270, "Multiple close & mid-field water cavities"),
        (340, "Road cracks & shallow patched surface"),
        (440, "Shadowed asphalt transition & longitudinal cracks"),
        (505, "Mixed asphalt patches & debris"),
        (605, "Forward corridor shadowed depressions"),
        (695, "Deep shadowed foreground cavity"),
        (740, "Water-filled cavity & pale wet reflections"),
        (780, "Shadowed twin craters & road shoulder"),
    ]
    for fnum, notes in rangoon_unseen:
        assert fnum not in RANGOON_V2_INDICES, f"Frame {fnum} overlaps dataset_v2_candidates!"
        catalog.append({
            "frame_id": f"dev_rangoon_f{fnum:04d}",
            "source_video": "Watch Video_ Terrible potholes in Rangoon Road.mp4",
            "session_id": "rangoon_road_unseen_temporal",
            "frame_number": fnum,
            "prior_detector_exposure": "UNSEEN_TEMPORAL_FRAME (35 other frames used as Phase 1/2 external eval)",
            "prior_seg_exposure": "UNSEEN",
            "notes": notes,
        })

    # 3. Survey Street Video raw_survey.mp4 (6 frames; 0 MD5 matches in train)
    survey_frames = [
        (0, "Urban asphalt cavity with curb/edge cues"),
        (20, "Oblique shadowed multi-pothole scene"),
        (40, "Low-angle flooded cavity & wet reflections"),
        (60, "Cracked wet asphalt depression"),
        (100, "Shallow road crack & shadow transition"),
        (120, "Multi-pothole urban street with traffic"),
    ]
    for fnum, notes in survey_frames:
        catalog.append({
            "frame_id": f"dev_survey_f{fnum:04d}",
            "source_video": "roadvision-dashboard/public/videos/raw_survey.mp4",
            "session_id": "raw_survey_video",
            "frame_number": fnum,
            "prior_detector_exposure": "UNSEEN_RAW_FRAMES (0 exact MD5 matches in train)",
            "prior_seg_exposure": "UNSEEN",
            "notes": notes,
        })

    # 4. Phone Validation Images (6 frames — explicitly flagged as unified provenance for Phase 2 detector)
    phone_imgs = [
        ("frame_0005.jpg", 5, "Oblique multi-pothole flooded street (8 GT boxes)"),
        ("frame_0014.jpg", 14, "Multiple puddles & distant cavities"),
        ("frame_0027.jpg", 27, "Large flooded foreground & mid cavities"),
        ("frame_0028.jpg", 28, "Foreground rimmed puddle + distant crater"),
        ("frame_0035.jpg", 35, "Oblique cracks & pale debris"),
        ("frame_0043.jpg", 43, "Dense gravel damage & puddles"),
    ]
    for fn, fnum, notes in phone_imgs:
        catalog.append({
            "frame_id": f"dev_phone_{os.path.splitext(fn)[0]}",
            "source_video": f"dataset_phone_potholes/images/val/{fn}",
            "session_id": "phone_val_session",
            "frame_number": fnum,
            "prior_detector_exposure": "PHONE_VAL_IN_UNIFIED_POOL",
            "prior_seg_exposure": "UNSEEN",
            "notes": notes,
        })

    return catalog


def load_frame(entry: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    src = os.path.join(BASE_DIR, entry["source_video"])
    if src.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, entry["frame_number"])
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed reading {src} frame {entry['frame_number']}")
        return frame, round(entry["frame_number"] / float(fps), 3)
    frame = cv2.imread(src)
    if frame is None:
        raise RuntimeError(f"Failed reading image {src}")
    return frame, -1.0


def run_full_frame_seg(seg_model, frame_bgr: np.ndarray, cfg: ROISegConfig, device: Any) -> Dict[str, Any]:
    H, W = frame_bgr.shape[:2]
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r0 = seg_model.predict(
        source=frame_bgr, imgsz=cfg.seg_imgsz, conf=cfg.seg_conf, iou=cfg.seg_iou,
        retina_masks=True, device=device, verbose=False,
    )[0]
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    union = np.zeros((H, W), dtype=bool)
    boxes_xyxy = []
    confs = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        boxes_xyxy = r0.boxes.xyxy.cpu().numpy().tolist()
        confs = [float(c) for c in r0.boxes.conf.cpu().numpy()]
        if r0.masks is not None and r0.masks.data is not None:
            m_np = r0.masks.data.cpu().numpy()
            for i in range(m_np.shape[0]):
                mi = m_np[i]
                if mi.shape[0] != H or mi.shape[1] != W:
                    mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_LINEAR)
                union |= (mi >= cfg.mask_bin_thresh)

    cov_pct = float(union.mean() * 100.0)
    return {
        "n_det": len(boxes_xyxy),
        "boxes_xyxy": boxes_xyxy,
        "confs": [round(c, 3) for c in confs],
        "union_mask": union,
        "coverage_pct": round(cov_pct, 3),
        "excessive_mask_over_35pct": cov_pct > 35.0,
        "latency_ms": round(dt_ms, 3),
    }


def render_4panel_comparison(
    entry: Dict[str, Any],
    ts: float,
    img_bgr: np.ndarray,
    full_seg_res: Dict[str, Any],
    roi_res: Dict[str, Any],
) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # Panel B: Strategy A (Detector Only)
    vis_det = rgb.copy()
    for p in roi_res["proposals"]:
        x1, y1, x2, y2 = [int(round(v)) for v in p["det_box_xyxy"]]
        cv2.rectangle(vis_det, (max(0, x1), max(0, y1)), (min(W - 1, x2), min(H - 1, y2)), (34, 197, 94), 2)
        cv2.putText(vis_det, f"{p['det_conf']:.2f}", (max(0, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 197, 94), 1, cv2.LINE_AA)

    # Panel C: Strategy B (Full-Frame YOLOv8n-seg)
    vis_full = rgb.copy()
    if full_seg_res["union_mask"].any():
        tint = vis_full.copy()
        tint[full_seg_res["union_mask"]] = [245, 158, 11]
        vis_full = cv2.addWeighted(vis_full, 0.58, tint, 0.42, 0)
        cnts, _ = cv2.findContours(full_seg_res["union_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_full, cnts, -1, (255, 220, 0), 2)
    for b, sc in zip(full_seg_res["boxes_xyxy"], full_seg_res["confs"]):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(vis_full, (max(0, x1), max(0, y1)), (min(W - 1, x2), min(H - 1, y2)), (56, 189, 248), 2)
        cv2.putText(vis_full, f"{sc:.2f}", (max(0, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)

    # Panel D: Strategy C (ROI-Guided Seg + Quality Fallback)
    vis_roi = rgb.copy()
    if roi_res["union_accepted_mask"].any():
        tint = vis_roi.copy()
        tint[roi_res["union_accepted_mask"]] = [16, 185, 129]  # Emerald green overlay
        vis_roi = cv2.addWeighted(vis_roi, 0.58, tint, 0.42, 0)
        cnts, _ = cv2.findContours(roi_res["union_accepted_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_roi, cnts, -1, (110, 255, 180), 2)

    for p in roi_res["proposals"]:
        x1, y1, x2, y2 = [int(round(v)) for v in p["det_box_xyxy"]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
        if p["accepted"]:
            cv2.rectangle(vis_roi, (x1, y1), (x2, y2), (34, 197, 94), 2)
            lbl = f"MASK {p['seg_conf']:.2f}"
            col = (34, 197, 94)
        else:
            cv2.rectangle(vis_roi, (x1, y1), (x2, y2), (255, 190, 40), 2)
            reason = p["rejection_reasons"][0].replace("_mask_prediction", "").replace("_", "-")[:14]
            lbl = f"FB:{reason}"
            col = (255, 190, 40)
        cv2.putText(vis_roi, lbl, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.1), dpi=130)
    ts_str = f"t={ts:.2f}s | " if ts >= 0 else ""
    s_c = roi_res["summary"]
    t_c = roi_res["timing_ms"]
    panels = [
        (rgb, f"Original ({ts_str}f={entry['frame_number']})"),
        (vis_det, f"A: Det Only (n={s_c['num_proposals']}, {t_c['detector_ms']:.1f}ms)"),
        (vis_full, f"B: Full-Frame Seg (n={full_seg_res['n_det']}, cov={full_seg_res['coverage_pct']:.1f}%)"),
        (vis_roi, f"C: ROI+QA (acc={s_c['accepted_masks']}, fb={s_c['bbox_fallbacks']}, {t_c['combined_total_ms']:.1f}ms)"),
    ]
    for ax, (im, ttl) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(ttl, fontsize=9.0)
        ax.axis("off")

    fig.suptitle(f"{entry['frame_id']} | {entry['source_video']} | {entry['notes']}", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    out_p = os.path.join(VIZ_DIR, f"{entry['frame_id']}.png")
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def compute_latency_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    mean_v = float(arr.mean()) if len(arr) else 0.0
    return {
        "mean_ms": round(mean_v, 2),
        "median_ms": round(float(np.median(arr)), 2) if len(arr) else 0.0,
        "p95_ms": round(float(np.percentile(arr, 95)), 2) if len(arr) else 0.0,
        "min_ms": round(float(arr.min()), 2) if len(arr) else 0.0,
        "max_ms": round(float(arr.max()), 2) if len(arr) else 0.0,
        "fps": round(1000.0 / mean_v, 2) if mean_v > 0 else 0.0,
    }


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    catalog = build_dev_frame_catalog()
    frames_loaded = []
    for entry in catalog:
        img, ts = load_frame(entry)
        frames_loaded.append((entry, ts, img))

    cfg_seq = ROISegConfig(batch_roi=False)
    cfg_bat = ROISegConfig(batch_roi=True, max_batch_size=8)

    system = ROIGuidedSegmenter(DEFAULT_DET_WEIGHTS, DEFAULT_SEG_WEIGHTS, config=cfg_seq)
    system.warmup(360, 640)

    # Multi-pass performance & accuracy pass (1 timed eval pass + 2 extra passes for stable latency quantiles)
    # First: Track Peak VRAM during Sequential vs Batched sweeps across all 36 frames
    def reset_vram():
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)

    def read_vram():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return (
                round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
                round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2),
            )
        return 0.0, 0.0

    # Strategy B (Full-frame seg) pass
    full_seg_results = []
    reset_vram()
    for _, _, img in frames_loaded:
        full_seg_results.append(run_full_frame_seg(system.seg_model, img, cfg_seq, system.device))
    vram_b_alloc, vram_b_res = read_vram()

    # Strategy C Sequential (first pass collects per-frame diagnostics + visualizations; passes 2-3 gather extra latency samples)
    reset_vram()
    seq_primary_outputs = []
    seq_timings = []
    for loop_i in range(3):
        for (entry, ts, img), f_res in zip(frames_loaded, full_seg_results):
            out = system.process_frame(img, cfg_override=cfg_seq)
            seq_timings.append({"n_prop": out["summary"]["num_proposals"], **out["timing_ms"]})
            if loop_i == 0:
                viz_p = render_4panel_comparison(entry, ts, img, f_res, out)
                seq_primary_outputs.append((entry, ts, img, f_res, out, viz_p))
    vram_seq_alloc, vram_seq_res = read_vram()

    # Strategy C Batched (3 passes across all 36 frames)
    reset_vram()
    bat_timings = []
    for _ in range(3):
        for _, _, img in frames_loaded:
            out_b = system.process_frame(img, cfg_override=cfg_bat)
            bat_timings.append({"n_prop": out_b["summary"]["num_proposals"], **out_b["timing_ms"]})
    vram_bat_alloc, vram_bat_res = read_vram()

    # Aggregate statistics
    per_frame_rows = []
    all_proposals_diag = []
    rej_counter = Counter()
    tot_prop = tot_acc = tot_rej = tot_empty = tot_fb = 0
    area_ratios_acc, area_ratios_rej = [], []
    inside_frac_acc, inside_frac_rej = [], []
    excessive_roi_raw = 0

    for entry, ts, img, f_res, out, viz_p in seq_primary_outputs:
        H, W = img.shape[:2]
        img_px = float(H * W)
        s = out["summary"]
        t = out["timing_ms"]
        tot_prop += s["num_proposals"]
        tot_acc += s["accepted_masks"]
        tot_rej += s["rejected_masks"]
        tot_empty += s["empty_masks"]
        tot_fb += s["bbox_fallbacks"]

        acc_cov = round(float(out["union_accepted_mask"].mean() * 100.0), 3)
        eff_cov = round(float(out["union_effective_mask"].mean() * 100.0), 3)

        for p in out["proposals"]:
            if p["accepted"]:
                area_ratios_acc.append(p["area_ratio_to_box"])
                inside_frac_acc.append(p["frac_inside_det_box"])
            else:
                area_ratios_rej.append(p["area_ratio_to_box"])
                inside_frac_rej.append(p["frac_inside_det_box"])
                for r in p["rejection_reasons"]:
                    rej_counter[r] += 1
                if "excessive_area_ratio" in p["rejection_reasons"] or "outside_detector_box_spill" in p["rejection_reasons"]:
                    excessive_roi_raw += 1

            all_proposals_diag.append({
                "frame_id": entry["frame_id"],
                "session_id": entry["session_id"],
                **{k: v for k, v in p.items() if k not in ("final_mask", "rejected_raw_mask")},
            })

        per_frame_rows.append({
            "frame_id": entry["frame_id"],
            "source_video": entry["source_video"],
            "session_id": entry["session_id"],
            "frame_number": entry["frame_number"],
            "timestamp_sec": ts,
            "resolution": f"{W}x{H}",
            "prior_detector_exposure": entry["prior_detector_exposure"],
            "prior_seg_exposure": entry["prior_seg_exposure"],
            "strat_a_det_count": s["num_proposals"],
            "strat_b_fullseg_count": f_res["n_det"],
            "strat_b_fullseg_cov_pct": f_res["coverage_pct"],
            "strat_b_excessive_gt35pct": f_res["excessive_mask_over_35pct"],
            "strat_b_latency_ms": f_res["latency_ms"],
            "strat_c_seg_attempts": s["seg_attempts"],
            "strat_c_accepted_masks": s["accepted_masks"],
            "strat_c_rejected_masks": s["rejected_masks"],
            "strat_c_empty_masks": s["empty_masks"],
            "strat_c_bbox_fallbacks": s["bbox_fallbacks"],
            "strat_c_detections_preserved": s["detections_preserved"],
            "strat_c_accepted_cov_pct": acc_cov,
            "strat_c_effective_cov_pct": eff_cov,
            "seq_detector_ms": t["detector_ms"],
            "seq_roi_extract_ms": t["roi_extract_ms"],
            "seq_seg_roi_ms": t["seg_roi_ms"],
            "seq_reconstruct_ms": t["mask_reconstruct_ms"],
            "seq_quality_ms": t["quality_eval_ms"],
            "seq_combined_ms": t["combined_total_ms"],
            "viz_path": viz_p,
        })

    # Write per-frame CSV
    csv_path = os.path.join(BENCH_DIR, "roi_segmentation_per_frame.csv")
    csv_cols = [
        "frame_id", "source_video", "session_id", "frame_number", "timestamp_sec", "resolution",
        "prior_detector_exposure", "prior_seg_exposure",
        "strat_a_det_count", "strat_b_fullseg_count", "strat_b_fullseg_cov_pct", "strat_b_excessive_gt35pct", "strat_b_latency_ms",
        "strat_c_seg_attempts", "strat_c_accepted_masks", "strat_c_rejected_masks", "strat_c_empty_masks",
        "strat_c_bbox_fallbacks", "strat_c_detections_preserved", "strat_c_accepted_cov_pct", "strat_c_effective_cov_pct",
        "seq_detector_ms", "seq_roi_extract_ms", "seq_seg_roi_ms", "seq_reconstruct_ms", "seq_quality_ms", "seq_combined_ms",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in per_frame_rows:
            w.writerow({k: r[k] for k in csv_cols})

    # Performance vs Number of Potholes per frame
    def bucket_scaling(timings_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        buckets = {"0": [], "1": [], "2": [], "3": [], "4+": []}
        for row in timings_list:
            n = row["n_prop"]
            key = "4+" if n >= 4 else str(n)
            buckets[key].append(row)
        out_b = {}
        for k, rows in buckets.items():
            if not rows:
                continue
            out_b[k] = {
                "samples": len(rows),
                "mean_n_prop": round(float(np.mean([r["n_prop"] for r in rows])), 2),
                "combined": compute_latency_stats([r["combined_total_ms"] for r in rows]),
                "seg_roi_only": compute_latency_stats([r["seg_roi_ms"] for r in rows]),
            }
        return out_b

    # Verified detection benchmarks (Box AP@50 & P/R/F1 for Strategy A vs Strategy B vs Strategy C)
    # Since Strategy C preserves 100% of Strategy A boxes, A and C have identical detection box metrics.
    rangoon_imgs = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_v2_candidates", "*.jpg")))
    phone_imgs = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_phone_potholes", "images", "val", "*.jpg")))
    det_verification = {
        "note": "Box-only detection verification on reviewed bbox GT sets; empty evaluation/ground_truth_labels excluded. Strategy C preserves 100% of Strategy A detector boxes.",
        "rangoon_35": {
            "strategy_a_and_c_detector": eval_bbox_dataset(system.det_model, rangoon_imgs, os.path.join(BASE_DIR, "dataset_v2_candidates", "labels"), system.device),
            "strategy_b_full_seg": eval_bbox_dataset(system.seg_model, rangoon_imgs, os.path.join(BASE_DIR, "dataset_v2_candidates", "labels"), system.device),
        },
        "phone_val_14": {
            "strategy_a_and_c_detector": eval_bbox_dataset(system.det_model, phone_imgs, os.path.join(BASE_DIR, "dataset_phone_potholes", "labels", "val"), system.device),
            "strategy_b_full_seg": eval_bbox_dataset(system.seg_model, phone_imgs, os.path.join(BASE_DIR, "dataset_phone_potholes", "labels", "val"), system.device),
        },
    }

    # Per-crop segmentation latency on non-empty proposals
    per_crop_seg_ms = [
        r["seg_roi_ms"] / r["n_prop"] for r in seq_timings if r["n_prop"] > 0
    ]

    non_zero_cov_b = [r["strat_b_fullseg_cov_pct"] for r in per_frame_rows if r["strat_b_fullseg_count"] > 0]
    non_zero_cov_c_acc = [r["strat_c_accepted_cov_pct"] for r in per_frame_rows if r["strat_c_accepted_masks"] > 0]

    report_json = {
        "experiment": "Phase 4A.4 Detector-Guided ROI Segmentation Comparison & Benchmark",
        "config": {k: v for k, v in vars(cfg_seq).items()},
        "dev_frames_count": len(per_frame_rows),
        "strategy_comparison_summary": {
            "strategy_a_total_detections": tot_prop,
            "strategy_b_total_detections": sum(r["strat_b_fullseg_count"] for r in per_frame_rows),
            "strategy_b_frames_with_any_detection": sum(1 for r in per_frame_rows if r["strat_b_fullseg_count"] > 0),
            "strategy_b_excessive_masks_gt35pct_frames": sum(1 for r in per_frame_rows if r["strat_b_excessive_gt35pct"]),
            "strategy_b_mean_cov_when_active_pct": round(float(np.mean(non_zero_cov_b)), 3) if non_zero_cov_b else 0.0,
            "strategy_c_proposals": tot_prop,
            "strategy_c_seg_attempts": tot_prop,
            "strategy_c_accepted_masks": tot_acc,
            "strategy_c_accepted_rate_pct": round(100.0 * tot_acc / max(1, tot_prop), 2),
            "strategy_c_rejected_masks": tot_rej,
            "strategy_c_rejected_rate_pct": round(100.0 * tot_rej / max(1, tot_prop), 2),
            "strategy_c_empty_masks": tot_empty,
            "strategy_c_empty_rate_pct": round(100.0 * tot_empty / max(1, tot_prop), 2),
            "strategy_c_bbox_fallbacks": tot_fb,
            "strategy_c_fallback_rate_pct": round(100.0 * tot_fb / max(1, tot_prop), 2),
            "strategy_c_detection_preservation_rate": 1.0,
            "strategy_c_excessive_raw_roi_masks": excessive_roi_raw,
            "strategy_c_accepted_excessive_gt35pct_frames": sum(1 for r in per_frame_rows if r["strat_c_accepted_cov_pct"] > 35.0),
            "strategy_c_mean_accepted_cov_when_active_pct": round(float(np.mean(non_zero_cov_c_acc)), 3) if non_zero_cov_c_acc else 0.0,
            "rejection_reason_counts": dict(rej_counter),
            "accepted_mask_geometry": {
                "mean_area_ratio_to_box": round(float(np.mean(area_ratios_acc)), 4) if area_ratios_acc else 0.0,
                "mean_frac_inside_det_box": round(float(np.mean(inside_frac_acc)), 4) if inside_frac_acc else 0.0,
            },
            "rejected_nonempty_mask_geometry": {
                "nonempty_count": len([x for x in area_ratios_rej if x > 0]),
                "mean_area_ratio_to_box_nonempty": round(float(np.mean([x for x in area_ratios_rej if x > 0])), 4) if any(x > 0 for x in area_ratios_rej) else 0.0,
            },
        },
        "verified_detection_comparison": det_verification,
        "performance_benchmark_rtx2050": {
            "sequential_roi": {
                "samples": len(seq_timings),
                "detector": compute_latency_stats([r["detector_ms"] for r in seq_timings]),
                "roi_extract": compute_latency_stats([r["roi_extract_ms"] for r in seq_timings]),
                "seg_roi_total": compute_latency_stats([r["seg_roi_ms"] for r in seq_timings]),
                "seg_per_crop": compute_latency_stats(per_crop_seg_ms),
                "mask_reconstruct": compute_latency_stats([r["mask_reconstruct_ms"] for r in seq_timings]),
                "quality_eval": compute_latency_stats([r["quality_eval_ms"] for r in seq_timings]),
                "combined_total": compute_latency_stats([r["combined_total_ms"] for r in seq_timings]),
                "peak_gpu_allocated_mb": vram_seq_alloc,
                "peak_gpu_reserved_mb": vram_seq_res,
                "scaling_by_num_proposals": bucket_scaling(seq_timings),
            },
            "batched_roi": {
                "samples": len(bat_timings),
                "detector": compute_latency_stats([r["detector_ms"] for r in bat_timings]),
                "roi_extract": compute_latency_stats([r["roi_extract_ms"] for r in bat_timings]),
                "seg_roi_total": compute_latency_stats([r["seg_roi_ms"] for r in bat_timings]),
                "mask_reconstruct": compute_latency_stats([r["mask_reconstruct_ms"] for r in bat_timings]),
                "quality_eval": compute_latency_stats([r["quality_eval_ms"] for r in bat_timings]),
                "combined_total": compute_latency_stats([r["combined_total_ms"] for r in bat_timings]),
                "peak_gpu_allocated_mb": vram_bat_alloc,
                "peak_gpu_reserved_mb": vram_bat_res,
                "scaling_by_num_proposals": bucket_scaling(bat_timings),
            },
            "strategy_b_full_frame_seg": {
                "samples": len(full_seg_results),
                "combined_total": compute_latency_stats([r["latency_ms"] for r in full_seg_results]),
                "peak_gpu_allocated_mb": vram_b_alloc,
                "peak_gpu_reserved_mb": vram_b_res,
            },
        },
        "per_frame_summary": per_frame_rows,
        "per_proposal_diagnostics": all_proposals_diag,
    }

    json_path = os.path.join(BENCH_DIR, "roi_segmentation_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    print("=== ROI SEGMENTATION BENCHMARK COMPLETE ===")
    print("Strategy Summary:", json.dumps(report_json["strategy_comparison_summary"], indent=2))
    print("Seq Combined:", report_json["performance_benchmark_rtx2050"]["sequential_roi"]["combined_total"])
    print("Batched Combined:", report_json["performance_benchmark_rtx2050"]["batched_roi"]["combined_total"])


if __name__ == "__main__":
    main()
