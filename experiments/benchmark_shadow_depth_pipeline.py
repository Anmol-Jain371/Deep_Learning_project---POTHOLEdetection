"""
========================================================================================
RoadVision AI -- Phase 5C: Shadow-Mode Integration & Production Readiness Benchmark
========================================================================================
Evaluates:
  1. Preprocessing 2x2 ablation: `bgr_legacy` vs `rgb_corrected` x `legacy_annular` vs
     `optimized_ransac` on 36 cached dev frames (Task 3).
  2. Multi-video shadow-mode A/B/C/D evaluation across 4 contiguous sequences (`567` frames):
       - ruralRoad_potHoles.mp4 (202f, 360p@25fps, unseen)
       - Rangoon Road (180f, 360p@25fps, 6/180 isolated frames in v2 candidates)
       - raw_survey.mp4 (160f, 720p@10fps, 0 MD5 matches in train)
       - test_clip.mp4 (25f, 360p@25fps)
  3. 11-scenario temporal lifecycle & failure injection test harness (Task 6).
  4. Optional adaptive 512 -> 1024 sampling ablation on synthetic + video streams (Task 7).
  5. Separate un-instrumented end-to-end runtime benchmark (decode, det+track, preprocess,
     MiDaS, extraction sub-stages, temporal filtering, serialization, rendering) across
     0, 1, 2, 3, and 4+ candidate frames + shadow overhead vs standalone pipeline (Task 8).
  6. Comparative plots + 2 side-by-side research-only MP4 demo videos (Task 10 & 12).

Zero production modifications; all experimental metrics strictly in relative units.
========================================================================================
"""

import os
import sys
import csv
import json
import time
import tracemalloc
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.roi_guided_segmentation import DEFAULT_DET_WEIGHTS
from experiments.benchmark_roi_segmentation import (
    RANGOON_V2_INDICES,
    build_dev_frame_catalog,
    load_frame,
    compute_latency_stats,
)
from experiments.robust_relative_depth import RobustDepthConfig, extract_relative_depth_for_box
from experiments.optimized_relative_depth import (
    OptimizedDepthConfig,
    extract_optimized_relative_depth_for_box,
)
from experiments.validity_aware_temporal_filter import TemporalFilterConfig
from experiments.test_relative_depth_synthetic import make_plane, elliptical_cavity
from experiments.benchmark_optimized_relative_depth import (
    _build_randomized_regression_scenarios,
    _spearman_rho,
    _guarded_rel_diff_pct,
    _summarize_vals,
)
from experiments.shadow_depth_pipeline import (
    ShadowDepthPipeline,
    SharedMiDaSRunner,
    LegacyMethodRunner,
    should_escalate_budget,
)
from roadvision_engine import AnnularDonutDepthExtractor, SpecularAquaplaneDetector

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "phase5c_shadow_viz")

VIDEO_SEQUENCES = [
    {
        "seq_id": "ruralRoad_202f",
        "path": "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
        "start_frame": 0,
        "max_frames": 202,
        "prior_exposure": "UNSEEN in detector/seg training (used for Phase 4A.4/5A/5B dev analysis)",
        "conditions": "Sparse & dense potholes, grass verges, puddles, approach/departure, track gaps",
    },
    {
        "seq_id": "rangoonRoad_180f",
        "path": "Watch Video_ Terrible potholes in Rangoon Road.mp4",
        "start_frame": 0,
        "max_frames": 180,
        "prior_exposure": "6/180 isolated frames (0,40,61,100,120,143) in dataset_v2_candidates; 174 unseen temporal frames",
        "conditions": "Heavy shadows, curb contamination, puddles, multi-pothole urban clusters",
    },
    {
        "seq_id": "rawSurvey_720p_160f",
        "path": "roadvision-dashboard/public/videos/raw_survey.mp4",
        "start_frame": 0,
        "max_frames": 160,
        "prior_exposure": "0 exact MD5 matches in detector training; 720p urban dev sequence",
        "conditions": "1280x720@10fps resolution test, oblique shadows, wet asphalt, traffic clutter",
    },
    {
        "seq_id": "testClip_25f",
        "path": "uploads/upload_1794546659_test_clip.mp4",
        "start_frame": 0,
        "max_frames": 25,
        "prior_exposure": "Development upload clip",
        "conditions": "Short contiguous sanity clip",
    },
]


def run_preprocessing_ablation(
    det_model: YOLO,
    midas: SharedMiDaSRunner,
) -> Dict[str, Any]:
    """Task 3: 2x2 factorial ablation (bgr_legacy vs rgb_corrected) x (legacy_annular vs optimized_ransac)."""
    catalog = build_dev_frame_catalog()
    opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    base_cfg = RobustDepthConfig()

    pre_ms_bgr, pre_ms_rgb, inf_ms_bgr, inf_ms_rgb = [], [], [], []
    depth_l1_diffs, depth_rmse_diffs, depth_pearsons, depth_spearmans = [], [], [], []
    cached = []

    for entry in catalog:
        img_bgr, _ = load_frame(entry)
        r = det_model.predict(img_bgr, imgsz=640, conf=0.25, iou=0.70, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if (r.boxes is not None and len(r.boxes) > 0) else []
        if not boxes:
            continue
        out_bgr = midas.predict(img_bgr, preprocess_mode="bgr_legacy")
        out_rgb = midas.predict(img_bgr, preprocess_mode="rgb_corrected")
        pre_ms_bgr.append(out_bgr["preprocess_ms"])
        pre_ms_rgb.append(out_rgb["preprocess_ms"])
        inf_ms_bgr.append(out_bgr["inference_ms"])
        inf_ms_rgb.append(out_rgb["inference_ms"])

        db, dr = out_bgr["depth_map"].astype(np.float64), out_rgb["depth_map"].astype(np.float64)
        diff = dr - db
        depth_l1_diffs.append(float(np.mean(np.abs(diff))))
        depth_rmse_diffs.append(float(np.sqrt(np.mean(diff ** 2))))
        fb, fr = db.ravel()[::16], dr.ravel()[::16]
        if np.std(fb) > 1e-8 and np.std(fr) > 1e-8:
            depth_pearsons.append(float(np.corrcoef(fb, fr)[0, 1]))
        sp = _spearman_rho(fb, fr)
        if sp is not None:
            depth_spearmans.append(sp)

        cached.append({"fid": entry["frame_id"], "img": img_bgr, "boxes": boxes, "d_bgr": out_bgr["depth_map"], "d_rgb": out_rgb["depth_map"]})

    cells = {
        "bgr_legacy_plus_legacy_annular": ("d_bgr", "legacy"),
        "rgb_corrected_plus_legacy_annular": ("d_rgb", "legacy"),
        "bgr_legacy_plus_optimized_ransac": ("d_bgr", "opt"),
        "rgb_corrected_plus_optimized_ransac": ("d_rgb", "opt"),
    }
    cell_outputs: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {k: {} for k in cells}
    for cname, (dkey, ext_kind) in cells.items():
        for cf in cached:
            dmap = cf[dkey]
            for idx, box in enumerate(cf["boxes"]):
                nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
                if ext_kind == "legacy":
                    res = extract_relative_depth_for_box(dmap, tuple(box), method="existing_annular", neighbor_boxes=nbrs, img_bgr=cf["img"], cfg=base_cfg)
                else:
                    res = extract_optimized_relative_depth_for_box(dmap, tuple(box), neighbor_boxes=nbrs, img_bgr=cf["img"], cfg=opt_cfg)
                cell_outputs[cname][(cf["fid"], idx)] = res

    keys = list(cell_outputs["bgr_legacy_plus_legacy_annular"].keys())
    common_all4 = [k for k in keys if all(cell_outputs[c][k]["is_valid"] for c in cells)]
    summary_cells = {}
    for cname in cells:
        val_keys = [k for k in keys if cell_outputs[cname][k]["is_valid"]]
        summary_cells[cname] = {
            "total_regions": len(keys),
            "valid_count": len(val_keys),
            "valid_pct": round(100.0 * len(val_keys) / max(1, len(keys)), 2),
            "admitted_ref_mad": _summarize_vals([cell_outputs[cname][k]["ref_diagnostics"]["ref_fit_mad"] for k in val_keys]),
            "admitted_p75": _summarize_vals([cell_outputs[cname][k]["relative_cavity_stats"]["p75_residual"] for k in val_keys]),
            "common_valid_count": len(common_all4),
            "common_ref_mad": _summarize_vals([cell_outputs[cname][k]["ref_diagnostics"]["ref_fit_mad"] for k in common_all4]),
            "common_p75": _summarize_vals([cell_outputs[cname][k]["relative_cavity_stats"]["p75_residual"] for k in common_all4]),
            "mean_extraction_ms": round(float(np.mean([cell_outputs[cname][k]["timing_ms"]["extraction_total_ms"] for k in keys])), 4),
        }

    # Direct preprocessing deltas holding extractor fixed
    pair_leg = [k for k in keys if cell_outputs["bgr_legacy_plus_legacy_annular"][k]["is_valid"] and cell_outputs["rgb_corrected_plus_legacy_annular"][k]["is_valid"]]
    pair_opt = [k for k in keys if cell_outputs["bgr_legacy_plus_optimized_ransac"][k]["is_valid"] and cell_outputs["rgb_corrected_plus_optimized_ransac"][k]["is_valid"]]
    leg_p75_diffs = [abs(cell_outputs["rgb_corrected_plus_legacy_annular"][k]["relative_cavity_stats"]["p75_residual"] - cell_outputs["bgr_legacy_plus_legacy_annular"][k]["relative_cavity_stats"]["p75_residual"]) for k in pair_leg]
    opt_p75_diffs = [abs(cell_outputs["rgb_corrected_plus_optimized_ransac"][k]["relative_cavity_stats"]["p75_residual"] - cell_outputs["bgr_legacy_plus_optimized_ransac"][k]["relative_cavity_stats"]["p75_residual"]) for k in pair_opt]

    return {
        "frames_evaluated": len(cached),
        "total_regions": len(keys),
        "midas_timing_ms": {
            "preprocess_bgr_mean_ms": round(float(np.mean(pre_ms_bgr)), 4),
            "preprocess_rgb_mean_ms": round(float(np.mean(pre_ms_rgb)), 4),
            "inference_bgr_mean_ms": round(float(np.mean(inf_ms_bgr)), 4),
            "inference_rgb_mean_ms": round(float(np.mean(inf_ms_rgb)), 4),
        },
        "depth_map_divergence_bgr_vs_rgb": {
            "mean_l1": round(float(np.mean(depth_l1_diffs)), 4),
            "mean_rmse": round(float(np.mean(depth_rmse_diffs)), 4),
            "mean_pearson_r": round(float(np.mean(depth_pearsons)), 4),
            "mean_spearman_rho": round(float(np.mean(depth_spearmans)), 4),
        },
        "factorial_2x2": summary_cells,
        "paired_preprocessing_effect": {
            "legacy_extractor_pair_count": len(pair_leg),
            "legacy_abs_p75_shift_median": round(float(np.median(leg_p75_diffs)), 4) if leg_p75_diffs else None,
            "opt_ransac_pair_count": len(pair_opt),
            "opt_ransac_abs_p75_shift_median": round(float(np.median(opt_p75_diffs)), 4) if opt_p75_diffs else None,
        },
        "note": "Channel order correction aligns input with ImageNet RGB weights; without physical ground truth, depthmap shift alone is NOT proof of metric accuracy gain.",
    }


def _make_demo_frame_side_by_side(
    img_bgr: np.ndarray,
    frame_res: Dict[str, Any],
    seq_name: str,
    f_idx: int,
    det_ms: float,
    midas_ms: float,
) -> np.ndarray:
    """Render Task 10 research-only left (Legacy A) vs right (Optimized RANSAC + RollingMed C) video frame."""
    H, W = img_bgr.shape[:2]
    left = img_bgr.copy()
    right = img_bgr.copy()

    res_a_list = frame_res["results_by_method"]["A_legacy_annular_ema"]
    res_c_list = frame_res["results_by_method"]["C_opt_ransac_rollmed_ema"]
    arr_list = frame_res["opt_arrays_by_box"]

    # Left panel: Legacy RoadVision extraction (raw disp / smoothed legacy)
    for ra in res_a_list:
        x1, y1, x2, y2 = [int(round(v)) for v in ra.box_xyxy]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        mx, my = max(15, min(25, int(bw * 0.25))), max(15, min(25, int(bh * 0.25)))
        rx1, ry1, rx2, ry2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
        cv2.rectangle(left, (rx1, ry1), (rx2, ry2), (180, 180, 180), 1)
        col = (0, 220, 0) if ra.is_valid else (0, 0, 220)
        cv2.rectangle(left, (x1, y1), (x2, y2), col, 2)
        tag = f"#{ra.track_id} {ra.status} | raw={ra.raw_relative_p75:.1f} legEMA={ra.smoothed_relative_score:.1f}"
        cv2.putText(left, tag, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    # Right panel: Optimized RANSAC + Rolling Median Validity-Aware Filter (RELATIVE UNITS ONLY)
    for idx, rc in enumerate(res_c_list):
        x1, y1, x2, y2 = [int(round(v)) for v in rc.box_xyxy]
        arr = arr_list[idx] if idx < len(arr_list) else None
        if arr is not None:
            ox1, oy1, ox2, oy2 = arr["region"]["outer_xyxy"]
            cv2.rectangle(right, (ox1, oy1), (ox2, oy2), (255, 200, 50), 1)
        color_map = {"VALID": (34, 197, 94), "STALE": (0, 195, 255), "EXPIRED": (80, 80, 235), "INVALID": (60, 60, 240), "NOT_EVALUATED": (180, 180, 180)}
        col = color_map.get(rc.status, (255, 255, 255))
        cv2.rectangle(right, (x1, y1), (x2, y2), col, 2)
        s_txt = f"{rc.smoothed_relative_score:.1f}" if rc.smoothed_relative_score is not None else "None"
        r_txt = f"{rc.raw_relative_p75:.1f}" if rc.raw_relative_p75 is not None else "None"
        fail_txt = rc.failure_reasons[0] if rc.failure_reasons else ("water" if rc.water_glare_warning else "ok")
        tag1 = f"#{rc.track_id} {rc.status} | relP75={r_txt} sm={s_txt} st={rc.stale_age_frames}"
        tag2 = f"{fail_txt} | {rc.extraction_latency_ms:.1f}ms"
        cv2.putText(right, tag1, (x1, max(16, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        cv2.putText(right, tag2, (x1, max(28, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1, cv2.LINE_AA)

    # Top banners cleanly distinguishing methods & NO CM/MM units
    cv2.rectangle(left, (0, 0), (W, 28), (20, 24, 30), -1)
    cv2.rectangle(right, (0, 0), (W, 28), (20, 24, 30), -1)
    cv2.putText(left, f"LEFT: Legacy Annular+EMA | {seq_name} f{f_idx} | A={frame_res['timings_ms']['A_ms']:.1f}ms", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(right, f"RIGHT: Opt RANSAC+RollMed (REL UNITS, NO CM) | C={frame_res['timings_ms']['C_ms']:.1f}ms", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 255, 160), 1, cv2.LINE_AA)
    return np.hstack([left, right])


def run_video_shadow_and_runtime(
    det_model: YOLO,
    midas: SharedMiDaSRunner,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]:
    """Tasks 5, 7 (real video adaptive), 8 (standalone vs shadow runtime), 9 & 10."""
    pipe = ShadowDepthPipeline(
        camera_profile="sedan_dashcam",
        opt_cfg=OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512),
        filter_cfg=TemporalFilterConfig(alpha=0.35, rolling_window=5, max_stale_frames=3, max_missed_frames_reset=8),
        enable_adaptive_variant=True,
        preprocess_mode="rgb_corrected",
    )

    all_methods = ("A_legacy_annular_ema", "B_opt_ransac_valid_ema", "C_opt_ransac_rollmed_ema", "D_opt_ransac_adaptive_1024")
    per_frame_csv_rows = []
    track_store: Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]] = {m: {} for m in all_methods}
    per_video_stats: Dict[str, Any] = {}
    all_obs_by_method: Dict[str, Dict[Tuple[str, int, int], Dict[str, Any]]] = {m: {} for m in all_methods}
    obs_meta: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    # Runtime scaling & stage timing accumulators (isolated passes cleanly timed)
    scaling_shadow_ms = {"0": [], "1": [], "2": [], "3": [], "4+": []}
    scaling_standalone_opt_ms = {"0": [], "1": [], "2": [], "3": [], "4+": []}
    scaling_legacy_ms = {"0": [], "1": [], "2": [], "3": [], "4+": []}
    stage_decode_ms, stage_det_track_ms, stage_pre_ms, stage_midas_ms = [], [], [], []
    stage_render_ms, stage_log_ms = [], []
    # Standalone CPU extraction stage timing
    sa_ann_ms, sa_fit_ms, sa_res_ms, sa_qa_ms, sa_flt_ms = [], [], [], [], []
    adaptive_escalations = 0
    total_video_regions = 0
    demo_video_paths = []
    cached_rural_for_runtime = []

    for seq in VIDEO_SEQUENCES:
        seq_id = seq["seq_id"]
        vpath = os.path.join(BASE_DIR, seq["path"])
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq["start_frame"])

        det_model.predictor = None  # Reset Ultralytics ByteTrack state per distinct video sequence
        pipe.reset_all_tracks()

        writer = None
        if seq_id in ("ruralRoad_202f", "rangoonRoad_180f"):
            out_v = os.path.join(VIZ_DIR, f"shadow_demo_{seq_id}.mp4")
            writer = cv2.VideoWriter(out_v, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2, H))
            demo_video_paths.append(out_v)

        seq_frames = 0
        seq_dets = 0
        seq_m_valid = {m: 0 for m in all_methods}
        seq_m_status = {m: Counter() for m in all_methods}
        seq_t_wall0 = time.perf_counter()

        for offset in range(seq["max_frames"]):
            td0 = time.perf_counter()
            ok, frame = cap.read()
            t_dec = (time.perf_counter() - td0) * 1000.0
            if not ok or frame is None:
                break
            f_idx = seq["start_frame"] + offset
            ts = round(f_idx / fps, 4)
            stage_decode_ms.append(t_dec)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            ty0 = time.perf_counter()
            res = det_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.50, imgsz=640, verbose=False)[0]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_dt = (time.perf_counter() - ty0) * 1000.0
            stage_det_track_ms.append(t_dt)

            boxes, confs, tids = [], [], []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy().tolist()]
                confs = [float(c) for c in res.boxes.conf.cpu().numpy().tolist()]
                if res.boxes.id is not None:
                    tids = [int(t) for t in res.boxes.id.int().cpu().tolist()]
                else:
                    tids = [i + 1 for i in range(len(boxes))]

            if boxes:
                mp = midas.predict(frame, preprocess_mode="rgb_corrected")
                dmap = mp["depth_map"]
                t_pre, t_inf = mp["preprocess_ms"], mp["inference_ms"]
            else:
                dmap = None
                t_pre, t_inf = 0.0, 0.0
            stage_pre_ms.append(t_pre)
            stage_midas_ms.append(t_inf)

            fid = f"{seq_id}_f{f_idx:04d}"
            t_sh0 = time.perf_counter()
            out_sh = pipe.process_frame(
                frame_id=fid,
                frame_idx=f_idx,
                timestamp_sec=ts,
                img_bgr=frame,
                depth_map=dmap,
                boxes_xyxy=boxes,
                confs=confs,
                track_ids=tids,
                return_opt_arrays=(writer is not None),
            )
            t_sh_ms = (time.perf_counter() - t_sh0) * 1000.0

            # Log CSV row and track aggregates cleanly
            tl0 = time.perf_counter()
            for idx_b, (bx, cf_v, tid) in enumerate(zip(boxes, confs, tids)):
                total_video_regions += 1
                okey = (seq_id, f_idx, tid)
                obs_meta[okey] = {"box_area": max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1])), "frame_idx": f_idx}
                for m in all_methods:
                    item = out_sh["results_by_method"][m][idx_b]
                    seq_m_status[m][item.status] += 1
                    if item.is_valid:
                        seq_m_valid[m] += 1
                    if m == "D_opt_ransac_adaptive_1024" and item.adaptive_budget_used == 1024:
                        adaptive_escalations += 1
                    d_rec = item.to_dict()
                    all_obs_by_method[m][okey] = d_rec
                    track_store[m].setdefault((seq_id, tid), []).append(d_rec)
            t_log_ms = (time.perf_counter() - tl0) * 1000.0
            stage_log_ms.append(t_log_ms)

            # Frame-level scaling stats
            n_b = len(boxes)
            bk = "4+" if n_b >= 4 else str(n_b)
            tm_f = out_sh["timings_ms"]
            t_standalone_opt = t_dec + t_dt + t_pre + t_inf + tm_f["water_ms"] + (tm_f["C_ms"] if n_b > 0 else 0.0)
            t_standalone_leg = t_dec + t_dt + t_pre + t_inf + tm_f["water_ms"] + tm_f["A_ms"]
            t_full_shadow = t_dec + t_dt + t_pre + t_inf + t_sh_ms
            scaling_shadow_ms[bk].append(t_full_shadow)
            scaling_standalone_opt_ms[bk].append(t_standalone_opt)
            scaling_legacy_ms[bk].append(t_standalone_leg)

            c_list = out_sh["results_by_method"]["C_opt_ransac_rollmed_ema"]
            a_list = out_sh["results_by_method"]["A_legacy_annular_ema"]
            per_frame_csv_rows.append({
                "seq_id": seq_id,
                "frame_idx": f_idx,
                "timestamp_sec": ts,
                "width": W,
                "height": H,
                "num_candidates": n_b,
                "decode_ms": round(t_dec, 3),
                "det_track_ms": round(t_dt, 3),
                "midas_pre_ms": round(t_pre, 3),
                "midas_inf_ms": round(t_inf, 3),
                "legacy_A_ms": tm_f["A_ms"],
                "opt_B_ms": tm_f["B_ms"],
                "opt_C_ms": tm_f["C_ms"],
                "A_valid": sum(1 for r_a in a_list if r_a.is_valid),
                "C_valid": sum(1 for r_c in c_list if r_c.status == "VALID"),
                "C_stale": sum(1 for r_c in c_list if r_c.status == "STALE"),
                "C_expired": sum(1 for r_c in c_list if r_c.status == "EXPIRED"),
                "C_invalid": sum(1 for r_c in c_list if r_c.status == "INVALID"),
            })

            if writer is not None:
                tr0 = time.perf_counter()
                demo_img = _make_demo_frame_side_by_side(frame, out_sh, seq_id, f_idx, t_dt, t_inf)
                writer.write(demo_img)
                stage_render_ms.append((time.perf_counter() - tr0) * 1000.0)

            if seq_id == "ruralRoad_202f":
                cached_rural_for_runtime.append((fid, f_idx, ts, frame, dmap, boxes, confs, tids))

            seq_frames += 1
            seq_dets += n_b

        cap.release()
        if writer is not None:
            writer.release()

        wall_s = time.perf_counter() - seq_t_wall0
        per_video_stats[seq_id] = {
            "resolution": f"{W}x{H}",
            "fps_source": fps,
            "frames_processed": seq_frames,
            "total_detections": seq_dets,
            "prior_exposure": seq["prior_exposure"],
            "conditions": seq["conditions"],
            "measured_shadow_throughput_fps": round(seq_frames / max(1e-6, wall_s), 2),
            "valid_counts": dict(seq_m_valid),
            "status_counts": {m: dict(cnt) for m, cnt in seq_m_status.items()},
        }

    # Aggregate global A/B/C/D video metrics: matched vs admitted
    all_keys = list(all_obs_by_method["A_legacy_annular_ema"].keys())
    common_abc = [k for k in all_keys if all(all_obs_by_method[m][k]["is_valid"] for m in ("A_legacy_annular_ema", "B_opt_ransac_valid_ema", "C_opt_ransac_rollmed_ema"))]

    method_agg = {}
    per_track_csv_rows = []
    for m in all_methods:
        recs = [all_obs_by_method[m][k] for k in all_keys]
        st_cnt = Counter(r["status"] for r in recs)
        fail_cnt = Counter(fr for r in recs for fr in r["failure_reasons"])
        val_recs = [r for r in recs if r["is_valid"]]

        # Matched Common-ABC vs own admitted
        matched_recs = [all_obs_by_method[m][k] for k in common_abc]

        # Per-track dispersion metrics on >=4 valid obs
        t_cv_list, t_mad_list, t_delta_raw, t_delta_smooth, t_rho_area = [], [], [], [], []
        for (sq, tid), tlist in sorted(track_store[m].items()):
            tlist_s = sorted(tlist, key=lambda x: x["frame_idx"])
            v_raw = [x["raw_relative_p75"] for x in tlist_s if x["is_valid"] and x["raw_relative_p75"] is not None]
            v_sm = [x["smoothed_relative_score"] for x in tlist_s if x["smoothed_relative_score"] is not None]
            areas = [obs_meta[(sq, x["frame_idx"], tid)]["box_area"] for x in tlist_s if x["is_valid"] and x["raw_relative_p75"] is not None]
            med_dr = float(np.median(np.abs(np.diff(v_raw)))) if len(v_raw) >= 2 else None
            med_ds = float(np.median(np.abs(np.diff(v_sm)))) if len(v_sm) >= 2 else None
            cv_s = float(np.std(v_sm, ddof=1) / np.mean(v_sm)) if (len(v_sm) >= 4 and np.mean(v_sm) > 0.05) else None
            mad_s = float(np.median(np.abs(np.asarray(v_sm) - np.median(v_sm)))) if len(v_sm) >= 4 else None
            rho_a = _spearman_rho(np.asarray(areas), np.asarray(v_raw)) if len(v_raw) >= 4 else None

            if med_dr is not None: t_delta_raw.append(med_dr)
            if med_ds is not None: t_delta_smooth.append(med_ds)
            if cv_s is not None: t_cv_list.append(cv_s)
            if mad_s is not None: t_mad_list.append(mad_s)
            if rho_a is not None: t_rho_area.append(rho_a)

            per_track_csv_rows.append({
                "seq_id": sq,
                "track_id": tid,
                "method": m,
                "total_obs": len(tlist_s),
                "valid_obs": sum(1 for x in tlist_s if x["is_valid"]),
                "stale_obs": sum(1 for x in tlist_s if x["status"] == "STALE"),
                "expired_obs": sum(1 for x in tlist_s if x["status"] == "EXPIRED"),
                "invalid_obs": sum(1 for x in tlist_s if x["status"] == "INVALID"),
                "median_abs_delta_raw": round(med_dr, 4) if med_dr is not None else None,
                "median_abs_delta_smooth": round(med_ds, 4) if med_ds is not None else None,
                "smooth_mad": round(mad_s, 4) if mad_s is not None else None,
                "smooth_cv": round(cv_s, 4) if cv_s is not None else None,
            })

        method_agg[m] = {
            "total_detections": len(recs),
            "status_counts": dict(st_cnt),
            "valid_pct": round(100.0 * st_cnt["VALID"] / max(1, len(recs)), 2),
            "stale_pct": round(100.0 * st_cnt["STALE"] / max(1, len(recs)), 2),
            "expired_pct": round(100.0 * st_cnt["EXPIRED"] / max(1, len(recs)), 2),
            "invalid_pct": round(100.0 * st_cnt["INVALID"] / max(1, len(recs)), 2),
            "failure_reason_counts": dict(fail_cnt),
            "admitted_raw_p75": _summarize_vals([r["raw_relative_p75"] for r in val_recs if r["raw_relative_p75"] is not None]),
            "admitted_ref_mad": _summarize_vals([r["ref_mad"] for r in val_recs if r["ref_mad"] is not None]),
            "common_abc_count": len(common_abc),
            "common_abc_raw_p75": _summarize_vals([r["raw_relative_p75"] for r in matched_recs if r["raw_relative_p75"] is not None]),
            "common_abc_ref_mad": _summarize_vals([r["ref_mad"] for r in matched_recs if r["ref_mad"] is not None]),
            "track_level": {
                "tracks_total": len(track_store[m]),
                "mean_track_med_delta_raw": round(float(np.mean(t_delta_raw)), 4) if t_delta_raw else None,
                "mean_track_med_delta_smooth": round(float(np.mean(t_delta_smooth)), 4) if t_delta_smooth else None,
                "mean_track_smooth_mad": round(float(np.mean(t_mad_list)), 4) if t_mad_list else None,
                "mean_track_smooth_cv": round(float(np.mean(t_cv_list)), 4) if t_cv_list else None,
                "mean_approach_spearman_rho": round(float(np.mean(t_rho_area)), 4) if t_rho_area else None,
            },
            "extraction_latency_ms": compute_latency_stats([r["extraction_latency_ms"] for r in recs]),
            "filtering_latency_ms": compute_latency_stats([r["filtering_latency_ms"] for r in recs]),
        }

    # Separate standalone vs shadow-mode memory & un-instrumented runtime sub-stage timing on ruralRoad_202f
    opt_cfg_sa = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    sa_times = []
    for fid, f_idx, ts, frm, dm, bxs, cfs, tds in cached_rural_for_runtime:
        if not bxs or dm is None:
            continue
        for i_b, bx in enumerate(bxs):
            nbrs = [bxs[j] for j in range(len(bxs)) if j != i_b]
            res_o = extract_optimized_relative_depth_for_box(dm, bx, neighbor_boxes=nbrs, img_bgr=frm, cfg=opt_cfg_sa)
            tm = res_o["timing_ms"]
            sa_ann_ms.append(tm["annular_region_ms"])
            sa_fit_ms.append(tm["ref_fit_ms"])
            sa_res_ms.append(tm["residual_compute_ms"])
            sa_qa_ms.append(tm["quality_eval_ms"])
            sa_times.append(tm["extraction_total_ms"])

    tracemalloc.start()
    pipe_mem = ShadowDepthPipeline()
    for fid, f_idx, ts, frm, dm, bxs, cfs, tds in cached_rural_for_runtime[:60]:
        _ = pipe_mem.process_frame(fid, f_idx, ts, frm, dm, bxs, cfs, tds)
    _, shadow_peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Write CSV exports cleanly
    pf_csv = os.path.join(BENCH_DIR, "shadow_depth_per_frame.csv")
    with open(pf_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_frame_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(per_frame_csv_rows)

    pt_csv = os.path.join(BENCH_DIR, "shadow_depth_per_track.csv")
    with open(pt_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_track_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(per_track_csv_rows)

    runtime_summary = {
        "total_video_frames": sum(v["frames_processed"] for v in per_video_stats.values()),
        "total_video_regions": total_video_regions,
        "adaptive_1024_triggers": adaptive_escalations,
        "adaptive_1024_trigger_pct": round(100.0 * adaptive_escalations / max(1, total_video_regions), 2),
        "stage_latencies_ms": {
            "video_decode": compute_latency_stats(stage_decode_ms),
            "det_and_bytetrack": compute_latency_stats(stage_det_track_ms),
            "midas_preprocess_active": compute_latency_stats([x for x in stage_pre_ms if x > 0]),
            "midas_inference_active": compute_latency_stats([x for x in stage_midas_ms if x > 0]),
            "annular_sampling_standalone": compute_latency_stats(sa_ann_ms),
            "ref_fitting_standalone": compute_latency_stats(sa_fit_ms),
            "cavity_extraction_standalone": compute_latency_stats(sa_res_ms),
            "quality_gating_standalone": compute_latency_stats(sa_qa_ms),
            "csv_logging_per_frame": compute_latency_stats(stage_log_ms),
            "demo_rendering_per_frame": compute_latency_stats(stage_render_ms),
        },
        "frame_scaling_end_to_end_ms": {
            "shadow_mode_all_branches": {k: compute_latency_stats(v) for k, v in scaling_shadow_ms.items() if v},
            "standalone_optimized_C_projected": {k: compute_latency_stats(v) for k, v in scaling_standalone_opt_ms.items() if v},
            "standalone_legacy_A_projected": {k: compute_latency_stats(v) for k, v in scaling_legacy_ms.items() if v},
        },
        "shadow_pipeline_peak_cpu_kb": round(shadow_peak_b / 1024.0, 2),
    }

    video_ab_summary = {
        "sequences": per_video_stats,
        "common_valid_abc_regions": len(common_abc),
        "methods": method_agg,
    }
    return video_ab_summary, runtime_summary, track_store, demo_video_paths


def run_temporal_lifecycle_suite() -> Dict[str, Any]:
    """Task 6: 11-point controlled temporal lifecycle, fault, resolution, neighbor & camera-profile tests."""
    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    def _d(cav=12.0, box=(260, 150, 360, 230), hh=360, ww=640):
        yyy, xxx = np.mgrid[0:hh, 0:ww].astype(np.float64)
        p = 0.05 * yyy - 0.02 * xxx + 250.0
        x1, y1, x2, y2 = box
        cy, cx = 0.5 * (y1 + y2), 0.5 * (x1 + x2)
        ry, rx = 0.45 * (y2 - y1), 0.45 * (x2 - x1)
        r2 = ((yyy - cy) / max(1.0, ry)) ** 2 + ((xxx - cx) / max(1.0, rx)) ** 2
        return (p - np.where(r2 <= 1.0, cav * np.maximum(0.0, 1.0 - r2) ** 0.75, 0.0)).astype(np.float32)

    img360 = np.full((360, 640, 3), 90, dtype=np.uint8)
    img720 = np.full((720, 1280, 3), 90, dtype=np.uint8)
    box360 = (260.0, 150.0, 360.0, 230.0)
    box720 = (520.0, 300.0, 720.0, 460.0)

    flt_cfg = TemporalFilterConfig(alpha=0.35, rolling_window=5, max_stale_frames=3, max_missed_frames_reset=8)
    report: Dict[str, Any] = {}

    # 1 & 2. Invalid frames -> STALE (state frozen) -> EXPIRED (>=4)
    p1 = ShadowDepthPipeline(filter_cfg=flt_cfg)
    d_val, d_flat = _d(14.0), _d(0.0)
    r0 = p1.process_frame("f0", 0, 0.0, img360, d_val, [box360], [0.9], [1])
    s0 = r0["results_by_method"]["B_opt_ransac_valid_ema"][0].smoothed_relative_score
    st_seq = []
    for k in range(1, 6):
        rk = p1.process_frame(f"f{k}", k, k * 0.1, img360, d_flat, [box360], [0.9], [1])["results_by_method"]["B_opt_ransac_valid_ema"][0]
        st_seq.append((rk.status, rk.smoothed_relative_score, rk.stale_age_frames))
    report["1_2_invalid_and_stale_expiry"] = {
        "initial_valid_score": s0,
        "stale_sequence_status_score_age": st_seq,
        "passed": (st_seq[0][0] == "STALE" and st_seq[0][1] == s0 and st_seq[3][0] == "EXPIRED" and st_seq[3][1] is None),
    }

    # 3 & 4 & 5. Temporary loss (<8 holds), expiry (>=8 resets), and track ID reuse without leakage
    p2 = ShadowDepthPipeline(filter_cfg=flt_cfg)
    p2.process_frame("f0", 0, 0.0, img360, _d(6.0), [box360], [0.9], [9])
    for k in range(1, 3):  # short 2-frame detection loss (<8) -> recovers old EMA smoothly
        p2.process_frame(f"m{k}", k, k * 0.1, img360, d_val, [], [], [])
    rec_short = p2.process_frame("f3", 3, 0.3, img360, _d(18.0), [box360], [0.9], [9])["results_by_method"]["B_opt_ransac_valid_ema"][0]
    for k in range(4, 13):  # long 9-frame detection loss (>=8) -> reset state completely; reused ID=9 starts age=0
        p2.process_frame(f"m{k}", k, k * 0.1, img360, d_val, [], [], [])
    rec_long = p2.process_frame("f13", 13, 1.3, img360, _d(28.0), [box360], [0.9], [9])["results_by_method"]["B_opt_ransac_valid_ema"][0]
    report["3_4_5_loss_expiry_id_reuse"] = {
        "short_loss_smoothed_blended": rec_short.smoothed_relative_score,
        "short_loss_raw": rec_short.raw_relative_p75,
        "long_loss_reset_age": rec_long.track_age_frames,
        "long_loss_zero_leak_diff": round(abs(rec_long.smoothed_relative_score - rec_long.raw_relative_p75), 6),
        "passed": (rec_short.smoothed_relative_score < rec_short.raw_relative_p75 and rec_long.track_age_frames == 0 and abs(rec_long.smoothed_relative_score - rec_long.raw_relative_p75) < 1e-4),
    }

    # 6. Abrupt step response lag (6 -> 22 at step 10)
    p3 = ShadowDepthPipeline(filter_cfg=flt_cfg)
    lag_res = {}
    traces = {m: [] for m in ("A_legacy_annular_ema", "B_opt_ransac_valid_ema", "C_opt_ransac_rollmed_ema")}
    d_pre, d_post = _d(6.0), _d(22.0)
    for k in range(25):
        out_k = p3.process_frame(f"s{k}", k, k * 0.1, img360, d_pre if k < 10 else d_post, [box360], [0.9], [1])
        for m in traces:
            traces[m].append(out_k["results_by_method"][m][0].smoothed_relative_score)
    for m in ("B_opt_ransac_valid_ema", "C_opt_ransac_rollmed_ema"):
        v_pre, v_post = traces[m][9], traces[m][-1]
        targ90 = v_pre + 0.90 * (v_post - v_pre)
        lag = next((idx - 10 for idx in range(10, 25) if traces[m][idx] >= targ90), None)
        lag_res[m] = lag
    report["6_abrupt_step_lag_frames"] = lag_res

    # 7 & 8. Extreme depth outliers (+500 NaN/Inf) & Missing depth_map=None
    p4 = ShadowDepthPipeline(filter_cfg=flt_cfg)
    d_nan = _d(12.0).copy(); d_nan[160:215, 270:350] = np.nan
    d_out = _d(12.0).copy(); d_out[115:142, 225:395] += 500.0
    o_none = p4.process_frame("fn", 0, 0.0, img360, None, [box360], [0.9], [1])
    o_nan = p4.process_frame("fnan", 1, 0.1, img360, d_nan, [box360], [0.9], [2])
    o_out = p4.process_frame("fout", 2, 0.2, img360, d_out, [box360], [0.9], [3])
    report["7_8_missing_nan_and_outliers"] = {
        "none_depth_C_status": o_none["results_by_method"]["C_opt_ransac_rollmed_ema"][0].status,
        "nan_crater_C_status": o_nan["results_by_method"]["C_opt_ransac_rollmed_ema"][0].status,
        "extreme_ring_outlier_C_valid": o_out["results_by_method"]["C_opt_ransac_rollmed_ema"][0].is_valid,
        "extreme_ring_outlier_C_mad": o_out["results_by_method"]["C_opt_ransac_rollmed_ema"][0].ref_mad,
    }

    # 9, 10 & 11. Resolution switch, neighboring detections, camera profile change, & fault isolation
    p5 = ShadowDepthPipeline(filter_cfg=flt_cfg)
    o_nbr = p5.process_frame("fnbr", 0, 0.0, img360, _d(12.0), [box360, (375.0, 150.0, 440.0, 225.0)], [0.9, 0.85], [1, 2])
    p5.set_camera_profile("commercial_truck_bus")
    o_720 = p5.process_frame("f720", 1, 0.1, img720, _d(12.0, box=(520, 300, 720, 460), hh=720, ww=1280), [box720], [0.9], [1])
    p5.fault_hooks["B_filter"] = lambda: (_ for _ in ()).throw(RuntimeError("injected fault"))
    o_flt = p5.process_frame("fflt", 2, 0.2, img360, _d(12.0), [box360], [0.9], [1])
    report["9_10_11_resolution_neighbor_profile_fault"] = {
        "neighbor_C_valid": o_nbr["results_by_method"]["C_opt_ransac_rollmed_ema"][0].is_valid,
        "res_720p_truck_C_status": o_720["results_by_method"]["C_opt_ransac_rollmed_ema"][0].status,
        "fault_A_status": o_flt["results_by_method"]["A_legacy_annular_ema"][0].status,
        "fault_B_status": o_flt["results_by_method"]["B_opt_ransac_valid_ema"][0].status,
        "fault_C_status": o_flt["results_by_method"]["C_opt_ransac_rollmed_ema"][0].status,
    }
    return report


def run_adaptive_synthetic_ablation() -> Dict[str, Any]:
    """Task 7: Evaluate fixed 512 vs adaptive 512->1024 trigger on synthetic noisy scenarios."""
    cases = _build_randomized_regression_scenarios([101, 202, 303])
    cfg_512 = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    cfg_1024 = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=1024)

    fixed_errs, adapt_errs = [], []
    noise_fixed, noise_adapt = [], []
    ms_512, ms_adapt = [], []
    triggers = 0
    for name, sd, grp, true_p, d_map, box, nbrs, exp_v, sc_div in cases:
        if not exp_v:
            continue
        x1, y1, x2, y2 = box
        t_res = (true_p[y1:y2, x1:x2] - d_map[y1:y2, x1:x2])
        t_p75 = float(np.percentile(t_res[np.isfinite(t_res)], 75.0))
        o512 = extract_optimized_relative_depth_for_box(d_map, box, neighbor_boxes=nbrs, cfg=cfg_512)
        e512 = abs(o512["relative_cavity_stats"]["raw_diagnostic_p75"] - t_p75) / sc_div
        t_tot = o512["timing_ms"]["extraction_total_ms"]
        ea = e512
        if should_escalate_budget(o512):
            triggers += 1
            o1024 = extract_optimized_relative_depth_for_box(d_map, box, neighbor_boxes=nbrs, cfg=cfg_1024)
            ea = abs(o1024["relative_cavity_stats"]["raw_diagnostic_p75"] - t_p75) / sc_div
            t_tot += o1024["timing_ms"]["extraction_total_ms"]
        fixed_errs.append(e512)
        adapt_errs.append(ea)
        ms_512.append(o512["timing_ms"]["extraction_total_ms"])
        ms_adapt.append(t_tot)
        if grp == "gaussian_noise":
            noise_fixed.append(e512)
            noise_adapt.append(ea)

    return {
        "valid_synthetic_cases": len(fixed_errs),
        "adaptive_triggers": triggers,
        "fixed_512_mean_p75_err": round(float(np.mean(fixed_errs)), 5),
        "adaptive_mean_p75_err": round(float(np.mean(adapt_errs)), 5),
        "high_noise_fixed_512_mean_p75_err": round(float(np.mean(noise_fixed)), 5),
        "high_noise_adaptive_mean_p75_err": round(float(np.mean(noise_adapt)), 5),
        "mean_ms_512": round(float(np.mean(ms_512)), 4),
        "mean_ms_adaptive": round(float(np.mean(ms_adapt)), 4),
    }


def render_phase5c_plots(
    pre_res: Dict[str, Any],
    video_res: Dict[str, Any],
    runtime_res: Dict[str, Any],
) -> List[str]:
    paths = []

    # 1. Preprocessing 2x2 Ablation Plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), dpi=130)
    f22 = pre_res["factorial_2x2"]
    labels = ["BGR +\nLegacy", "RGB +\nLegacy", "BGR +\nOpt RANSAC", "RGB +\nOpt RANSAC"]
    keys = ["bgr_legacy_plus_legacy_annular", "rgb_corrected_plus_legacy_annular", "bgr_legacy_plus_optimized_ransac", "rgb_corrected_plus_optimized_ransac"]
    vals_pct = [f22[k]["valid_pct"] for k in keys]
    mad_meds = [f22[k]["common_ref_mad"]["median"] or 0.0 for k in keys]
    axes[0].bar(labels, vals_pct, color=["#94a3b8", "#64748b", "#3b82f6", "#16a34a"])
    axes[0].set_ylabel("Valid Measurement Coverage (%)")
    axes[0].set_title("2x2 Preprocessing x Extractor Coverage (n=98)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, mad_meds, color=["#94a3b8", "#64748b", "#3b82f6", "#16a34a"])
    axes[1].set_ylabel("Common-Valid Median Ref MAD (Rel Units)")
    axes[1].set_title("Reference Fit MAD on Common Valid Subset")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Phase 5C Task 3 — BGR Legacy vs RGB Corrected Preprocessing Ablation", fontsize=11.5, fontweight="bold")
    p1 = os.path.join(VIZ_DIR, "shadow_preprocessing_ablation.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # 2. Latency Scaling (0, 1, 2, 3, 4+ Candidates) & Pipeline Stage Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), dpi=130)
    cats = ["0", "1", "2", "3", "4+"]
    sc = runtime_res["frame_scaling_end_to_end_ms"]
    sh_y = [sc["shadow_mode_all_branches"].get(c, {}).get("mean_ms", np.nan) for c in cats]
    so_y = [sc["standalone_optimized_C_projected"].get(c, {}).get("mean_ms", np.nan) for c in cats]
    sl_y = [sc["standalone_legacy_A_projected"].get(c, {}).get("mean_ms", np.nan) for c in cats]
    axes[0].plot(cats, sh_y, "o-", color="#dc2626", linewidth=2, label="Shadow Mode (A+B+C+D)")
    axes[0].plot(cats, so_y, "s-", color="#16a34a", linewidth=2, label="Standalone Opt RANSAC+RollMed (C)")
    axes[0].plot(cats, sl_y, "^--", color="#2563eb", linewidth=2, label="Standalone Legacy (A)")
    axes[0].set_xlabel("Detected Pothole Candidates per Frame")
    axes[0].set_ylabel("Mean Per-Frame Latency (ms)")
    axes[0].set_title("Per-Frame Scaling Across 0..4+ Candidates (567 Video Frames)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    st = runtime_res["stage_latencies_ms"]
    st_names = ["Decode", "YOLO+\nTrack", "MiDaS\nPreproc", "MiDaS\nInfer", "Opt\nAnnular", "Opt\nRANSAC", "Res+QA"]
    st_vals = [
        st["video_decode"]["mean_ms"],
        st["det_and_bytetrack"]["mean_ms"],
        st["midas_preprocess_active"]["mean_ms"],
        st["midas_inference_active"]["mean_ms"],
        st["annular_sampling_standalone"]["mean_ms"],
        st["ref_fitting_standalone"]["mean_ms"],
        st["cavity_extraction_standalone"]["mean_ms"] + st["quality_gating_standalone"]["mean_ms"],
    ]
    axes[1].bar(st_names, st_vals, color="#0ea5e9")
    axes[1].set_ylabel("Mean Latency (ms)")
    axes[1].set_title("Component Stage Means (GPU + Standalone Per-Region CPU)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Phase 5C Task 8 — End-to-End Runtime Scaling & Shadow Overhead", fontsize=11.5, fontweight="bold")
    p2 = os.path.join(VIZ_DIR, "shadow_latency_scaling_and_stages.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)

    # 3. Video Coverage & Status Distribution (A, B, C, D)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.5), dpi=130)
    ms = ["A_legacy_annular_ema", "B_opt_ransac_valid_ema", "C_opt_ransac_rollmed_ema", "D_opt_ransac_adaptive_1024"]
    m_short = ["A: Legacy\nAnnular+EMA", "B: Opt RANSAC\n+ValidEMA", "C: Opt RANSAC\n+RollMedEMA", "D: Adaptive\n512->1024"]
    mv = video_res["methods"]
    v_pcts = np.array([mv[m]["valid_pct"] for m in ms])
    s_pcts = np.array([mv[m]["stale_pct"] for m in ms])
    e_pcts = np.array([mv[m]["expired_pct"] for m in ms])
    i_pcts = np.array([mv[m]["invalid_pct"] for m in ms])
    x = np.arange(len(ms))
    axes[0].bar(x, v_pcts, label="VALID", color="#16a34a")
    axes[0].bar(x, s_pcts, bottom=v_pcts, label="STALE", color="#f59e0b")
    axes[0].bar(x, e_pcts, bottom=v_pcts + s_pcts, label="EXPIRED", color="#7c3aed")
    axes[0].bar(x, i_pcts, bottom=v_pcts + s_pcts + e_pcts, label="INVALID", color="#dc2626")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(m_short, fontsize=8.5)
    axes[0].set_ylabel("Share of 567-Frame Detections (%)")
    axes[0].set_title("Status Breakdown Across All Video Sequences")
    axes[0].legend(fontsize=8)

    d_raw = [mv[m]["track_level"]["mean_track_med_delta_raw"] or 0.0 for m in ms]
    d_smt = [mv[m]["track_level"]["mean_track_med_delta_smooth"] or 0.0 for m in ms]
    axes[1].bar(x - 0.18, d_raw, width=0.35, label="Raw Relative Delta", color="#94a3b8")
    axes[1].bar(x + 0.18, d_smt, width=0.35, label="Smoothed Score Delta", color="#2563eb")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(m_short, fontsize=8.5)
    axes[1].set_ylabel("Mean Track Median |Δ Score| (Rel Units)")
    axes[1].set_title("Within-Track Frame-to-Frame Dispersion (Raw vs Smoothed)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Phase 5C Task 9 — Multi-Video A/B/C/D Coverage, Lifecycle & Temporal Dispersion", fontsize=11.5, fontweight="bold")
    p3 = os.path.join(VIZ_DIR, "shadow_ab_status_and_dispersion.png")
    plt.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)

    return paths


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = YOLO(DEFAULT_DET_WEIGHTS)
    midas = SharedMiDaSRunner(device=device_str)
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    _ = det_model.predict(dummy, imgsz=640, conf=0.25, iou=0.70, device=0 if device_str == "cuda" else "cpu", verbose=False)
    midas.warmup(360, 640)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)

    pre_ablation = run_preprocessing_ablation(det_model, midas)
    video_ab, runtime_res, _, demo_videos = run_video_shadow_and_runtime(det_model, midas)
    lifecycle_res = run_temporal_lifecycle_suite()
    adaptive_synth = run_adaptive_synthetic_ablation()
    plot_paths = render_phase5c_plots(pre_ablation, video_ab, runtime_res)

    peak_gpu_alloc = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0
    peak_gpu_res = round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0

    results = {
        "experiment": "Phase 5C Shadow-Mode Integration & Production Readiness Evaluation",
        "peak_gpu_allocated_mb": peak_gpu_alloc,
        "peak_gpu_reserved_mb": peak_gpu_res,
        "preprocessing_ablation_task3": pre_ablation,
        "video_shadow_evaluation_task5_9": video_ab,
        "temporal_lifecycle_failure_suite_task6": lifecycle_res,
        "adaptive_512_1024_synthetic_ablation_task7": adaptive_synth,
        "end_to_end_runtime_task8": runtime_res,
        "visualization_paths": plot_paths,
        "demo_video_paths": demo_videos,
    }

    out_json = os.path.join(BENCH_DIR, "shadow_depth_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== PHASE 5C BENCHMARK COMPLETE ===")
    print("Frames processed:", runtime_res["total_video_frames"], "| Regions:", runtime_res["total_video_regions"])
    for m, d in video_ab["methods"].items():
        print(f"  {m:28s} | VALID={d['valid_pct']:5.2f}% STALE={d['stale_pct']:5.2f}% EXP={d['expired_pct']:5.2f}% | ext_ms={d['extraction_latency_ms']['mean_ms']:.2f}")
    print("Adaptive 512->1024 video triggers:", runtime_res["adaptive_1024_triggers"], f"({runtime_res['adaptive_1024_trigger_pct']}%)")
    print("Adaptive synthetic high-noise err:", adaptive_synth["high_noise_fixed_512_mean_p75_err"], "->", adaptive_synth["high_noise_adaptive_mean_p75_err"])


if __name__ == "__main__":
    main()
