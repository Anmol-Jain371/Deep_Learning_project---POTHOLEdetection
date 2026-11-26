"""
========================================================================================
RoadVision AI -- Phase 6B: Depth Anything V2 Resolution Scaling & Optimization Benchmark
========================================================================================
Controlled evaluation of `DepthAnythingV2Small` across:
  - input_size in {266, 336, 392, 518 (model reference, NOT ground truth)}
  - Baseline: MiDaS_small (256, rgb_corrected)
  - Cavity strategies: `elliptical_core` (primary) and `full_box` (control)
  - Reference surface: Optimized RANSAC (`stratified_quadrant`, max_ref=512, 16..128 iters)

Stages strictly separated:
  1. Stage A (Dev Debug): 36 dev frames (`98` regions) + `±8%` perturbation per size bucket.
  2. Freeze exact policy, thresholds, strata, and RANSAC/core configs before Stage B/C.
  3. Stage B (Exploratory Corpus): 360f Phase 6A footage (`575` regions).
  4. Stage C (Untouched Holdout): `rangoonRoad_f785_1004` (never evaluated in prior depth
     phases) + Phase-6A-unseen `ruralRoad_f120_201` + `testClip_f0_24` (frozen settings).
  5. Matched temporal stability (`raw` vs `validity_aware_ema`), RTX 2050 GPU profiling
     (runtime vs isolated VRAM), FP32 vs FP16 stability, Task 8 acceleration PoC, and Task 10 plots.

Exploratory research ONLY — zero production edits, zero metric cm claims.
========================================================================================
"""

import os
import sys
import csv
import json
import time
import hashlib
import platform
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.roi_guided_segmentation import DEFAULT_DET_WEIGHTS
from experiments.benchmark_roi_segmentation import build_dev_frame_catalog, load_frame, compute_latency_stats
from experiments.optimized_relative_depth import OptimizedDepthConfig, build_and_sample_annular_region, fit_ransac_optimized
from experiments.core_aware_cavity_extraction import CoreExtractionConfig, extract_cavity_with_strategy
from experiments.depth_model_comparison import MiDaSSmallAdapter, DepthAnythingV2SmallAdapter
from experiments.validity_aware_temporal_filter import TemporalFilterConfig, ValidityAwareTrackFilter
from experiments.benchmark_optimized_relative_depth import _spearman_rho, _guarded_rel_diff_pct
from experiments.dav2_inference_optimization import (
    spearman_fast,
    affine_align,
    benchmark_fp32_vs_fp16,
    run_acceleration_feasibility,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "phase6b_viz")

DAV2_RESOLUTIONS = [266, 336, 392, 518]

# Frozen provisional engineering acceptance policy (Task 9) BEFORE corpus evaluation
FROZEN_ACCEPTANCE_POLICY = {
    "policy_version": "phase6b_v1_frozen_before_corpus",
    "min_coverage_retention_vs_518": 0.90,
    "min_matched_spearman_vs_518": 0.85,
    "min_validity_agreement_vs_518": 0.88,
    "min_urban_retention_vs_518": 0.85,
    "min_small_pothole_retention_vs_518": 0.85,
    "max_temporal_cv_inflation_vs_518": 1.15,
    "target_realtime_e2e_ms": 45.0,  # ~22.2 FPS reference engineering target
    "ransac_config": {
        "sampling_mode": "stratified_quadrant",
        "max_ref_samples": 512,
        "min_iters": 16,
        "max_iters": 128,
    },
    "temporal_filter_config": {
        "mode": "validity_aware_ema",
        "alpha": 0.35,
        "rolling_window": 5,
        "max_stale_frames": 3,
        "max_missed_frames_reset": 8,
    },
}

EXPLORATORY_SOURCES = [
    {
        "source_id": "ruralRoad_f0_119",
        "road_env": "rural",
        "path": "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
        "start_frame": 0,
        "count": 120,
        "split": "exploratory_360f",
    },
    {
        "source_id": "rangoonRoad_f200_319",
        "road_env": "rangoon",
        "path": "Watch Video_ Terrible potholes in Rangoon Road.mp4",
        "start_frame": 200,
        "count": 120,
        "split": "exploratory_360f",
    },
    {
        "source_id": "rawSurvey_720p_f0_119",
        "road_env": "urban_rawSurvey",
        "path": "roadvision-dashboard/public/videos/raw_survey.mp4",
        "start_frame": 0,
        "count": 120,
        "split": "exploratory_360f",
    },
]

HOLDOUT_SOURCES = [
    {
        "source_id": "rangoonRoad_f785_1004_untouched",
        "road_env": "rangoon_holdout",
        "path": "Watch Video_ Terrible potholes in Rangoon Road.mp4",
        "start_frame": 785,
        "count": 220,
        "split": "holdout_untouched",
    },
    {
        "source_id": "ruralRoad_f120_201_unseen6a",
        "road_env": "rural_holdout",
        "path": "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
        "start_frame": 120,
        "count": 82,
        "split": "holdout_untouched",
    },
    {
        "source_id": "testClip_f0_24_unseen6a",
        "road_env": "test_clip",
        "path": "uploads/upload_1794546659_test_clip.mp4",
        "start_frame": 0,
        "count": 25,
        "split": "holdout_untouched",
    },
]


def classify_candidate_strata(
    box_xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    net_hw: Tuple[int, int],
    conf: float,
    road_env: str,
) -> Dict[str, Any]:
    """Assign frozen Task 4 sensitivity strata for a candidate detection."""
    bw = max(1.0, float(box_xyxy[2] - box_xyxy[0]))
    bh = max(1.0, float(box_xyxy[3] - box_xyxy[1]))
    area_px = bw * bh
    area_frac = area_px / float(max(1, img_w * img_h))
    y2_ratio = float(box_xyxy[3]) / float(max(1, img_h))

    net_h, net_w = int(net_hw[0]), int(net_hw[1])
    eff_w = bw * (float(net_w) / float(max(1, img_w)))
    eff_h = bh * (float(net_h) / float(max(1, img_h)))
    eff_min_dim = min(eff_w, eff_h)

    if area_frac < 0.005:
        area_bucket = "tiny_lt_0.5pct"
    elif area_frac < 0.015:
        area_bucket = "small_0.5_to_1.5pct"
    elif area_frac < 0.05:
        area_bucket = "medium_1.5_to_5pct"
    else:
        area_bucket = "large_ge_5pct"

    if eff_min_dim < 14.0:
        eff_dim_bucket = "sub1_vit_patch_lt14px"
    elif eff_min_dim < 28.0:
        eff_dim_bucket = "patch1to2_14_to_28px"
    elif eff_min_dim < 56.0:
        eff_dim_bucket = "patch2to4_28_to_56px"
    else:
        eff_dim_bucket = "patch_ge4_ge56px"

    if y2_ratio < 0.50:
        y_bucket = "distant_top_y2_lt_0.50"
    elif y2_ratio < 0.75:
        y_bucket = "mid_road_0.50_to_0.75"
    else:
        y_bucket = "near_bot_ge_0.75"

    if conf < 0.40:
        conf_bucket = "low_conf_lt_0.40"
    elif conf < 0.65:
        conf_bucket = "mid_conf_0.40_to_0.65"
    else:
        conf_bucket = "high_conf_ge_0.65"

    return {
        "area_px": round(area_px, 1),
        "area_frac": round(area_frac, 6),
        "area_bucket": area_bucket,
        "is_small_pothole": bool(area_frac < 0.015),
        "y2_ratio": round(y2_ratio, 4),
        "y_bucket": y_bucket,
        "eff_w": round(eff_w, 2),
        "eff_h": round(eff_h, 2),
        "eff_min_dim": round(eff_min_dim, 2),
        "eff_dim_bucket": eff_dim_bucket,
        "conf_bucket": conf_bucket,
        "frame_res": f"{img_w}x{img_h}",
        "road_env": road_env,
    }


def _collect_env_and_checkpoint_audit(dav2_518: DepthAnythingV2SmallAdapter, midas: MiDaSSmallAdapter) -> Dict[str, Any]:
    """Task 1: Verify Phase 6A DAv2 checkpoint hash, architecture, and environment metadata."""
    with open(DEFAULT_DET_WEIGHTS, "rb") as fd:
        det_sha = hashlib.sha256(fd.read()).hexdigest().upper()
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "cuda_torch": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "yolo_weights": DEFAULT_DET_WEIGHTS,
        "yolo_sha256": det_sha,
        "dav2_ckpt_path": dav2_518.ckpt_path,
        "dav2_ckpt_sha256": dav2_518.ckpt_sha256,
        "dav2_ckpt_bytes": dav2_518.ckpt_bytes,
        "dav2_params": dav2_518.param_count,
        "midas_params": midas.param_count,
        "frozen_policy": FROZEN_ACCEPTANCE_POLICY,
    }


def _extract_cached_arms_for_frame(
    frame_meta: Dict[str, Any],
    preds_by_model: Dict[str, Any],
    opt_cfg: OptimizedDepthConfig,
    core_cfg: CoreExtractionConfig,
    want_arrays_models: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Given cached depth maps for a frame across {midas_256, dav2_266, 336, 392, 518},
    run identical RANSAC reference fitting and compare elliptical_core vs full_box cleanly.
    """
    img_bgr = frame_meta["img_bgr"]
    H, W = img_bgr.shape[:2]
    boxes = frame_meta["boxes"]
    confs = frame_meta["confs"]
    tids = frame_meta["tids"]
    sid = frame_meta["source_id"]
    fidx = frame_meta["frame_idx"]
    ts = frame_meta["ts"]
    road_env = frame_meta["road_env"]
    split = frame_meta["split"]

    # Frame-level representation & IQR stats vs DAv2_518 (model reference, NOT GT!)
    d_518 = preds_by_model["dav2_518"].depth_map
    iqr_map: Dict[str, float] = {}
    frame_rep_vs_518: Dict[str, Any] = {}
    for mkey, pr in preds_by_model.items():
        dm = pr.depth_map
        q25, q75 = np.percentile(dm, [25.0, 75.0])
        iqr_map[mkey] = max(1e-5, float(q75 - q25))
        if mkey.startswith("dav2_") and mkey != "dav2_518":
            sp = spearman_fast(dm, d_518, stride=6)
            _, a_s, b_s, mae_a, nrmse_a = affine_align(dm, d_518)
            frame_rep_vs_518[mkey] = {
                "within_image_spearman": round(sp, 5),
                "affine_scale_a": a_s,
                "affine_shift_b": b_s,
                "affine_aligned_mae": mae_a,
                "affine_aligned_nrmse_iqr": nrmse_a,
                "network_hw": list(pr.network_input_hw),
            }
        elif mkey == "dav2_518":
            frame_rep_vs_518[mkey] = {"within_image_spearman": 1.0, "network_hw": list(pr.network_input_hw)}

    rows_out: List[Dict[str, Any]] = []
    per_box_arms: Dict[int, Dict[str, Any]] = {}

    for idx_b, (bx, cf, tid) in enumerate(zip(boxes, confs, tids)):
        nbrs = [boxes[j] for j in range(len(boxes)) if j != idx_b]
        per_box_arms[idx_b] = {}

        # Precompute RANSAC reference once per depth map so full_box vs elliptical_core use exact same plane
        fits_by_m = {}
        for mkey, pr in preds_by_model.items():
            reg = build_and_sample_annular_region(pr.depth_map, bx, neighbor_boxes=nbrs, cfg=opt_cfg)
            fit = fit_ransac_optimized(pr.depth_map, reg, cfg=opt_cfg)
            fits_by_m[mkey] = (reg, fit)

        for mkey, pr in preds_by_model.items():
            strategies = ("full_box",) if mkey == "midas_256" else ("elliptical_core", "full_box")
            reg, fit = fits_by_m[mkey]
            strata = classify_candidate_strata(bx, W, H, pr.network_input_hw, cf, road_env)
            for cstrat in strategies:
                arm_name = f"{mkey}_{'elliptical' if cstrat == 'elliptical_core' else 'full'}"
                need_arr = bool(want_arrays_models and arm_name in want_arrays_models)
                o = extract_cavity_with_strategy(
                    pr.depth_map, bx, strategy=cstrat, neighbor_boxes=nbrs, img_bgr=img_bgr,
                    opt_cfg=opt_cfg, core_cfg=core_cfg, precomputed_region=reg, precomputed_fit=fit,
                    return_arrays=need_arr,
                )
                per_box_arms[idx_b][arm_name] = o
                st = o["core_cavity_stats"]
                p75_raw = st["p75_residual"]
                sn_p75 = st["scale_normalized_p75"]
                fiqr_p75 = (p75_raw / iqr_map[mkey]) if (p75_raw is not None) else None

                rows_out.append({
                    "split": split,
                    "source_id": sid,
                    "road_env": road_env,
                    "frame_idx": fidx,
                    "ts": ts,
                    "det_idx": idx_b,
                    "track_id": tid,
                    "conf": round(cf, 4),
                    "frame_res": strata["frame_res"],
                    "area_px": strata["area_px"],
                    "area_frac": strata["area_frac"],
                    "area_bucket": strata["area_bucket"],
                    "is_small_pothole": strata["is_small_pothole"],
                    "y2_ratio": strata["y2_ratio"],
                    "y_bucket": strata["y_bucket"],
                    "eff_w": strata["eff_w"],
                    "eff_h": strata["eff_h"],
                    "eff_min_dim": strata["eff_min_dim"],
                    "eff_dim_bucket": strata["eff_dim_bucket"],
                    "conf_bucket": strata["conf_bucket"],
                    "arm": arm_name,
                    "model_key": mkey,
                    "strategy": cstrat,
                    "net_h": pr.network_input_hw[0],
                    "net_w": pr.network_input_hw[1],
                    "is_valid": o["is_valid"],
                    "failure_reasons": "|".join(o["failure_reasons"]),
                    "is_water": o["is_water_suspected"],
                    "inlier_ratio": o["ref_diagnostics"]["inlier_ratio"],
                    "ref_mad": o["ref_diagnostics"]["ref_fit_mad"],
                    "populated_quadrants": o["ref_diagnostics"]["populated_quadrants"],
                    "pos_frac": st["positive_frac"],
                    "raw_diag_p75": st["raw_diag_p75"],
                    "raw_p75": p75_raw,
                    "scale_norm_p75": sn_p75,
                    "frame_iqr_norm_p75": round(fiqr_p75, 6) if fiqr_p75 is not None else None,
                    "extract_ms": o["total_ms"],
                    "box_xyxy": [round(float(v), 1) for v in bx],
                })

    return rows_out, {"frame_rep_vs_518": frame_rep_vs_518, "per_box_arms": per_box_arms}


def run_dev36_debug_and_perturbation(
    det_model: YOLO,
    adapters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Task 4 & 5 Stage A: 36 development frames (`98` detections) checking actual tensor dimensions,
    all arms, and `±8%` bounding-box perturbation sensitivity stratified by pothole size (`small <1.5%`
    vs `medium/large >=1.5%`).
    """
    catalog = build_dev_frame_catalog()
    opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    core_cfg = CoreExtractionConfig()

    arm_keys = ["midas_256_full"] + [f"dav2_{r}_{s}" for r in DAV2_RESOLUTIONS for s in ("elliptical", "full")]
    arm_stats = {a: {"total": 0, "valid": 0, "small_tot": 0, "small_v": 0, "pert_base": 0, "pert_ret": 0,
                     "pert_rel": [], "small_pert_base": 0, "small_pert_ret": 0, "small_pert_rel": [],
                     "reasons": Counter()} for a in arm_keys}
    net_shapes: Dict[str, set] = {k: set() for k in adapters}
    rng_p = np.random.default_rng(42)

    for entry in catalog:
        img_bgr, ts = load_frame(entry)
        H, W = img_bgr.shape[:2]
        r = det_model.predict(img_bgr, imgsz=640, conf=0.25, iou=0.70, verbose=False)[0]
        boxes = [tuple(float(v) for v in b) for b in r.boxes.xyxy.cpu().numpy().tolist()] if (r.boxes is not None and len(r.boxes) > 0) else []
        if not boxes:
            continue
        confs = [float(c) for c in r.boxes.conf.cpu().numpy().tolist()]

        preds = {mk: ad.predict(img_bgr) for mk, ad in adapters.items()}
        for mk, pr in preds.items():
            net_shapes[mk].add((f"{W}x{H}", f"{pr.network_input_hw[0]}x{pr.network_input_hw[1]}"))

        for i, (bx, cf) in enumerate(zip(boxes, confs)):
            nbrs = [boxes[j] for j in range(len(boxes)) if j != i]
            bw, bh = max(1.0, bx[2] - bx[0]), max(1.0, bx[3] - bx[1])
            is_small = (bw * bh) / float(max(1, W * H)) < 0.015
            dx1, dx2 = rng_p.uniform(-0.08 * bw, 0.08 * bw, size=2)
            dy1, dy2 = rng_p.uniform(-0.08 * bh, 0.08 * bh, size=2)
            pbx = (max(0.0, bx[0] + dx1), max(0.0, bx[1] + dy1), min(float(W), bx[2] + dx2), min(float(H), bx[3] + dy2))

            for mk, pr in preds.items():
                strats = ("full_box",) if mk == "midas_256" else ("elliptical_core", "full_box")
                reg = build_and_sample_annular_region(pr.depth_map, bx, neighbor_boxes=nbrs, cfg=opt_cfg)
                fit = fit_ransac_optimized(pr.depth_map, reg, cfg=opt_cfg)
                for st in strats:
                    aname = f"{mk}_{'elliptical' if st == 'elliptical_core' else 'full'}"
                    o0 = extract_cavity_with_strategy(pr.depth_map, bx, strategy=st, neighbor_boxes=nbrs, img_bgr=img_bgr,
                                                      opt_cfg=opt_cfg, core_cfg=core_cfg, precomputed_region=reg, precomputed_fit=fit)
                    op = extract_cavity_with_strategy(pr.depth_map, pbx, strategy=st, neighbor_boxes=nbrs, img_bgr=img_bgr,
                                                      opt_cfg=opt_cfg, core_cfg=core_cfg)
                    d = arm_stats[aname]
                    d["total"] += 1
                    if is_small:
                        d["small_tot"] += 1
                    if o0["is_valid"]:
                        d["valid"] += 1
                        d["pert_base"] += 1
                        if is_small:
                            d["small_v"] += 1
                            d["small_pert_base"] += 1
                        if op["is_valid"]:
                            d["pert_ret"] += 1
                            rel_s = _guarded_rel_diff_pct(op["core_cavity_stats"]["p75_residual"], o0["core_cavity_stats"]["p75_residual"])
                            d["pert_rel"].append(rel_s)
                            if is_small:
                                d["small_pert_ret"] += 1
                                d["small_pert_rel"].append(rel_s)
                    else:
                        for rs in o0["failure_reasons"]:
                            d["reasons"][rs] += 1

    return {
        "observed_network_tensor_shapes": {k: sorted(list(v)) for k, v in net_shapes.items()},
        "arms": {
            a: {
                "total": d["total"],
                "valid": d["valid"],
                "valid_pct": round(100.0 * d["valid"] / max(1, d["total"]), 2),
                "small_valid": f"{d['small_v']}/{d['small_tot']}",
                "pert_8pct_retained": f"{d['pert_ret']}/{d['pert_base']}",
                "pert_8pct_med_rel_shift_pct": round(float(np.median(d["pert_rel"])), 2) if d["pert_rel"] else None,
                "small_pert_retained": f"{d['small_pert_ret']}/{d['small_pert_base']}",
                "small_pert_med_rel_shift_pct": round(float(np.median(d["small_pert_rel"])), 2) if d["small_pert_rel"] else None,
                "reasons": dict(d["reasons"]),
            }
            for a, d in arm_stats.items()
        },
    }


def _evaluate_video_corpus(
    sources: List[Dict[str, Any]],
    det_model: YOLO,
    adapters: Dict[str, Any],
    capture_viz_tags: Optional[Dict[Tuple[str, int, int], str]] = None,
    measure_timing: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a list of video sources with identical detector + ByteTrack boxes/tracks across
    all depth resolutions (`MiDaS 256` and `DAv2 {266, 336, 392, 518}`).
    """
    opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    core_cfg = CoreExtractionConfig()

    all_rows: List[Dict[str, Any]] = []
    frame_rep_records: List[Dict[str, Any]] = []
    viz_items: List[Dict[str, Any]] = []
    tensor_shapes: Dict[str, set] = {k: set() for k in adapters}

    # Isolated runtime accumulators (when measure_timing=True: zero memory calls inside loop)
    t_det_all: List[float] = []
    model_timings = {
        k: {"pre": [], "inf": [], "res": [], "tot": [], "ext_ell": [], "e2e_buckets": {"0": [], "1": [], "2": [], "3": [], "4+": []}}
        for k in adapters
    }

    viz_arm_set = {"midas_256_full", "dav2_266_elliptical", "dav2_336_elliptical", "dav2_392_elliptical", "dav2_518_elliptical"}

    for src in sources:
        sid = src["source_id"]
        road_env = src["road_env"]
        split = src["split"]
        cap = cv2.VideoCapture(os.path.join(BASE_DIR, src["path"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src["start_frame"])
        det_model.predictor = None

        for off in range(src["count"]):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            f_idx = src["start_frame"] + off
            ts = round(f_idx / fps, 4)

            if measure_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            td0 = time.perf_counter()
            res = det_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.50, imgsz=640, verbose=False)[0]
            if measure_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_dt = (time.perf_counter() - td0) * 1000.0
            if measure_timing:
                t_det_all.append(t_dt)

            boxes, confs, tids = [], [], []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy().tolist()]
                confs = [float(c) for c in res.boxes.conf.cpu().numpy().tolist()]
                tids = [int(t) for t in res.boxes.id.int().cpu().tolist()] if res.boxes.id is not None else [i + 1 for i in range(len(boxes))]

            nb_key = "4+" if len(boxes) >= 4 else str(len(boxes))
            if not boxes:
                if measure_timing:
                    for k in adapters:
                        model_timings[k]["e2e_buckets"]["0"].append(t_dt)
                continue

            preds = {mk: ad.predict(frame) for mk, ad in adapters.items()}
            for mk, pr in preds.items():
                tensor_shapes[mk].add((f"{W}x{H}", f"{pr.network_input_hw[0]}x{pr.network_input_hw[1]}"))
                if measure_timing:
                    model_timings[mk]["pre"].append(pr.timings_ms["preprocess_ms"])
                    model_timings[mk]["inf"].append(pr.timings_ms["inference_ms"])
                    model_timings[mk]["res"].append(pr.timings_ms["resize_ms"])
                    model_timings[mk]["tot"].append(pr.timings_ms["total_ms"])

            want_viz_map = {}
            if capture_viz_tags:
                for ib in range(len(boxes)):
                    if (sid, f_idx, ib) in capture_viz_tags:
                        want_viz_map[ib] = capture_viz_tags[(sid, f_idx, ib)]

            fmeta = {
                "img_bgr": frame, "boxes": boxes, "confs": confs, "tids": tids,
                "source_id": sid, "road_env": road_env, "split": split, "frame_idx": f_idx, "ts": ts,
            }
            rows, diag = _extract_cached_arms_for_frame(
                fmeta, preds, opt_cfg, core_cfg, want_arrays_models=viz_arm_set if want_viz_map else None
            )
            all_rows.extend(rows)
            frame_rep_records.append({"source_id": sid, "frame_idx": f_idx, "frame_res": f"{W}x{H}", **diag["frame_rep_vs_518"]})

            if measure_timing:
                # Sum per-frame extraction time per elliptical/baseline arm cleanly from rows
                ext_per_model = Counter()
                for rw in rows:
                    if (rw["model_key"] == "midas_256" and rw["strategy"] == "full_box") or (rw["model_key"].startswith("dav2_") and rw["strategy"] == "elliptical_core"):
                        ext_per_model[rw["model_key"]] += rw["extract_ms"]
                for mk in adapters:
                    em = float(ext_per_model[mk])
                    model_timings[mk]["ext_ell"].append(em)
                    model_timings[mk]["e2e_buckets"][nb_key].append(t_dt + preds[mk].timings_ms["total_ms"] + em)

            for ib, vtag in want_viz_map.items():
                viz_items.append({
                    "tag": vtag,
                    "source_id": sid,
                    "frame_idx": f_idx,
                    "img_bgr": frame.copy(),
                    "box": boxes[ib],
                    "conf": confs[ib],
                    "depth_maps": {mk: preds[mk].depth_map.copy() for mk in adapters},
                    "net_hws": {mk: list(preds[mk].network_input_hw) for mk in adapters},
                    "arms": {ak: av for ak, av in diag["per_box_arms"][ib].items() if ak in viz_arm_set},
                })

        cap.release()

    return {
        "rows": all_rows,
        "frame_rep": frame_rep_records,
        "tensor_shapes": {k: sorted(list(v)) for k, v in tensor_shapes.items()},
        "viz_items": viz_items,
        "timing": {"det_track_ms": t_det_all, "models": model_timings} if measure_timing else None,
    }


def _safe_spearman_round(xs: List[float], ys: List[float], digits: int = 4) -> Optional[float]:
    if len(xs) < 5 or len(xs) != len(ys):
        return None
    r = _spearman_rho(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    return round(float(r), digits) if r is not None else None


def _compute_cohens_kappa(pairs: List[Tuple[bool, bool]]) -> Optional[float]:
    if not pairs:
        return None
    n = len(pairs)
    tp = sum(1 for a, b in pairs if a and b)
    tn = sum(1 for a, b in pairs if (not a) and (not b))
    fp = sum(1 for a, b in pairs if a and (not b))
    fn = sum(1 for a, b in pairs if (not a) and b)
    po = (tp + tn) / float(n)
    pa = (tp + fp) / float(n)
    pb = (tp + fn) / float(n)
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    if abs(1.0 - pe) < 1e-9:
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1.0 - pe), 4)


def summarize_corpus_and_representation(
    rows: List[Dict[str, Any]],
    frame_rep: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Task 3 & 4: Compute per-arm coverage, failure breakdown, within-image & matched-region
    representation agreement vs DAv2_518 (model reference), and full stratification.
    """
    # Index by (source_id, frame_idx, det_idx) -> {arm: row}
    by_region: Dict[Tuple[str, int, int], Dict[str, Dict[str, Any]]] = {}
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for rw in rows:
        by_region.setdefault((rw["source_id"], rw["frame_idx"], rw["det_idx"]), {})[rw["arm"]] = rw
        by_arm.setdefault(rw["arm"], []).append(rw)

    arm_summary = {}
    for arm, rlist in by_arm.items():
        val_r = [r for r in rlist if r["is_valid"]]
        inval_r = [r for r in rlist if not r["is_valid"]]
        reasons = Counter(rs for r in inval_r for rs in r["failure_reasons"].split("|") if rs)
        # Per source breakdown
        src_bd = {}
        for sid in sorted(set(r["source_id"] for r in rlist)):
            sr = [r for r in rlist if r["source_id"] == sid]
            sv = sum(1 for r in sr if r["is_valid"])
            src_bd[sid] = {"valid": sv, "total": len(sr), "valid_pct": round(100.0 * sv / max(1, len(sr)), 2)}
        arm_summary[arm] = {
            "total": len(rlist),
            "valid": len(val_r),
            "valid_pct": round(100.0 * len(val_r) / max(1, len(rlist)), 2),
            "inlier_ratio_median": round(float(np.median([r["inlier_ratio"] for r in rlist])), 4),
            "ref_mad_median": round(float(np.median([r["ref_mad"] for r in rlist])), 4),
            "pos_frac_median": round(float(np.median([r["pos_frac"] for r in rlist])), 4),
            "unexpected_polarity_count": int(reasons.get("unexpected_residual_polarity", 0)),
            "low_inlier_count": int(reasons.get("low_inlier_ratio", 0)),
            "water_count": int(reasons.get("suspected_water_reflection", 0)),
            "reasons": dict(reasons),
            "per_source": src_bd,
        }

    # Task 3: Frame & Region representation agreement vs DAv2 518 (elliptical_core as main comparator)
    rep_vs_518 = {}
    for res in DAV2_RESOLUTIONS:
        mk = f"dav2_{res}"
        arm_e = f"{mk}_elliptical"
        ref_e = "dav2_518_elliptical"
        # Within-image stats
        w_sp = [f[mk]["within_image_spearman"] for f in frame_rep if mk in f]
        w_mae = [f[mk].get("affine_aligned_mae") for f in frame_rep if mk in f and "affine_aligned_mae" in f[mk]]
        w_nrmse = [f[mk].get("affine_aligned_nrmse_iqr") for f in frame_rep if mk in f and "affine_aligned_nrmse_iqr" in f[mk]]

        pairs_v = []
        sn_cur, sn_518 = [], []
        iq_cur, iq_518 = [], []
        diag_cur, diag_518 = [], []
        pol_flip_valid518 = 0
        lost_valid_reasons = Counter()
        gained_valid_518_reasons = Counter()

        for rk, adict in by_region.items():
            if arm_e not in adict or ref_e not in adict:
                continue
            rc, r5 = adict[arm_e], adict[ref_e]
            vc, v5 = bool(rc["is_valid"]), bool(r5["is_valid"])
            pairs_v.append((vc, v5))
            if v5 and (not vc):
                for rs in rc["failure_reasons"].split("|"):
                    if rs:
                        lost_valid_reasons[rs] += 1
                if "unexpected_residual_polarity" in rc["failure_reasons"]:
                    pol_flip_valid518 += 1
            elif vc and (not v5):
                for rs in r5["failure_reasons"].split("|"):
                    if rs:
                        gained_valid_518_reasons[rs] += 1
            if vc and v5:
                sn_cur.append(rc["scale_norm_p75"])
                sn_518.append(r5["scale_norm_p75"])
                iq_cur.append(rc["frame_iqr_norm_p75"])
                iq_518.append(r5["frame_iqr_norm_p75"])
            diag_cur.append(rc["raw_diag_p75"])
            diag_518.append(r5["raw_diag_p75"])

        tp = sum(1 for a, b in pairs_v if a and b)
        tn = sum(1 for a, b in pairs_v if (not a) and (not b))
        lost = sum(1 for a, b in pairs_v if (not a) and b)
        gained = sum(1 for a, b in pairs_v if a and (not b))

        rep_vs_518[arm_e] = {
            "within_image_spearman_med": round(float(np.median(w_sp)), 4) if w_sp else None,
            "within_image_spearman_p10": round(float(np.percentile(w_sp, 10.0)), 4) if w_sp else None,
            "affine_aligned_mae_med": round(float(np.median(w_mae)), 5) if w_mae else 0.0,
            "affine_aligned_nrmse_iqr_med": round(float(np.median(w_nrmse)), 4) if w_nrmse else 0.0,
            "validity_agreement_pct": round(100.0 * (tp + tn) / max(1, len(pairs_v)), 2),
            "cohens_kappa": _compute_cohens_kappa(pairs_v),
            "confusion_vs_518": {"both_valid": tp, "both_invalid": tn, "lost_vs_518": lost, "gained_vs_518": gained},
            "polarity_flips_on_518_valid": pol_flip_valid518,
            "lost_valid_reasons": dict(lost_valid_reasons),
            "native_518_invalid_when_lowres_valid_reasons": dict(gained_valid_518_reasons),
            "matched_valid_scale_norm_spearman": _safe_spearman_round(sn_cur, sn_518, 4),
            "matched_valid_iqr_norm_spearman": _safe_spearman_round(iq_cur, iq_518, 4),
            "matched_valid_iqr_norm_med_rel_diff_pct": round(
                float(np.median([_guarded_rel_diff_pct(a, b) for a, b in zip(iq_cur, iq_518)])), 2
            ) if iq_cur else None,
        }

    # Document native 518 invalid cases cleanly
    native518_list = by_arm.get("dav2_518_elliptical", [])
    native518_inval = [r for r in native518_list if not r["is_valid"]]
    native518_audit = {
        "total": len(native518_list),
        "invalid_count": len(native518_inval),
        "invalid_reasons": dict(Counter(rs for r in native518_inval for rs in r["failure_reasons"].split("|") if rs)),
        "high_conf_ge65_invalid": sum(1 for r in native518_inval if r["conf"] >= 0.65),
        "high_conf_non_water_invalid_reasons": dict(Counter(
            rs for r in native518_inval if (r["conf"] >= 0.65 and not r["is_water"])
            for rs in r["failure_reasons"].split("|") if rs
        )),
    }

    # Task 4: Full Stratification across key dimensions
    cmp_arms = ["midas_256_full", "dav2_266_elliptical", "dav2_336_elliptical", "dav2_392_elliptical", "dav2_518_elliptical"]
    strata_fields = ["area_bucket", "eff_dim_bucket", "y_bucket", "conf_bucket", "frame_res", "road_env"]
    stratified = {}
    for sf in strata_fields:
        buckets = sorted(set(r[sf] for r in rows))
        stratified[sf] = {}
        for bk in buckets:
            bdict = {}
            for ak in cmp_arms:
                sub = [r for r in by_arm.get(ak, []) if r[sf] == bk]
                if not sub:
                    continue
                sv = sum(1 for r in sub if r["is_valid"])
                pol = sum(1 for r in sub if "unexpected_residual_polarity" in r["failure_reasons"])
                inl_fail = sum(1 for r in sub if "low_inlier_ratio" in r["failure_reasons"])
                bdict[ak] = {
                    "n": len(sub),
                    "valid": sv,
                    "valid_pct": round(100.0 * sv / len(sub), 2),
                    "pol_fail_pct": round(100.0 * pol / len(sub), 2),
                    "inlier_fail_pct": round(100.0 * inl_fail / len(sub), 2),
                    "med_sn_p75_valid": round(float(np.median([r["scale_norm_p75"] for r in sub if r["is_valid"]])), 3) if sv > 0 else None,
                }
            stratified[sf][bk] = bdict

    return {
        "arm_summary": arm_summary,
        "representation_vs_518": rep_vs_518,
        "native_518_failure_audit": native518_audit,
        "stratified_sensitivity": stratified,
    }


def analyze_matched_temporal_stability(
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Task 6: Matched temporal stability evaluation using identical tracks & timestamps across
    `midas_256_full` and `dav2_{266,336,392,518}_elliptical` for both RAW and Validity-Aware EMA.
    """
    cmp_arms = ["midas_256_full", "dav2_266_elliptical", "dav2_336_elliptical", "dav2_392_elliptical", "dav2_518_elliptical"]
    dav2_arms = [a for a in cmp_arms if a.startswith("dav2_")]
    tcfg = TemporalFilterConfig(**{k: v for k, v in FROZEN_ACCEPTANCE_POLICY["temporal_filter_config"].items() if k != "mode"})

    # Group by (arm, source_id, track_id) sorted chronologically
    grouped: Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]] = {a: {} for a in cmp_arms}
    for rw in rows:
        if rw["arm"] in grouped:
            grouped[rw["arm"]].setdefault((rw["source_id"], rw["track_id"]), []).append(rw)

    per_track_csv: List[Dict[str, Any]] = []
    # Run ValidityAwareTrackFilter deterministically per arm x track
    track_metrics: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {a: {} for a in cmp_arms}
    for a in cmp_arms:
        for tkey, obs in grouped[a].items():
            obs.sort(key=lambda x: x["frame_idx"])
            flt_sn = ValidityAwareTrackFilter(mode="validity_aware_ema", cfg=tcfg)
            flt_iq = ValidityAwareTrackFilter(mode="validity_aware_ema", cfg=tcfg)
            raw_sn, ema_sn, ema_iq, areas, y2s = [], [], [], [], []
            n_val = 0
            for o in obs:
                out_sn = flt_sn.update(o["track_id"], o["scale_norm_p75"], o["is_valid"], o["frame_idx"], o["ts"])
                out_iq = flt_iq.update(o["track_id"], o["frame_iqr_norm_p75"], o["is_valid"], o["frame_idx"], o["ts"])
                if o["is_valid"] and o["scale_norm_p75"] is not None:
                    n_val += 1
                    raw_sn.append(float(o["scale_norm_p75"]))
                    areas.append(float(o["area_px"]))
                    y2s.append(float(o["y2_ratio"]))
                if out_sn["filtered_value"] is not None:
                    ema_sn.append(float(out_sn["filtered_value"]))
                if out_iq["filtered_value"] is not None:
                    ema_iq.append(float(out_iq["filtered_value"]))

            n_tot = len(obs)
            dropout_rate = 1.0 - (n_val / float(max(1, n_tot)))

            def _seq_stats(vals: List[float]) -> Dict[str, Optional[float]]:
                if len(vals) < 2:
                    return {"med_abs_diff": None, "mad": None, "cv": None}
                arr = np.asarray(vals, dtype=np.float64)
                med = float(np.median(arr))
                mean = float(np.mean(arr))
                return {
                    "med_abs_diff": round(float(np.median(np.abs(np.diff(arr)))), 4),
                    "mad": round(float(np.median(np.abs(arr - med))), 4),
                    "cv": round(float(np.std(arr) / max(1e-5, abs(mean))), 4),
                }

            rs = _seq_stats(raw_sn)
            es = _seq_stats(ema_sn)
            iqs = _seq_stats(ema_iq)
            rho_a = _safe_spearman_round(raw_sn, areas, 4)
            rho_y = _safe_spearman_round(raw_sn, y2s, 4)

            rec = {
                "arm": a,
                "source_id": tkey[0],
                "track_id": tkey[1],
                "total_obs": n_tot,
                "valid_obs": n_val,
                "dropout_rate": round(dropout_rate, 4),
                "raw_med_delta": rs["med_abs_diff"],
                "raw_mad": rs["mad"],
                "raw_cv": rs["cv"],
                "ema_med_delta": es["med_abs_diff"],
                "ema_mad": es["mad"],
                "ema_cv": es["cv"],
                "ema_iqr_norm_cv": iqs["cv"],
                "raw_rho_area": rho_a,
                "raw_rho_y2": rho_y,
            }
            track_metrics[a][tkey] = rec
            if n_tot >= 3:
                per_track_csv.append(rec)

    # Identify common valid tracks (`>= 5` valid obs) across ALL 4 DAv2 arms AND across ALL 5 arms
    all_tkeys = sorted(set(k for a in cmp_arms for k in track_metrics[a].keys()))
    common_dav2_tracks = [k for k in all_tkeys if all(k in track_metrics[a] and track_metrics[a][k]["valid_obs"] >= 5 for a in dav2_arms)]
    common_all5_tracks = [k for k in common_dav2_tracks if (k in track_metrics["midas_256_full"] and track_metrics["midas_256_full"][k]["valid_obs"] >= 5)]
    active_ge3_tracks = [k for k in all_tkeys if all(k in track_metrics[a] and track_metrics[a][k]["total_obs"] >= 3 for a in cmp_arms)]

    def _agg_subset(arm: str, tkeys: List[Tuple[str, int]]) -> Dict[str, Any]:
        sub = [track_metrics[arm][k] for k in tkeys if k in track_metrics[arm]]
        if not sub:
            return {"tracks": 0}
        def _med_field(f: str) -> Optional[float]:
            vs = [x[f] for x in sub if x[f] is not None]
            return round(float(np.median(vs)), 4) if vs else None
        def _mean_field(f: str) -> Optional[float]:
            vs = [x[f] for x in sub if x[f] is not None]
            return round(float(np.mean(vs)), 4) if vs else None
        return {
            "tracks": len(sub),
            "sum_valid_obs": int(sum(x["valid_obs"] for x in sub)),
            "sum_total_obs": int(sum(x["total_obs"] for x in sub)),
            "mean_dropout_rate": _mean_field("dropout_rate"),
            "raw_scale_norm": {
                "med_frame_delta": _med_field("raw_med_delta"),
                "med_mad": _med_field("raw_mad"),
                "mean_cv": _mean_field("raw_cv"),
                "mean_rho_area": _mean_field("raw_rho_area"),
                "mean_rho_y2": _mean_field("raw_rho_y2"),
            },
            "filtered_validity_ema": {
                "med_frame_delta": _med_field("ema_med_delta"),
                "med_mad": _med_field("ema_mad"),
                "mean_cv": _mean_field("ema_cv"),
                "mean_iqr_norm_cv": _mean_field("ema_iqr_norm_cv"),
            },
        }

    summary = {
        "common_dav2_valid_tracks_ge5": len(common_dav2_tracks),
        "common_all5_valid_tracks_ge5": len(common_all5_tracks),
        "all_tracked_ge3_obs_count": len(active_ge3_tracks),
        "matched_dav2_tracks": {a: _agg_subset(a, common_dav2_tracks) for a in cmp_arms},
        "matched_all5_tracks": {a: _agg_subset(a, common_all5_tracks) for a in cmp_arms},
        "all_ge3_dropout_summary": {
            a: {
                "tracks": len(active_ge3_tracks),
                "mean_dropout_rate": round(float(np.mean([track_metrics[a][k]["dropout_rate"] for k in active_ge3_tracks])), 4) if active_ge3_tracks else None,
                "sum_valid_obs": sum(track_metrics[a][k]["valid_obs"] for k in active_ge3_tracks),
                "sum_total_obs": sum(track_metrics[a][k]["total_obs"] for k in active_ge3_tracks),
            }
            for a in cmp_arms
        },
    }
    return summary, per_track_csv


def measure_isolated_vram_per_config(
    adapters: Dict[str, Any],
    sample_img_360: np.ndarray,
    sample_img_720: np.ndarray,
) -> Dict[str, Any]:
    """Task 7: Separate GPU memory audit completely isolated from timing measurements."""
    if not torch.cuda.is_available():
        return {}
    out = {}
    for mk, ad in adapters.items():
        res_entry = {"load_ms": ad.load_ms, "param_count": ad.param_count}
        for rtag, img in (("360p_640x360", sample_img_360), ("720p_1280x720", sample_img_720)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            pr = ad.predict(img)
            torch.cuda.synchronize()
            res_entry[rtag] = {
                "network_hw": list(pr.network_input_hw),
                "base_alloc_mb": round(base_mb, 2),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
                "delta_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2) - base_mb, 2),
            }
        out[mk] = res_entry
    return out


def evaluate_pareto_and_policy(
    summary_exp: Dict[str, Any],
    summary_hold: Dict[str, Any],
    timing: Dict[str, Any],
    vram: Dict[str, Any],
) -> Dict[str, Any]:
    """Task 9: Evaluate each resolution against the frozen provisional engineering policy."""
    pol = FROZEN_ACCEPTANCE_POLICY
    ref_arm = "dav2_518_elliptical"
    ref_cov = summary_exp["arm_summary"][ref_arm]["valid_pct"]
    ref_urb = summary_exp["arm_summary"][ref_arm]["per_source"].get("rawSurvey_720p_f0_119", {}).get("valid_pct", 0.0)
    ref_small_s = summary_exp["stratified_sensitivity"]["area_bucket"].get("small_0.5_to_1.5pct", {}).get(ref_arm, {}).get("valid", 0)
    ref_tiny_s = summary_exp["stratified_sensitivity"]["area_bucket"].get("tiny_lt_0.5pct", {}).get(ref_arm, {}).get("valid", 0)
    ref_small_tot_valid = ref_small_s + ref_tiny_s

    ref_e2e_med = compute_latency_stats(
        [v for b in ("1", "2", "3", "4+") for v in timing["models"]["dav2_518"]["e2e_buckets"][b]]
    )["median_ms"]

    pareto_rows = []
    for key in ["midas_256", "dav2_266", "dav2_336", "dav2_392", "dav2_518"]:
        arm = "midas_256_full" if key == "midas_256" else f"{key}_elliptical"
        exp_a = summary_exp["arm_summary"][arm]
        hold_a = summary_hold["arm_summary"][arm]
        t_m = timing["models"][key]
        d_stat = compute_latency_stats(t_m["tot"])
        e2e_vals = [v for b in ("1", "2", "3", "4+") for v in t_m["e2e_buckets"][b]]
        e2e_stat = compute_latency_stats(e2e_vals)

        cur_cov = exp_a["valid_pct"]
        cov_ret = round(cur_cov / max(1e-5, ref_cov), 4)
        cur_urb = exp_a["per_source"].get("rawSurvey_720p_f0_119", {}).get("valid_pct", 0.0)
        urb_ret = round(cur_urb / max(1e-5, ref_urb), 4)

        cur_sm = (
            summary_exp["stratified_sensitivity"]["area_bucket"].get("small_0.5_to_1.5pct", {}).get(arm, {}).get("valid", 0)
            + summary_exp["stratified_sensitivity"]["area_bucket"].get("tiny_lt_0.5pct", {}).get(arm, {}).get("valid", 0)
        )
        sm_ret = round(cur_sm / max(1, ref_small_tot_valid), 4)

        rep = summary_exp["representation_vs_518"].get(arm, {})
        sp_img = 1.0 if key == "dav2_518" else rep.get("within_image_spearman_med")
        sp_match = 1.0 if key == "dav2_518" else rep.get("matched_valid_scale_norm_spearman")
        val_agree = 100.0 if key == "dav2_518" else rep.get("validity_agreement_pct", 0.0)

        checks = {}
        if key.startswith("dav2_"):
            checks = {
                "cov_retention_ge_90pct": bool(cov_ret >= pol["min_coverage_retention_vs_518"]),
                "matched_spearman_ge_0.85": bool((sp_match or 0.0) >= pol["min_matched_spearman_vs_518"]),
                "validity_agree_ge_88pct": bool((val_agree or 0.0) >= pol["min_validity_agreement_vs_518"] * 100.0),
                "urban_retention_ge_85pct": bool(urb_ret >= pol["min_urban_retention_vs_518"]),
                "small_pothole_retention_ge_85pct": bool(sm_ret >= pol["min_small_pothole_retention_vs_518"]),
                "lower_latency_than_518": bool((e2e_stat["mean_ms"] or 1e9) < (ref_e2e_med or 1e9)),
                "meets_realtime_e2e_le_45ms": bool((e2e_stat["mean_ms"] or 1e9) <= pol["target_realtime_e2e_ms"]),
            }

        pareto_rows.append({
            "config_key": key,
            "arm": arm,
            "exp_360f_valid_pct": cur_cov,
            "holdout_valid_pct": hold_a["valid_pct"],
            "coverage_retention_vs_518": cov_ret,
            "urban_720p_valid_pct": cur_urb,
            "urban_retention_vs_518": urb_ret,
            "small_pothole_valid_count": cur_sm,
            "small_retention_vs_518": sm_ret,
            "within_image_spearman_vs_518": sp_img,
            "matched_valid_spearman_vs_518": sp_match,
            "validity_agreement_pct_vs_518": val_agree,
            "depth_only_mean_ms": d_stat["mean_ms"],
            "depth_only_p95_ms": d_stat["p95_ms"],
            "active_e2e_mean_ms": e2e_stat["mean_ms"],
            "active_e2e_median_ms": e2e_stat["median_ms"],
            "active_e2e_p95_ms": e2e_stat["p95_ms"],
            "active_e2e_fps": round(1000.0 / e2e_stat["mean_ms"], 2) if e2e_stat.get("mean_ms") else None,
            "vram_360p_peak_alloc_mb": vram.get(key, {}).get("360p_640x360", {}).get("peak_alloc_mb"),
            "vram_720p_peak_alloc_mb": vram.get(key, {}).get("720p_1280x720", {}).get("peak_alloc_mb"),
            "policy_checks": checks,
        })

    return {"policy": pol, "pareto_table": pareto_rows}


def generate_phase6b_visuals(
    pareto_rows: List[Dict[str, Any]],
    timing: Dict[str, Any],
    summary_exp: Dict[str, Any],
    viz_items: List[Dict[str, Any]],
) -> List[str]:
    """Task 10 & 11: Resolution vs coverage/latency/Pareto charts + 11-panel visual comparisons."""
    paths: List[str] = []

    # Chart 1: Resolution vs Coverage + Resolution vs Latency (two subplots)
    dav_rows = [r for r in pareto_rows if r["config_key"].startswith("dav2_")]
    mid_row = [r for r in pareto_rows if r["config_key"] == "midas_256"][0]
    labels = [r["config_key"].replace("dav2_", "DAv2 ") for r in dav_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=130)
    axes[0].plot(x, [r["exp_360f_valid_pct"] for r in dav_rows], "o-", color="#16a34a", linewidth=2.2, label="DAv2 Exploratory 360f (%)")
    axes[0].plot(x, [r["holdout_valid_pct"] for r in dav_rows], "s--", color="#0284c7", linewidth=2.0, label="DAv2 Holdout Untouched (%)")
    axes[0].plot(x, [r["urban_720p_valid_pct"] for r in dav_rows], "^:", color="#9333ea", linewidth=2.0, label="DAv2 Urban 720p (%)")
    axes[0].axhline(mid_row["exp_360f_valid_pct"], color="#dc2626", linestyle="-.", linewidth=1.6, label=f"MiDaS 256 Baseline ({mid_row['exp_360f_valid_pct']}%)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Valid Measurement Coverage (%)"); axes[0].set_ylim(0, 85)
    axes[0].set_title("DAv2 Resolution vs Valid Measurement Coverage")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].bar(x - 0.17, [r["depth_only_mean_ms"] for r in dav_rows], width=0.34, color="#2563eb", label="Depth Stage Only (ms)")
    axes[1].bar(x + 0.17, [r["active_e2e_mean_ms"] for r in dav_rows], width=0.34, color="#f59e0b", label="Active E2E Mean (ms)")
    axes[1].axhline(mid_row["active_e2e_mean_ms"], color="#dc2626", linestyle="--", linewidth=1.6, label=f"MiDaS 256 Active E2E ({mid_row['active_e2e_mean_ms']} ms)")
    axes[1].axhline(45.0, color="#16a34a", linestyle=":", linewidth=1.6, label="45 ms Real-Time Target (~22 FPS)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Latency (ms)"); axes[1].set_title("DAv2 Resolution vs RTX 2050 Latency")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)

    p1 = os.path.join(VIZ_DIR, "phase6b_res_vs_coverage_and_latency.png")
    plt.savefig(p1, bbox_inches="tight"); plt.close(fig)
    paths.append(p1)

    # Chart 2: Pareto Frontier (Coverage vs Active E2E Latency & Proposals Scaling 0..4+)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=130)
    for r in pareto_rows:
        col = "#dc2626" if r["config_key"] == "midas_256" else "#16a34a"
        mk_sz = (r["vram_720p_peak_alloc_mb"] or 400.0) * 0.45
        axes[0].scatter(r["active_e2e_mean_ms"], r["exp_360f_valid_pct"], s=mk_sz, color=col, alpha=0.82, edgecolors="black")
        axes[0].annotate(r["config_key"], (r["active_e2e_mean_ms"], r["exp_360f_valid_pct"]), textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    axes[0].axvline(45.0, color="#16a34a", linestyle=":", linewidth=1.4, label="45 ms RT Target")
    axes[0].set_xlabel("Active E2E Mean Latency (ms) [Lower Faster]")
    axes[0].set_ylabel("Valid Coverage (%) [Higher Better]")
    axes[0].set_title("Pareto Trade-Off (Bubble Area ~ Peak VRAM MB)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    cats = ["0", "1", "2", "3", "4+"]
    colors = {"midas_256": "#dc2626", "dav2_266": "#0ea5e9", "dav2_336": "#16a34a", "dav2_392": "#d97706", "dav2_518": "#7c3aed"}
    for mk, col in colors.items():
        ys = [compute_latency_stats(timing["models"][mk]["e2e_buckets"][c]).get("mean_ms", np.nan) for c in cats]
        axes[1].plot(cats, ys, "o-", color=col, linewidth=1.8, label=mk)
    axes[1].set_xlabel("Detector Proposals per Frame"); axes[1].set_ylabel("Mean Frame Latency (ms)")
    axes[1].set_title("RTX 2050 E2E Latency Across Proposal Counts (0..4+)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    p2 = os.path.join(VIZ_DIR, "phase6b_pareto_and_scaling.png")
    plt.savefig(p2, bbox_inches="tight"); plt.close(fig)
    paths.append(p2)

    # Chart 3: Multi-resolution matched visual diagnostic panels
    model_order = ["midas_256", "dav2_266", "dav2_336", "dav2_392", "dav2_518"]
    arm_order = ["midas_256_full", "dav2_266_elliptical", "dav2_336_elliptical", "dav2_392_elliptical", "dav2_518_elliptical"]
    lat_lookup = {r["config_key"]: r["depth_only_mean_ms"] for r in pareto_rows}

    for item in viz_items:
        rgb = cv2.cvtColor(item["img_bgr"], cv2.COLOR_BGR2RGB)
        bx = [int(round(v)) for v in item["box"]]
        fig = plt.figure(figsize=(18.5, 9.8), dpi=130)
        gs = GridSpec(2, 6, figure=fig, hspace=0.32, wspace=0.22)

        # Panel (0, 0): Original image + ROI bbox
        ax0 = fig.add_subplot(gs[0, 0])
        vis = rgb.copy()
        cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (34, 197, 94), 3)
        ax0.imshow(vis)
        ax0.set_title(f"RGB + Box (conf={item['conf']:.2f})", fontsize=9)
        ax0.axis("off")

        # Panels (0, 1..5): Relative Depth Maps across MiDaS 256 + DAv2 266..518
        for col_i, mk in enumerate(model_order, start=1):
            ax = fig.add_subplot(gs[0, col_i])
            im = ax.imshow(item["depth_maps"][mk], cmap="inferno")
            nhw = item["net_hws"][mk]
            ax.set_title(f"{mk}\nnet={nhw[0]}x{nhw[1]} | {lat_lookup.get(mk, 0):.1f}ms", fontsize=8.5)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis("off")

        # Panels (1, 1..5): Signed Cavity Residuals + Validity status per arm
        for col_i, ak in enumerate(arm_order, start=1):
            ax = fig.add_subplot(gs[1, col_i])
            ao = item["arms"][ak]
            rmap = ao["arrays"]["signed_residual_map"]
            vm = max(1e-4, float(np.nanpercentile(np.abs(rmap), 98))) if np.any(np.isfinite(rmap)) else 1.0
            imr = ax.imshow(rmap, cmap="coolwarm", vmin=-vm, vmax=vm)
            status_str = "VALID" if ao["is_valid"] else "|".join(ao["failure_reasons"])[:26]
            inl = ao["ref_diagnostics"]["inlier_ratio"]
            snp = ao["core_cavity_stats"]["scale_normalized_p75"]
            ax.set_title(f"{ak}\n{status_str}\ninl={inl:.2f} snP75={snp if snp is not None else 'N/A'}", fontsize=8.0)
            plt.colorbar(imr, ax=ax, fraction=0.046)
            ax.axis("off")

        # Panel (1, 0): Reference Diagnostics Summary Card
        ax_txt = fig.add_subplot(gs[1, 0]); ax_txt.axis("off")
        lines = ["Ref-Plane & Validity (Relative Units Only):", f"Frame: {item['source_id']} f{item['frame_idx']}"]
        for ak in arm_order:
            ao = item["arms"][ak]
            lines.append(f"{ak[:16]:16s}: {'OK' if ao['is_valid'] else 'REJ'} MAD={ao['ref_diagnostics']['ref_fit_mad']:.4f}")
        ax_txt.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=7.6, bbox=dict(boxstyle="round", facecolor="#f8fafc"))

        fig.suptitle(f"Phase 6B Multi-Resolution Depth Comparison | {item['tag']} (Relative Inverse Depth Only)", fontsize=11.5, fontweight="bold")
        vp = os.path.join(VIZ_DIR, f"phase6b_panel_{item['tag']}.png")
        plt.savefig(vp, bbox_inches="tight"); plt.close(fig)
        paths.append(vp)

    return paths


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    det_model = YOLO(DEFAULT_DET_WEIGHTS)
    midas = MiDaSSmallAdapter(device=device_str, use_fp16=True)
    adapters = {
        "midas_256": midas,
        "dav2_266": DepthAnythingV2SmallAdapter(device=device_str, input_size=266, use_fp16=True),
        "dav2_336": DepthAnythingV2SmallAdapter(device=device_str, input_size=336, use_fp16=True),
        "dav2_392": DepthAnythingV2SmallAdapter(device=device_str, input_size=392, use_fp16=True),
        "dav2_518": DepthAnythingV2SmallAdapter(device=device_str, input_size=518, use_fp16=True),
    }

    # Synchronized warmup across all detectors & adapters before any benchmark
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    _ = det_model.predict(dummy, imgsz=640, conf=0.25, iou=0.70, device=0 if device_str == "cuda" else "cpu", verbose=False)
    for ad in adapters.values():
        ad.warmup(360, 640)

    # Task 1: Environment & Checkpoint verification
    env_audit = _collect_env_and_checkpoint_audit(adapters["dav2_518"], midas)

    # Stage A (Task 5 & Task 4 Perturbation): 36 Dev Frames
    print("[Phase 6B] Stage A: Running Dev-36 debug + ±8% perturbation scaling...")
    dev36_res = run_dev36_debug_and_perturbation(det_model, adapters)

    # Representative visual diagnostic frame targets (Urban dry pit, Water puddle, Shadowed Rangoon, Distant/Small pothole)
    viz_targets = {
        ("rawSurvey_720p_f0_119", 0, 0): "urban_dry_pit_f0000",
        ("rawSurvey_720p_f0_119", 21, 0): "water_puddle_f0021",
        ("rangoonRoad_f200_319", 211, 0): "shadowed_rangoon_f0211",
        ("ruralRoad_f0_119", 45, 0): "rural_multi_pothole_f0045",
        ("ruralRoad_f0_119", 37, 1): "small_distant_rural_f0037",
    }

    # Stage B (Frozen Evaluation): 360f Phase 6A Exploratory Corpus (with clean timing pass)
    print("[Phase 6B] Stage B: Running Exploratory 360f corpus + RTX 2050 timing...")
    exp_out = _evaluate_video_corpus(EXPLORATORY_SOURCES, det_model, adapters, capture_viz_tags=viz_targets, measure_timing=True)
    summary_exp = summarize_corpus_and_representation(exp_out["rows"], exp_out["frame_rep"])
    temporal_exp, track_csv_exp = analyze_matched_temporal_stability(exp_out["rows"])

    # Stage C (Untouched Holdout Split, Zero Tuning): Rangoon f785..1004 + Rural f120..201 + TestClip
    print("[Phase 6B] Stage C: Running Untouched Holdout evaluation...")
    hold_out = _evaluate_video_corpus(HOLDOUT_SOURCES, det_model, adapters, capture_viz_tags=None, measure_timing=False)
    summary_hold = summarize_corpus_and_representation(hold_out["rows"], hold_out["frame_rep"])
    temporal_hold, track_csv_hold = analyze_matched_temporal_stability(hold_out["rows"])

    # Task 7: Isolated VRAM instrumentation pass + FP32 vs FP16 verification on subset of frames
    print("[Phase 6B] Task 7: Measuring isolated VRAM + FP32 vs FP16 numerical fidelity...")
    sample_360 = exp_out["viz_items"][-1]["img_bgr"] if exp_out["viz_items"] else np.zeros((360, 640, 3), dtype=np.uint8)
    sample_720 = exp_out["viz_items"][0]["img_bgr"] if exp_out["viz_items"] else np.zeros((720, 1280, 3), dtype=np.uint8)
    vram_stats = measure_isolated_vram_per_config(adapters, sample_360, sample_720)

    fp_frames = [
        {"img_bgr": item["img_bgr"], "boxes": [tuple(item["box"])]}
        for item in exp_out["viz_items"][:4]
    ]
    # Free main adapters first so FP32 + FP16 side-by-side runs comfortably inside 4GB RTX 2050 VRAM
    del adapters
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fp32_vs_fp16 = benchmark_fp32_vs_fp16(fp_frames, input_sizes=DAV2_RESOLUTIONS, device=device_str) if fp_frames else {}

    # Task 8: Optional Acceleration Feasibility PoC
    print("[Phase 6B] Task 8: Probing inference_mode, torch.compile, and ONNX/ORT backends...")
    accel_frames = [f["img_bgr"] for f in fp_frames] or [sample_360]
    accel_audit = run_acceleration_feasibility(accel_frames, input_sizes=DAV2_RESOLUTIONS, device=device_str)

    # Task 9: Provisional Acceptance Policy & Pareto Table
    pareto = evaluate_pareto_and_policy(summary_exp, summary_hold, exp_out["timing"], vram_stats)

    # Task 7 timing tables formatted cleanly
    timing_summary = {
        "detector_bytetrack_640": compute_latency_stats(exp_out["timing"]["det_track_ms"]),
        "models": {
            mk: {
                "preprocess": compute_latency_stats(md["pre"]),
                "gpu_inference": compute_latency_stats(md["inf"]),
                "output_resize": compute_latency_stats(md["res"]),
                "depth_only_total": compute_latency_stats(md["tot"]),
                "extraction_elliptical": compute_latency_stats(md["ext_ell"]),
                "active_e2e_all": compute_latency_stats([v for b in ("1", "2", "3", "4+") for v in md["e2e_buckets"][b]]),
                "e2e_by_proposal_count": {b: compute_latency_stats(md["e2e_buckets"][b]) for b in ("0", "1", "2", "3", "4+")},
            }
            for mk, md in exp_out["timing"]["models"].items()
        },
        "vram_isolated": vram_stats,
    }

    # Task 10 & 11: Generate Plots & Multi-panel visuals
    viz_paths = generate_phase6b_visuals(pareto["pareto_table"], exp_out["timing"], summary_exp, exp_out["viz_items"])

    # Save CSVs
    all_rows = exp_out["rows"] + hold_out["rows"]
    csv_reg_path = os.path.join(BENCH_DIR, "phase6b_per_region.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_reg_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                rw = dict(r)
                rw["box_xyxy"] = json.dumps(rw["box_xyxy"])
                w.writerow(rw)

    all_tracks_csv = [dict({"split": "exploratory_360f"}, **r) for r in track_csv_exp] + [dict({"split": "holdout_untouched"}, **r) for r in track_csv_hold]
    csv_trk_path = os.path.join(BENCH_DIR, "phase6b_per_track.csv")
    if all_tracks_csv:
        tfields = list(all_tracks_csv[0].keys())
        with open(csv_trk_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=tfields)
            w.writeheader()
            w.writerows(all_tracks_csv)

    # Save JSON
    out_json = {
        "experiment": "Phase 6B Depth Anything V2 Resolution Scaling and Computational Optimization",
        "task1_env_and_checkpoint_audit": env_audit,
        "task5_dev36_debug_and_perturbation": dev36_res,
        "observed_network_tensor_shapes_corpus": exp_out["tensor_shapes"],
        "exploratory_360f_evaluation": summary_exp,
        "holdout_untouched_evaluation": summary_hold,
        "task6_temporal_stability_exploratory": temporal_exp,
        "task6_temporal_stability_holdout": temporal_hold,
        "task7_gpu_performance_and_vram": timing_summary,
        "task7_fp32_vs_fp16_audit": fp32_vs_fp16,
        "task8_acceleration_feasibility": accel_audit,
        "task9_pareto_and_policy": pareto,
        "task10_visualization_paths": viz_paths,
        "csv_outputs": {"per_region": csv_reg_path, "per_track": csv_trk_path},
    }
    json_path = os.path.join(BENCH_DIR, "phase6b_resolution_scaling.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(out_json, jf, indent=2)

    print("=== PHASE 6B BENCHMARK COMPLETE ===")
    for pr in pareto["pareto_table"]:
        print(
            f"{pr['config_key']:10s} | exp={pr['exp_360f_valid_pct']:5.2f}% | hold={pr['holdout_valid_pct']:5.2f}% | "
            f"urb={pr['urban_720p_valid_pct']:5.2f}% | sm={pr['small_retention_vs_518']*100:5.1f}% | "
            f"depth={pr['depth_only_mean_ms']:6.2f}ms | e2e={pr['active_e2e_mean_ms']:6.2f}ms ({pr['active_e2e_fps']} FPS) | "
            f"checks={pr['policy_checks']}"
        )


if __name__ == "__main__":
    main()
