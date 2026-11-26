"""
========================================================================================
RoadVision AI -- Phase 7B: Dynamic Horizon Estimation & Camera Geometry Benchmark
========================================================================================
Covers Tasks 5, 6, 7, 8, 9, and 10:
  - Task 5: Synthetic projective verification across 10 scenarios + negative controls
    + temporal step/dropout test across `lsd_320`, `hough_320`, `hybrid_320`, `lsd_fullres`.
  - Task 6: Real-video exploratory evaluation across 5 local footage sources
    (`ruralRoad_202f`, `rangoonRoad_f200_399`, `rawSurvey_160f_10fps`,
     `phonePotholes_70f`, `testClip_25f`) comparing policies `A_fixed`, `B_dynamic_raw`,
    `C_dynamic_smoothed`, `D_dynamic_fallback`. Clearly labeled exploratory (no GT labels).
  - Task 7: Exact `MetricDepthCalibrator` analytic & numerical sensitivity sweeps
    + matched-track approach-drift test (DAv2 336 fixed vs dynamic horizon).
  - Task 9: Stage-level latency benchmark (320px vs full-res, selective refresh cadence).
  - Task 10: Publication-grade diagnostic figures, JSON results, and per-frame CSV.

Zero production file edits; zero model training; zero metric accuracy claims.
========================================================================================
"""

import os
import sys
import csv
import glob
import json
import math
import time
import platform
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.dynamic_horizon_estimator import (
    HorizonEstimatorConfig,
    HorizonPolicyConfig,
    estimate_vanishing_point,
    DynamicHorizonTracker,
)
from experiments.roi_guided_segmentation import DEFAULT_DET_WEIGHTS
from experiments.optimized_relative_depth import OptimizedDepthConfig, build_and_sample_annular_region, fit_ransac_optimized
from experiments.core_aware_cavity_extraction import CoreExtractionConfig, extract_cavity_with_strategy
from experiments.benchmark_adaptive_depth_scheduler import FastDAv2Runner, FROZEN_PHASE7A_CONFIG, _stats
from experiments.benchmark_optimized_relative_depth import _spearman_rho
from tests.test_dynamic_horizon_estimator import render_synthetic_perspective_road

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "phase7b_viz")

REAL_SOURCES = [
    {
        "source_id": "ruralRoad_202f",
        "kind": "video",
        "path": "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
        "start_frame": 0,
        "count": 202,
        "assumed_fps": None,
        "provenance": "Pre-examined (Phase 5C/6A/6B/7A); 25 FPS 640x360 rural unpaved road",
    },
    {
        "source_id": "rangoonRoad_f200_399",
        "kind": "video",
        "path": "Watch Video_ Terrible potholes in Rangoon Road.mp4",
        "start_frame": 200,
        "count": 200,
        "assumed_fps": None,
        "provenance": "Partially examined urban corridor (f200-319 in Phase 6A/7A + f320-399); 25 FPS 640x360 heavy traffic/shadows",
    },
    {
        "source_id": "rawSurvey_160f_10fps",
        "kind": "video",
        "path": "roadvision-dashboard/public/videos/raw_survey.mp4",
        "start_frame": 0,
        "count": 160,
        "assumed_fps": None,
        "provenance": "Pre-examined (Phase 5C/6A/7A); native 10 FPS 1280x720 urban survey road",
    },
    {
        "source_id": "phonePotholes_70f",
        "kind": "image_dir",
        "path": "dataset_phone_potholes/images",
        "start_frame": 0,
        "count": 70,
        "assumed_fps": 25.0,
        "provenance": "Sequential handheld phone frames frame_0000..0069.jpg (synthetic 25 FPS timestamps; pitch/handheld jitter stress)",
    },
    {
        "source_id": "testClip_25f",
        "kind": "video",
        "path": "uploads/upload_1794546659_test_clip.mp4",
        "start_frame": 0,
        "count": 25,
        "assumed_fps": None,
        "provenance": "Short 25-frame clip; 25 FPS smoke/sanity footage",
    },
]

ESTIMATOR_VARIANTS = {
    "lsd_320": HorizonEstimatorConfig(detector_mode="lsd", process_width=320),
    "hough_320": HorizonEstimatorConfig(detector_mode="hough", process_width=320),
    "hybrid_320": HorizonEstimatorConfig(detector_mode="hybrid", process_width=320),
    "lsd_fullres": HorizonEstimatorConfig(detector_mode="lsd", process_width=None),
}


def _load_source_frames(src: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    full_path = os.path.join(BASE_DIR, src["path"])
    if src["kind"] == "video":
        cap = cv2.VideoCapture(full_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or src.get("assumed_fps") or 25.0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src["start_frame"])
        for off in range(src["count"]):
            ok, img = cap.read()
            if not ok or img is None:
                break
            frames.append({
                "source_id": src["source_id"],
                "frame_idx": src["start_frame"] + off,
                "seq_pos": off,
                "ts_sec": round(off / fps, 6),
                "img_bgr": img,
            })
        cap.release()
    else:
        fps = float(src.get("assumed_fps") or 25.0)
        paths = sorted(glob.glob(os.path.join(full_path, "*.jpg")))[src["start_frame"]:src["start_frame"] + src["count"]]
        W, H = 0, 0
        for off, p in enumerate(paths):
            img = cv2.imread(p)
            if img is None:
                continue
            H, W = img.shape[:2]
            frames.append({
                "source_id": src["source_id"],
                "frame_idx": off,
                "seq_pos": off,
                "ts_sec": round(off / fps, 6),
                "img_bgr": img,
            })
    meta = {
        "source_id": src["source_id"],
        "n_frames": len(frames),
        "fps": round(fps, 3),
        "wh": [W, H],
        "provenance": src["provenance"],
    }
    return frames, meta


def _render_occluded_road(w: int = 640, h: int = 360, vp_norm=(0.50, 0.38)) -> np.ndarray:
    img = render_synthetic_perspective_road(w, h, vp_norm=vp_norm, seed=19)
    # Draw large truck/vehicle rear occluding center-mid region of road
    cv2.rectangle(img, (int(0.28 * w), int(0.42 * h)), (int(0.72 * w), int(0.84 * h)), (45, 50, 55), -1)
    cv2.rectangle(img, (int(0.28 * w), int(0.42 * h)), (int(0.72 * w), int(0.84 * h)), (180, 180, 185), 2)
    return img


def run_synthetic_suite() -> Dict[str, Any]:
    """Task 5: Controlled synthetic projective evaluation across 10 positive + 4 negative controls."""
    positive_cases: List[Dict[str, Any]] = []
    # 1. Multi-pitch
    for vy in (0.25, 0.33, 0.40, 0.48, 0.56):
        positive_cases.append({"group": "multi_pitch", "label": f"vy_{vy:.2f}", "wh": (640, 360),
                               "vp_gt": (0.50, vy), "img": render_synthetic_perspective_road(640, 360, vp_norm=(0.50, vy), seed=101)})
    # 2. Angled horizontal road direction
    for vx in (0.25, 0.38, 0.50, 0.62, 0.75):
        positive_cases.append({"group": "angled_road", "label": f"vx_{vx:.2f}", "wh": (640, 360),
                               "vp_gt": (vx, 0.40), "img": render_synthetic_perspective_road(640, 360, vp_norm=(vx, 0.40), seed=102)})
    # 3. Resolution scaling (360p vs 720p)
    for w, h in ((640, 360), (1280, 720)):
        positive_cases.append({"group": "resolution_scaling", "label": f"{w}x{h}", "wh": (w, h),
                               "vp_gt": (0.54, 0.36), "img": render_synthetic_perspective_road(w, h, vp_norm=(0.54, 0.36), seed=103)})
    # 4. Missing lane markings (curbs only)
    positive_cases.append({"group": "missing_markings", "label": "curbs_only", "wh": (640, 360),
                           "vp_gt": (0.50, 0.40), "img": render_synthetic_perspective_road(640, 360, vp_norm=(0.50, 0.40), missing_markings=True, seed=104)})
    # 5. Partial occlusion (truck blocking center road)
    positive_cases.append({"group": "partial_occlusion", "label": "center_truck", "wh": (640, 360),
                           "vp_gt": (0.50, 0.38), "img": _render_occluded_road(640, 360, (0.50, 0.38))})
    # 6. Spurious clutter (vertical poles + horizontal cracks)
    positive_cases.append({"group": "spurious_clutter", "label": "poles_cracks", "wh": (640, 360),
                           "vp_gt": (0.46, 0.42), "img": render_synthetic_perspective_road(640, 360, vp_norm=(0.46, 0.42), spurious_clutter=True, seed=106)})
    # 7. Harsh transverse shadows
    positive_cases.append({"group": "shadows", "label": "shadow_band", "wh": (640, 360),
                           "vp_gt": (0.52, 0.38), "img": render_synthetic_perspective_road(640, 360, vp_norm=(0.52, 0.38), shadow_band=True, seed=107)})
    # 8. Heavy Gaussian noise
    for sig in (12.0, 22.0):
        positive_cases.append({"group": "heavy_noise", "label": f"sigma_{int(sig)}", "wh": (640, 360),
                               "vp_gt": (0.50, 0.40), "img": render_synthetic_perspective_road(640, 360, vp_norm=(0.50, 0.40), noise_sigma=sig, seed=108)})
    # 9. Mild camera roll
    for rdeg in (-6.0, 6.0):
        positive_cases.append({"group": "camera_roll", "label": f"roll_{int(rdeg)}deg", "wh": (640, 360),
                               "vp_gt": (0.50, 0.40), "img": render_synthetic_perspective_road(640, 360, vp_norm=(0.50, 0.40), roll_deg=rdeg, seed=109)})
    # 10. Incompatible competing secondary VP
    positive_cases.append({"group": "incompatible_dual_vp", "label": "cross_street_vp", "wh": (640, 360),
                           "vp_gt": (0.50, 0.40), "img": render_synthetic_perspective_road(640, 360, vp_norm=(0.50, 0.40), incompatible_second_vp=True, seed=110)})

    # Negative controls (must NOT emit VALID; ideally UNAVAILABLE or LOW_CONFIDENCE)
    neg_controls: List[Dict[str, Any]] = []
    neg_controls.append({"label": "blank_asphalt", "img": np.full((360, 640, 3), 100, dtype=np.uint8)})
    par_img = np.full((360, 640, 3), 80, dtype=np.uint8)
    for off in (-120, -40, 40, 120):
        cv2.line(par_img, (220 + offset_x(off), 100), (120 + offset_x(off), 330), (230, 230, 230), 2)
    neg_controls.append({"label": "parallel_lines", "img": par_img})
    vert_img = np.full((360, 640, 3), 85, dtype=np.uint8)
    for vx in (120, 240, 360, 480, 560):
        cv2.line(vert_img, (vx, 40), (vx, 340), (225, 225, 225), 2)
    neg_controls.append({"label": "vertical_clutter_only", "img": vert_img})
    horiz_img = np.full((360, 640, 3), 85, dtype=np.uint8)
    for hy in (100, 160, 220, 280):
        cv2.line(horiz_img, (60, hy), (580, hy), (225, 225, 225), 2)
    neg_controls.append({"label": "horizontal_clutter_only", "img": horiz_img})

    variant_results: Dict[str, Any] = {}
    for vname, cfg in ESTIMATOR_VARIANTS.items():
        pos_rows = []
        by_group: Dict[str, List[float]] = {}
        for case in positive_cases:
            res = estimate_vanishing_point(case["img"], cfg=cfg)
            gt_x, gt_y = case["vp_gt"]
            W, H = case["wh"]
            err_y_norm = abs(res["vp_norm"][1] - gt_y) if res["vp_norm"] is not None else None
            err_x_norm = abs(res["vp_norm"][0] - gt_x) if res["vp_norm"] is not None else None
            err_y_px = err_y_norm * H if err_y_norm is not None else None
            pos_rows.append({
                "group": case["group"],
                "label": case["label"],
                "wh": list(case["wh"]),
                "state": res["state"],
                "reasons": res["reasons"],
                "vp_gt": list(case["vp_gt"]),
                "vp_est": res["vp_norm"],
                "err_y_norm": round(err_y_norm, 5) if err_y_norm is not None else None,
                "err_x_norm": round(err_x_norm, 5) if err_x_norm is not None else None,
                "err_y_px": round(err_y_px, 2) if err_y_px is not None else None,
                "total_ms": res["timings_ms"]["total_ms"],
                "timings_ms": res["timings_ms"],
            })
            if err_y_norm is not None:
                by_group.setdefault(case["group"], []).append(err_y_norm)

        neg_rows = []
        false_valid = 0
        for nc in neg_controls:
            res = estimate_vanishing_point(nc["img"], cfg=cfg)
            if res["state"] == "VALID":
                false_valid += 1
            neg_rows.append({"label": nc["label"], "state": res["state"], "reasons": res["reasons"]})

        errs_valid = [r["err_y_norm"] for r in pos_rows if r["state"] == "VALID" and r["err_y_norm"] is not None]
        errs_all = [r["err_y_norm"] for r in pos_rows if r["err_y_norm"] is not None]
        valid_n = sum(1 for r in pos_rows if r["state"] == "VALID")
        scale_rows = [r for r in pos_rows if r["group"] == "resolution_scaling" and r["vp_est"] is not None]
        res_diff_norm = abs(scale_rows[0]["vp_est"][1] - scale_rows[1]["vp_est"][1]) if len(scale_rows) == 2 else None

        variant_results[vname] = {
            "positive_count": len(pos_rows),
            "valid_count": valid_n,
            "valid_rate": round(valid_n / len(pos_rows), 4),
            "err_y_norm_all": _stats(errs_all),
            "err_y_norm_valid_only": _stats(errs_valid),
            "resolution_360_vs_720_diff_norm": round(res_diff_norm, 5) if res_diff_norm is not None else None,
            "neg_false_valid_count": false_valid,
            "neg_control_total": len(neg_rows),
            "group_mean_err_y_norm": {g: round(float(np.mean(vs)), 5) for g, vs in by_group.items()},
            "cases": pos_rows,
            "neg_controls": neg_rows,
        }

    # Temporal synthetic verification: step change 0.34 -> 0.48 + 12-frame occlusion dropout + recovery
    seq_imgs = []
    gt_seq = []
    for f in range(55):
        if f < 18:
            vy = 0.34
            img = render_synthetic_perspective_road(640, 360, vp_norm=(0.50, vy), seed=200 + f)
        elif f < 30:
            vy = None  # complete blank occlusion for 12 frames (0.48s at 25 FPS)
            img = np.full((360, 640, 3), 90, dtype=np.uint8)
        else:
            vy = 0.48
            img = render_synthetic_perspective_road(640, 360, vp_norm=(0.50, vy), seed=200 + f)
        seq_imgs.append(img)
        gt_seq.append(vy)

    temp_results: Dict[str, Any] = {}
    est_lsd = ESTIMATOR_VARIANTS["lsd_320"]
    pre_seq = [estimate_vanishing_point(im, cfg=est_lsd) for im in seq_imgs]
    for pname in ("A_fixed", "B_dynamic_raw", "C_dynamic_smoothed", "D_dynamic_fallback"):
        trk = DynamicHorizonTracker(est_lsd, HorizonPolicyConfig(policy=pname, fixed_horizon_ratio=0.40, max_stale_sec=0.25, max_expired_sec=0.40))
        traj, states, errs = [], [], []
        for f, (im, est_r, gtv) in enumerate(zip(seq_imgs, pre_seq, gt_seq)):
            u = trk.update(im, f, f / 25.0, session_id="syn_seq", camera_profile="sedan_dashcam", precomputed_est=est_r)
            traj.append(u["active_horizon_ratio"])
            states.append(u["output_state"])
            if gtv is not None and u["active_horizon_ratio"] is not None:
                errs.append(abs(u["active_horizon_ratio"] - gtv))
        avail_rate = sum(1 for v in traj if v is not None) / len(traj)
        deltas = [abs(traj[i] - traj[i - 1]) for i in range(1, len(traj)) if traj[i] is not None and traj[i - 1] is not None]
        temp_results[pname] = {
            "availability_rate": round(avail_rate, 4),
            "gt_visible_err_norm": _stats(errs),
            "frame_to_frame_jitter": _stats(deltas),
            "state_counts": dict(Counter(states)),
        }

    return {"variants": variant_results, "temporal_synthetic_sequence": temp_results}


def offset_x(v: int) -> int:
    return v


def run_real_video_evaluation(sources_data: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Task 6 & 9: Exploratory evaluation across 5 local footage sources + timing breakdown
    (lsd_320 vs hough_320 vs hybrid_320 vs lsd_fullres, plus A/B/C/D policies on lsd_320).
    """
    all_csv_rows: List[Dict[str, Any]] = []
    detector_summary: Dict[str, Any] = {}
    timing_breakdown: Dict[str, Any] = {}

    # Cache lsd_320 raw estimates per source for deterministic policy replay
    lsd320_cache: Dict[str, List[Dict[str, Any]]] = {}

    for vname, cfg in ESTIMATOR_VARIANTS.items():
        per_src_det: Dict[str, Any] = {}
        stage_acc: Dict[str, List[float]] = {k: [] for k in ("preprocess_ms", "edge_ms", "line_detect_ms", "ransac_ms", "qa_ms", "total_ms")}
        stage_by_res: Dict[str, Dict[str, List[float]]] = {
            "360p": {k: [] for k in stage_acc},
            "720p": {k: [] for k in stage_acc},
        }

        for sid, (frames, meta) in sources_data.items():
            # Skip fullres on short clip? No, it's fast enough (~657 frames total is just a few seconds)
            ests = []
            st_cnt = Counter()
            rej_reasons = Counter()
            raw_ys, acc_lines, raw_lines = [], [], []
            for fm in frames:
                e = estimate_vanishing_point(fm["img_bgr"], cfg=cfg)
                ests.append(e)
                st_cnt[e["state"]] += 1
                for r in e["reasons"]:
                    rej_reasons[r] += 1
                if e["horizon_y_ratio"] is not None:
                    raw_ys.append(e["horizon_y_ratio"])
                raw_lines.append(e["diagnostics"]["raw_line_count"])
                acc_lines.append(e["diagnostics"]["accepted_line_count"])
                res_key = "720p" if meta["wh"][1] >= 700 else "360p"
                for k in stage_acc:
                    v_ms = float(e["timings_ms"].get(k, 0.0))
                    stage_acc[k].append(v_ms)
                    stage_by_res[res_key][k].append(v_ms)

            if vname == "lsd_320":
                lsd320_cache[sid] = ests

            n_f = max(1, len(frames))
            diffs = [abs(raw_ys[i] - raw_ys[i - 1]) for i in range(1, len(raw_ys))]
            per_src_det[sid] = {
                "n_frames": len(frames),
                "state_counts": dict(st_cnt),
                "valid_rate": round(st_cnt["VALID"] / n_f, 4),
                "low_conf_rate": round(st_cnt["LOW_CONFIDENCE"] / n_f, 4),
                "unavailable_rate": round(st_cnt["UNAVAILABLE"] / n_f, 4),
                "horizon_y_ratio_stats": _stats(raw_ys),
                "raw_step_jitter": _stats(diffs),
                "mean_raw_lines": round(float(np.mean(raw_lines)), 2) if raw_lines else 0.0,
                "mean_accepted_lines": round(float(np.mean(acc_lines)), 2) if acc_lines else 0.0,
                "top_reasons": dict(rej_reasons.most_common(6)),
                "latency_total_ms": _stats([e["timings_ms"]["total_ms"] for e in ests]),
            }
        detector_summary[vname] = per_src_det
        timing_breakdown[vname] = {
            "overall": {k: _stats(v) for k, v in stage_acc.items()},
            "by_resolution": {rk: {k: _stats(v) for k, v in rd.items()} for rk, rd in stage_by_res.items()},
        }

    # Evaluate Policies A/B/C/D on lsd_320 across all 5 sources
    policy_summary: Dict[str, Any] = {}
    trajectories_for_plot: Dict[str, Dict[str, Any]] = {}
    est_cfg = ESTIMATOR_VARIANTS["lsd_320"]

    for pname in ("A_fixed", "B_dynamic_raw", "C_dynamic_smoothed", "D_dynamic_fallback"):
        policy_summary[pname] = {}
        for sid, (frames, meta) in sources_data.items():
            trk = DynamicHorizonTracker(est_cfg, HorizonPolicyConfig(policy=pname, fixed_horizon_ratio=0.40))
            ests = lsd320_cache[sid]
            st_cnt = Counter()
            fb_cnt = 0
            act_ys: List[Optional[float]] = []
            filt_ms: List[float] = []
            ts_list: List[float] = []
            out_states: List[str] = []

            for fm, est_r in zip(frames, ests):
                u = trk.update(
                    fm["img_bgr"], fm["frame_idx"], fm["ts_sec"],
                    session_id=sid, camera_profile="sedan_dashcam",
                    precomputed_est=est_r,
                )
                st_cnt[u["output_state"]] += 1
                if u["fallback_used"]:
                    fb_cnt += 1
                act_ys.append(u["active_horizon_ratio"])
                out_states.append(u["output_state"])
                filt_ms.append(u["filter_ms"])
                ts_list.append(fm["ts_sec"])

                all_csv_rows.append({
                    "source_id": sid,
                    "frame_idx": fm["frame_idx"],
                    "ts_sec": fm["ts_sec"],
                    "policy": pname,
                    "raw_state": est_r["state"],
                    "output_state": u["output_state"],
                    "active_horizon_ratio": u["active_horizon_ratio"] if u["active_horizon_ratio"] is not None else "",
                    "raw_horizon_ratio": u["raw_horizon_ratio"] if u["raw_horizon_ratio"] is not None else "",
                    "smoothed_horizon_ratio": u["smoothed_horizon_ratio"] if u["smoothed_horizon_ratio"] is not None else "",
                    "fallback_used": int(u["fallback_used"]),
                    "fallback_reason": u["fallback_reason"] or "",
                    "raw_lines": est_r["diagnostics"]["raw_line_count"],
                    "accepted_lines": est_r["diagnostics"]["accepted_line_count"],
                    "inlier_count": est_r["diagnostics"]["inlier_count"],
                    "left_inliers": est_r["diagnostics"]["left_inliers"],
                    "right_inliers": est_r["diagnostics"]["right_inliers"],
                    "cond_number": est_r["diagnostics"]["cond_number"],
                    "vp_dispersion": est_r["diagnostics"]["vp_dispersion_norm"] or "",
                    "heuristic_confidence": est_r["heuristic_confidence"],
                    "total_ms": round(est_r["timings_ms"]["total_ms"] + u["filter_ms"], 4),
                })

            valid_vals = [y for y in act_ys if y is not None]
            deltas = [abs(act_ys[i] - act_ys[i - 1]) for i in range(1, len(act_ys)) if act_ys[i] is not None and act_ys[i - 1] is not None]
            n_f = max(1, len(frames))
            policy_summary[pname][sid] = {
                "n_frames": len(frames),
                "output_availability_rate": round(len(valid_vals) / n_f, 4),
                "fallback_rate": round(fb_cnt / n_f, 4),
                "output_state_counts": dict(st_cnt),
                "active_horizon_stats": _stats(valid_vals),
                "frame_to_frame_jitter": _stats(deltas),
                "max_step_delta": round(float(max(deltas)), 5) if deltas else 0.0,
                "filter_ms": _stats(filt_ms),
            }
            if sid in ("ruralRoad_202f", "rangoonRoad_f200_399", "rawSurvey_160f_10fps", "phonePotholes_70f"):
                trajectories_for_plot.setdefault(sid, {"ts_sec": ts_list})[pname] = {
                    "active_y": act_ys,
                    "states": out_states,
                    "raw_y": [e["horizon_y_ratio"] for e in ests],
                    "raw_state": [e["state"] for e in ests],
                }

    # Selective horizon refresh cadence timing check (lsd_320)
    lsd_mean_ms = timing_breakdown["lsd_320"]["overall"]["total_ms"]["mean"]
    refresh_cadences = {f"every_k{k}": round(lsd_mean_ms / k, 4) for k in (1, 2, 3, 5, 10)}

    return {
        "note": "EXPLORATORY ONLY: Real footage lacks physical horizon labels, pitch extrinsics, or metric calibration.",
        "detector_summary_by_source": detector_summary,
        "policy_summary_lsd320": policy_summary,
        "timing_breakdown_ms": timing_breakdown,
        "selective_refresh_amortized_ms": refresh_cadences,
        "csv_rows": all_csv_rows,
        "trajectories_for_plot": trajectories_for_plot,
    }


def _calibrator_formulas(
    H_cam: float,
    vfov_deg: float,
    horizon_ratio: float,
    frame_h: int,
    y2_px: float,
    raw_disp: float = 1.0,
    k_disp: float = 0.082,
    depth_exponent: float = 1.35,
) -> Dict[str, float]:
    """Exact mirror of roadvision_engine.MetricDepthCalibrator equations for sensitivity analysis."""
    horizon_y = int(frame_h * horizon_ratio)
    fy = (frame_h / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
    dy_raw = float(y2_px - horizon_y)
    dy = max(15.0, dy_raw)
    Z = (H_cam * fy) / dy
    depth_cm = raw_disp * (Z ** depth_exponent) * k_disp
    return {"horizon_y": horizon_y, "fy": fy, "dy_raw": dy_raw, "dy_clamped": dy, "Z_road_m": Z, "depth_mult_cm_per_unit": depth_cm}


def run_geometry_sensitivity_and_track_test(
    sources_data: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    trajectories: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Tasks 7 & 8:
      1) Analytic & numerical sensitivity of Z_road_m and depth_cm to horizon_ratio, y2, H_cam, VFOV, and 360p vs 720p 15px clamp.
      2) Matched-track approach-drift test on ruralRoad_202f, rawSurvey_160f_10fps, and rangoonRoad_f200_399 (first 120f)
         using DAv2 336 + elliptical_core comparing fixed horizon (0.40) vs D_dynamic_fallback horizon.
    """
    # 1. Horizon ratio & y2 ratio grid (720p sedan baseline)
    horizon_sweep = []
    for y2_r in (0.46, 0.55, 0.70, 0.85):
        pts = []
        base = _calibrator_formulas(1.25, 60.0, 0.40, 720, y2_r * 720.0)
        for hr in (0.30, 0.35, 0.40, 0.44, 0.48):
            c = _calibrator_formulas(1.25, 60.0, hr, 720, y2_r * 720.0)
            pts.append({
                "horizon_ratio": hr,
                "Z_road_m": round(c["Z_road_m"], 3),
                "depth_mult_cm_per_unit": round(c["depth_mult_cm_per_unit"], 3),
                "rel_depth_vs_h040": round(c["depth_mult_cm_per_unit"] / max(1e-9, base["depth_mult_cm_per_unit"]), 4),
            })
        horizon_sweep.append({"y2_ratio": y2_r, "sweep": pts})

    # 2. Resolution asymmetry due to hardcoded `max(15.0, y2 - horizon_y)` pixel clamp
    clamp_asymmetry = []
    for y2_r in (0.42, 0.43, 0.45, 0.50, 0.65, 0.85):
        c360 = _calibrator_formulas(1.25, 60.0, 0.40, 360, y2_r * 360.0)
        c720 = _calibrator_formulas(1.25, 60.0, 0.40, 720, y2_r * 720.0)
        clamp_asymmetry.append({
            "y2_ratio": y2_r,
            "Z_360p_m": round(c360["Z_road_m"], 3),
            "Z_720p_m": round(c720["Z_road_m"], 3),
            "depth_mult_360p": round(c360["depth_mult_cm_per_unit"], 3),
            "depth_mult_720p": round(c720["depth_mult_cm_per_unit"], 3),
            "ratio_720_over_360_depth": round(c720["depth_mult_cm_per_unit"] / c360["depth_mult_cm_per_unit"], 4),
        })

    # 3. Camera profile variance (sedan vs suv vs truck vs two-wheeler) at y2_ratio=0.65 (720p)
    profiles_cfg = {
        "sedan_dashcam": (1.25, 60.0, 0.40),
        "suv_crossover": (1.55, 65.0, 0.42),
        "commercial_truck_bus": (2.20, 70.0, 0.46),
        "light_mobility_twowheeler": (0.95, 55.0, 0.38),
    }
    profile_sens = {}
    for pn, (hc, vf, hr) in profiles_cfg.items():
        c = _calibrator_formulas(hc, vf, hr, 720, 0.65 * 720.0)
        profile_sens[pn] = {
            "H_cam_m": hc, "vfov_deg": vf, "horizon_ratio": hr,
            "Z_at_y65_m": round(c["Z_road_m"], 3),
            "depth_mult_at_y65": round(c["depth_mult_cm_per_unit"], 3),
        }

    # 4. Matched-track DAv2 336 approach-drift test (Task 7 empirical test on real footage)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(DEFAULT_DET_WEIGHTS)
    runner = FastDAv2Runner(336, device=device)
    runner.warmup(360, 640)
    opt_cfg = OptimizedDepthConfig(**FROZEN_PHASE7A_CONFIG["ransac"])
    core_cfg = CoreExtractionConfig()

    target_sids = ["ruralRoad_202f", "rawSurvey_160f_10fps", "rangoonRoad_f200_399"]
    track_obs: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for sid in target_sids:
        frames, meta = sources_data[sid]
        H_fr = meta["wh"][1]
        dyn_ys = trajectories[sid]["D_dynamic_fallback"]["active_y"]
        det.predictor = None
        for idx_f, (fm, dyn_r) in enumerate(zip(frames, dyn_ys)):
            res = det.track(fm["img_bgr"], persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.50, imgsz=640, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            dmap = runner.predict(fm["img_bgr"]).depth_map
            boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy().tolist()]
            tids = [int(t) for t in res.boxes.id.int().cpu().tolist()] if res.boxes.id is not None else [i + 1 for i in range(len(boxes))]
            for b_i, (bx, tid) in enumerate(zip(boxes, tids)):
                nbrs = [boxes[j] for j in range(len(boxes)) if j != b_i]
                reg = build_and_sample_annular_region(dmap, bx, neighbor_boxes=nbrs, cfg=opt_cfg)
                fit = fit_ransac_optimized(dmap, reg, cfg=opt_cfg)
                ext = extract_cavity_with_strategy(dmap, bx, strategy="elliptical_core", neighbor_boxes=nbrs, img_bgr=fm["img_bgr"], opt_cfg=opt_cfg, core_cfg=core_cfg, precomputed_region=reg, precomputed_fit=fit)
                if not ext["is_valid"]:
                    continue
                score = ext["core_cavity_stats"]["scale_normalized_p75"]
                if score is None:
                    continue
                y2_px = bx[3]
                c_fix = _calibrator_formulas(1.25, 60.0, 0.40, H_fr, y2_px, raw_disp=score)
                c_dyn = _calibrator_formulas(1.25, 60.0, float(dyn_r), H_fr, y2_px, raw_disp=score)
                track_obs.setdefault((sid, tid), []).append({
                    "frame_idx": fm["frame_idx"],
                    "y2_norm": y2_px / float(H_fr),
                    "score": float(score),
                    "depth_fixed_cm": c_fix["depth_mult_cm_per_unit"],
                    "depth_dyn_cm": c_dyn["depth_mult_cm_per_unit"],
                    "Z_fixed_m": c_fix["Z_road_m"],
                    "Z_dyn_m": c_dyn["Z_road_m"],
                })

    # Analyze multi-frame tracks (>= 5 valid detections)
    cvs_raw, cvs_fix, cvs_dyn = [], [], []
    abs_rhos_fix, abs_rhos_dyn, signed_rhos_fix, signed_rhos_dyn = [], [], [], []
    approaching_tracks = 0
    for (sid, tid), rows in track_obs.items():
        if len(rows) < 5:
            continue
        sc = np.array([r["score"] for r in rows], dtype=np.float64)
        dfx = np.array([r["depth_fixed_cm"] for r in rows], dtype=np.float64)
        ddy = np.array([r["depth_dyn_cm"] for r in rows], dtype=np.float64)
        y2s = np.array([r["y2_norm"] for r in rows], dtype=np.float64)
        ts = np.arange(len(rows), dtype=np.float64)
        cvs_raw.append(float(np.std(sc) / max(1e-6, np.mean(sc))))
        cvs_fix.append(float(np.std(dfx) / max(1e-6, np.mean(dfx))))
        cvs_dyn.append(float(np.std(ddy) / max(1e-6, np.mean(ddy))))
        if (y2s[-1] - y2s[0]) > 0.03:
            approaching_tracks += 1
            rf = _spearman_rho(ts, dfx)
            rd = _spearman_rho(ts, ddy)
            if rf is not None and rd is not None:
                abs_rhos_fix.append(abs(rf))
                abs_rhos_dyn.append(abs(rd))
                signed_rhos_fix.append(rf)
                signed_rhos_dyn.append(rd)

    del det, runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "horizon_sweep_720p": horizon_sweep,
        "clamp_asymmetry_360_vs_720": clamp_asymmetry,
        "profile_sensitivity_720p_y65": profile_sens,
        "matched_track_approach_drift_test": {
            "qualified_tracks_ge5": len(cvs_fix),
            "approaching_tracks": approaching_tracks,
            "within_track_cv_raw_dav2_score": _stats(cvs_raw),
            "within_track_cv_depth_fixed_h040": _stats(cvs_fix),
            "within_track_cv_depth_dynamic_D": _stats(cvs_dyn),
            "approaching_abs_spearman_rho_fixed": _stats(abs_rhos_fix),
            "approaching_abs_spearman_rho_dynamic": _stats(abs_rhos_dyn),
            "approaching_signed_spearman_rho_fixed": _stats(signed_rhos_fix),
            "approaching_signed_spearman_rho_dynamic": _stats(signed_rhos_dyn),
            "note": "Testing purely whether substituting dynamic VP horizon D for fixed 0.40 reduces within-track variation or approach drift without changing k_disp/depth_exponent.",
        },
    }


def generate_phase7b_plots(syn: Dict[str, Any], real: Dict[str, Any], geom: Dict[str, Any]) -> List[str]:
    os.makedirs(VIZ_DIR, exist_ok=True)
    saved: List[str] = []

    # 1. Synthetic validation plot
    groups = ["multi_pitch", "angled_road", "resolution_scaling", "missing_markings",
              "partial_occlusion", "spurious_clutter", "shadows", "heavy_noise",
              "camera_roll", "incompatible_dual_vp"]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(len(groups))
    w = 0.20
    for idx_v, (vname, col) in enumerate([("lsd_320", "#1f77b4"), ("hough_320", "#ff7f0e"), ("hybrid_320", "#2ca02c"), ("lsd_fullres", "#9467bd")]):
        gmap = syn["variants"][vname]["group_mean_err_y_norm"]
        vals = [gmap.get(g, np.nan) for g in groups]
        ax.bar(x + (idx_v - 1.5) * w, vals, width=w, label=vname, color=col, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=8.5)
    ax.set_ylabel("Mean |horizon_y_est - GT_y| (normalized)")
    ax.set_title("Phase 7B Task 5: Synthetic Projective Horizon Error Across 10 Controlled Scenarios")
    ax.axhline(0.03, color="red", linestyle="--", linewidth=1.0, label="3% height tolerance guide")
    ax.legend(ncol=3, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    p1 = os.path.join(VIZ_DIR, "synthetic_horizon_validation.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # 2. Real-video horizon trajectories (4 sources)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
    sids = ["ruralRoad_202f", "rangoonRoad_f200_399", "rawSurvey_160f_10fps", "phonePotholes_70f"]
    for ax, sid in zip(axes.ravel(), sids):
        tr = real["trajectories_for_plot"][sid]
        ts = tr["ts_sec"]
        ax.axhline(0.40, color="#666666", linestyle="--", linewidth=1.2, label="A_fixed (0.40)")
        raw_y = [v if v is not None else np.nan for v in tr["B_dynamic_raw"]["active_y"]]
        sm_y = [v if v is not None else np.nan for v in tr["C_dynamic_smoothed"]["active_y"]]
        fb_y = [v if v is not None else np.nan for v in tr["D_dynamic_fallback"]["active_y"]]
        ax.scatter(ts, raw_y, s=9, color="#ff7f0e", alpha=0.45, label="B_dynamic_raw")
        ax.plot(ts, sm_y, color="#1f77b4", linewidth=1.5, label="C_dynamic_smoothed")
        ax.plot(ts, fb_y, color="#2ca02c", linewidth=1.3, linestyle=":", label="D_dynamic_fallback")
        ax.set_ylim(0.15, 0.70)
        ax.invert_yaxis()
        ax.set_title(f"{sid} (Exploratory)", fontsize=10)
        ax.set_xlabel("Timestamp (s)")
        ax.set_ylabel("Horizon y/H (top=0)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Phase 7B Task 6: Real Footage Exploratory Horizon Trajectories (No GT Labels)", fontsize=12)
    p2 = os.path.join(VIZ_DIR, "real_video_horizon_trajectories.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    # 3. Geometry sensitivity & 360p vs 720p asymmetry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for item in geom["horizon_sweep_720p"]:
        hrs = [p["horizon_ratio"] for p in item["sweep"]]
        rels = [p["rel_depth_vs_h040"] for p in item["sweep"]]
        ax1.plot(hrs, rels, marker="o", label=f"y2_ratio={item['y2_ratio']:.2f}")
    ax1.axvline(0.40, color="gray", linestyle="--")
    ax1.axhline(1.00, color="gray", linestyle=":")
    ax1.set_title("Sensitivity of depth_cm to horizon_ratio (720p Sedan)")
    ax1.set_xlabel("Assumed horizon_ratio")
    ax1.set_ylabel("Relative depth_cm multiplier (vs h=0.40)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ca = geom["clamp_asymmetry_360_vs_720"]
    y2rs = [c["y2_ratio"] for c in ca]
    r_360_720 = [c["ratio_720_over_360_depth"] for c in ca]
    ax2.plot(y2rs, r_360_720, marker="s", color="#d62728", linewidth=2.0)
    ax2.axhline(1.0, color="gray", linestyle="--")
    ax2.set_title("Scale Asymmetry from Hardcoded max(15px, y2 - h_y)")
    ax2.set_xlabel("Normalized bottom coordinate y2/H")
    ax2.set_ylabel("depth_cm(720p) / depth_cm(360p) at identical ratio")
    ax2.grid(alpha=0.3)
    p3 = os.path.join(VIZ_DIR, "geometry_sensitivity_analysis.png")
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved.append(p3)

    # 4. Latency stage breakdown (360p vs 720p)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    labels, pre_v, edge_v, line_v, ran_v, qa_v, p95_v = [], [], [], [], [], [], []
    for vname in ("lsd_320", "hough_320", "hybrid_320", "lsd_fullres"):
        for rkey in ("360p", "720p"):
            d = real["timing_breakdown_ms"][vname]["by_resolution"][rkey]
            labels.append(f"{vname}\n({rkey})")
            pre_v.append(d["preprocess_ms"]["mean"] or 0.0)
            edge_v.append(d["edge_ms"]["mean"] or 0.0)
            line_v.append(d["line_detect_ms"]["mean"] or 0.0)
            ran_v.append(d["ransac_ms"]["mean"] or 0.0)
            qa_v.append(d["qa_ms"]["mean"] or 0.0)
            p95_v.append(d["total_ms"]["p95"] or 0.0)
    xi = np.arange(len(labels))
    b1 = np.array(pre_v)
    b2 = b1 + np.array(edge_v)
    b3 = b2 + np.array(line_v)
    b4 = b3 + np.array(ran_v)
    ax.bar(xi, pre_v, label="preprocess")
    ax.bar(xi, edge_v, bottom=b1, label="edge")
    ax.bar(xi, line_v, bottom=b2, label="line_detect")
    ax.bar(xi, ran_v, bottom=b3, label="ransac")
    ax.bar(xi, qa_v, bottom=b4, label="qa")
    ax.scatter(xi, p95_v, color="black", marker="D", s=28, zorder=5, label="total P95 ms")
    ax.set_xticks(xi)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Latency (ms / frame)")
    ax.set_title("Phase 7B Task 9: Classical Horizon Estimator Stage Latency Breakdown")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    p4 = os.path.join(VIZ_DIR, "latency_stage_breakdown.png")
    fig.tight_layout()
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved.append(p4)

    return saved


def main() -> None:
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    print("[Phase 7B] Loading 5 real sources...")
    sources_data = {}
    sources_meta = {}
    for src in REAL_SOURCES:
        frames, meta = _load_source_frames(src)
        sources_data[src["source_id"]] = (frames, meta)
        sources_meta[src["source_id"]] = meta
        print(f"  - {src['source_id']}: {meta['n_frames']} frames @ {meta['fps']} FPS ({meta['wh'][0]}x{meta['wh'][1]})")

    print("[Phase 7B] Running Task 5 synthetic projective suite...")
    syn_res = run_synthetic_suite()

    print("[Phase 7B] Running Tasks 6 & 9 real-video horizon & latency evaluation...")
    real_res = run_real_video_evaluation(sources_data)

    print("[Phase 7B] Running Tasks 7 & 8 sensitivity + DAv2 matched-track drift test...")
    geom_res = run_geometry_sensitivity_and_track_test(sources_data, real_res["trajectories_for_plot"])

    print("[Phase 7B] Generating Phase 7B diagnostic plots...")
    plot_paths = generate_phase7b_plots(syn_res, real_res, geom_res)

    csv_path = os.path.join(BENCH_DIR, "phase7b_per_frame_horizon.csv")
    csv_rows = real_res.pop("csv_rows")
    real_res.pop("trajectories_for_plot", None)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    out_json = {
        "phase": "7B",
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "sources_meta": sources_meta,
        "synthetic_validation": syn_res,
        "real_video_exploratory": real_res,
        "geometry_sensitivity": geom_res,
        "generated_plots": [os.path.relpath(p, BASE_DIR).replace("\\", "/") for p in plot_paths],
        "per_frame_csv": os.path.relpath(csv_path, BASE_DIR).replace("\\", "/"),
    }
    json_path = os.path.join(BENCH_DIR, "phase7b_horizon_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"[Phase 7B] Complete!\n  JSON: {json_path}\n  CSV:  {csv_path}\n  Plots: {len(plot_paths)} files in {VIZ_DIR}")


if __name__ == "__main__":
    main()
