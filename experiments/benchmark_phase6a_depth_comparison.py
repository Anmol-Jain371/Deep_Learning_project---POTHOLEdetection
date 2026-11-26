"""
========================================================================================
RoadVision AI -- Phase 6A: Core Extraction & MiDaS vs Depth Anything V2 Benchmark
========================================================================================
Covers:
  1. Task 4: Synthetic regression across all 7 cavity strategies over 12 controlled scenarios.
  2. Task 3 & 7 (Debug Set): 36 Phase 5A development frames (98 regions) comparing
     all cavity strategies + 2x2 factorial (`MiDaS_small` vs `DAv2 Small`) x (`full_box` vs
     `elliptical_core`/`rect_trim_20`) + `±8%` box perturbation sensitivity.
  3. Task 7 (Exploratory Grouped Video Set): 360 contiguous video frames across 3 capture
     sources (`ruralRoad` 120f unseen, `Rangoon` f200..319 120f, `rawSurvey` 720p 120f)
     with identical YOLOv8n + ByteTrack boxes, track IDs, and timestamps.
  4. Task 8: Within-track vs pooled approach-distance correlation & affine invariance
     across `raw_p75`, `local_scale_norm`, `frame_iqr_norm`, `area_weighted_iqr`, `y2_weighted_iqr`.
  5. Task 9: Synchronized RTX 2050 latency, VRAM, and `0..4+` candidate scaling benchmark
     (`MiDaS 256`, `DAv2 518 native`, `DAv2 266` accuracy-independent runtime ablation).
  6. Task 10 & 11: Multi-panel visual diagnostics, performance plots, JSON, and CSVs.

Explicitly exploratory — zero production file edits, zero training, relative units ONLY.
========================================================================================
"""

import os
import sys
import csv
import json
import time
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

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
from experiments.optimized_relative_depth import (
    OptimizedDepthConfig,
    build_and_sample_annular_region,
    fit_ransac_optimized,
)
from experiments.core_aware_cavity_extraction import (
    CAVITY_STRATEGIES,
    CoreExtractionConfig,
    extract_cavity_with_strategy,
)
from experiments.depth_model_comparison import (
    MiDaSSmallAdapter,
    DepthAnythingV2SmallAdapter,
)
from experiments.benchmark_optimized_relative_depth import (
    _spearman_rho,
    _guarded_rel_diff_pct,
    _summarize_vals,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "phase6a_viz")

EXPLORATORY_VIDEO_SOURCES = [
    {
        "source_id": "ruralRoad_f0_119",
        "path": "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
        "start_frame": 0,
        "count": 120,
        "prior_exposure": "UNSEEN in detector/seg train (reused dev video from Phase 4A.4-5C)",
    },
    {
        "source_id": "rangoonRoad_f200_319",
        "path": "Watch Video_ Terrible potholes in Rangoon Road.mp4",
        "start_frame": 200,
        "count": 120,
        "prior_exposure": "Contiguous span f200..319 disjoint from Phase 5C f0..179 (4 isolated frames 211,255,284,313 in v2_candidates)",
    },
    {
        "source_id": "rawSurvey_720p_f0_119",
        "path": "roadvision-dashboard/public/videos/raw_survey.mp4",
        "start_frame": 0,
        "count": 120,
        "prior_exposure": "0 MD5 matches in detector train (dev 720p urban video from Phase 5C)",
    },
]


def _make_plane(h=240, w=320, a=0.05, b=-0.02, c=15.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return a * yy + b * xx + c


def _make_ellipse(h=240, w=320, cy=150.0, cx=160.0, ry=24.0, rx=28.0, depth=3.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return np.where(r2 <= 1.0, depth * np.maximum(0.0, 1.0 - r2) ** 0.75, 0.0)


def run_synthetic_core_suite() -> Dict[str, Any]:
    """Task 4: Evaluate all 7 cavity strategies across 12 synthetic relative-depth scenarios."""
    h, w = 240, 320
    std_box = (115, 105, 205, 195)
    p = _make_plane(h, w)
    rng = np.random.default_rng(123)

    scenarios = [
        ("centered_depression", p, p - _make_ellipse(h, w, 150, 160, 26, 30, 3.5), std_box, True),
        ("off_center_depression", p, p - _make_ellipse(h, w, 128, 144, 20, 22, 3.5), std_box, True),
        ("irregular_cavity", p, p - np.maximum(_make_ellipse(h, w, 142, 148, 18, 22, 3.5), _make_ellipse(h, w, 162, 174, 16, 20, 3.0)), std_box, True),
        ("multiple_depressions", p, p - _make_ellipse(h, w, 132, 138, 19, 19, 3.2) - _make_ellipse(h, w, 168, 180, 19, 19, 3.0), std_box, True),
        ("flat_road", p, p.copy(), std_box, False),
        ("raised_road_feature", p, p + _make_ellipse(h, w, 150, 160, 26, 30, 3.5), std_box, False),
        ("shadow_edge_step", p, p - _make_ellipse(h, w, 150, 160, 25, 28, 3.5) + np.where(np.arange(w)[None, :] < 128, 0.6, 0.0), std_box, True),
        ("ref_bias_curb", p, (p - _make_ellipse(h, w, 150, 160, 26, 30, 3.5)) + np.where((np.arange(h)[:, None] >= 65) & (np.arange(h)[:, None] <= 98), 10.0, 0.0), std_box, True),
        ("contaminated_bbox_edges", p, (p - _make_ellipse(h, w, 150, 160, 20, 24, 3.5)) + np.where((np.abs(np.arange(h)[:, None] - 150) >= 28) | (np.abs(np.arange(w)[None, :] - 160) >= 28), 2.8, 0.0) * ((np.arange(h)[:, None] >= 105) & (np.arange(h)[:, None] <= 195) & (np.arange(w)[None, :] >= 115) & (np.arange(w)[None, :] <= 205)), std_box, True),
        ("severe_noise", p, p - _make_ellipse(h, w, 150, 160, 26, 30, 3.5) + rng.normal(0.0, 0.45, size=(h, w)), std_box, True),
        ("weak_depression", p, p - _make_ellipse(h, w, 150, 160, 26, 30, 0.6) + rng.normal(0.0, 0.02, size=(h, w)), std_box, True),
        ("partial_clipped_cavity", p, p - _make_ellipse(h, w, 218, 284, 20, 24, 3.5), (254, 194, 319, 239), True),
    ]

    opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    core_cfg = CoreExtractionConfig()
    strat_summary = {s: {"match": 0, "false_accept": 0, "p75_errs": [], "covs": [], "pos_valid": 0} for s in CAVITY_STRATEGIES}
    per_scenario = {}

    for sc_name, t_plane, dmap, bx, exp_v in scenarios:
        x1, y1, x2, y2 = bx
        t_res = t_plane[y1:y2, x1:x2] - dmap[y1:y2, x1:x2]
        true_full_p75 = float(np.percentile(t_res[np.isfinite(t_res)], 75.0))
        reg = build_and_sample_annular_region(dmap, bx, cfg=opt_cfg)
        fit = fit_ransac_optimized(dmap, reg, cfg=opt_cfg)

        sc_dict = {"expected_valid": exp_v, "true_full_box_p75": round(true_full_p75, 4), "strategies": {}}
        for st in CAVITY_STRATEGIES:
            out = extract_cavity_with_strategy(
                dmap, bx, strategy=st, opt_cfg=opt_cfg, core_cfg=core_cfg,
                precomputed_region=reg, precomputed_fit=fit, return_arrays=True
            )
            cmask = out["arrays"]["core_mask"]
            # Compare against TRUE underlying cavity residual inside THAT strategy's selected interior!
            true_core_vals = t_res[cmask & np.isfinite(t_res)]
            true_core_p75 = float(np.percentile(true_core_vals, 75.0)) if len(true_core_vals) > 0 else 0.0
            est_p75 = out["core_cavity_stats"]["raw_diag_p75"]
            err = abs(est_p75 - true_core_p75)

            matched = (out["is_valid"] == exp_v)
            strat_summary[st]["match"] += int(matched)
            if not exp_v and out["is_valid"]:
                strat_summary[st]["false_accept"] += 1
            if exp_v:
                strat_summary[st]["pos_valid"] += int(out["is_valid"])
                strat_summary[st]["p75_errs"].append(err)
            strat_summary[st]["covs"].append(out["interior_coverage_ratio"])
            sc_dict["strategies"][st] = {
                "is_valid": out["is_valid"],
                "reasons": out["failure_reasons"],
                "coverage_ratio": out["interior_coverage_ratio"],
                "est_p75": est_p75,
                "true_selected_p75": round(true_core_p75, 4),
                "abs_err": round(err, 4),
            }
        per_scenario[sc_name] = sc_dict

    summary = {}
    for st, d in strat_summary.items():
        summary[st] = {
            "validity_accuracy": f"{d['match']}/{len(scenarios)}",
            "positive_scenarios_valid": f"{d['pos_valid']}/10",
            "flat_or_raised_false_accepts": f"{d['false_accept']}/2",
            "mean_p75_abs_err_vs_selected_true": round(float(np.mean(d["p75_errs"])), 4),
            "max_p75_abs_err_vs_selected_true": round(float(np.max(d["p75_errs"])), 4),
            "mean_interior_coverage_ratio": round(float(np.mean(d["covs"])), 4),
            "selection_biased": (st == "spatially_coherent_residual"),
        }
    return {"summary": summary, "per_scenario": per_scenario}


def run_dev_36_comparison(
    det_model: YOLO,
    midas: MiDaSSmallAdapter,
    dav2: DepthAnythingV2SmallAdapter,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Task 3 & 7: 36 dev frames comparing cavity strategies & 2x2 Model x Cavity + ±8% perturbation."""
    catalog = build_dev_frame_catalog()
    opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    core_cfg = CoreExtractionConfig()

    cached = []
    for entry in catalog:
        img_bgr, ts = load_frame(entry)
        r = det_model.predict(img_bgr, imgsz=640, conf=0.25, iou=0.70, verbose=False)[0]
        boxes = [tuple(float(v) for v in b) for b in r.boxes.xyxy.cpu().numpy().tolist()] if (r.boxes is not None and len(r.boxes) > 0) else []
        if not boxes:
            continue
        pm = midas.predict(img_bgr)
        pd = dav2.predict(img_bgr)
        cached.append({
            "entry": entry, "ts": ts, "img_bgr": img_bgr, "boxes": boxes,
            "confs": [float(c) for c in r.boxes.conf.cpu().numpy().tolist()],
            "d_midas": pm.depth_map, "d_dav2": pd.depth_map,
        })

    # Part A: All 7 cavity strategies on MiDaS_small over the 98 dev regions
    cav_ablation = {s: {"valid": 0, "reasons": Counter(), "sn_p75": [], "cov": [], "ms": []} for s in CAVITY_STRATEGIES}
    for cf in cached:
        for i, bx in enumerate(cf["boxes"]):
            nbrs = [cf["boxes"][j] for j in range(len(cf["boxes"])) if j != i]
            reg = build_and_sample_annular_region(cf["d_midas"], bx, neighbor_boxes=nbrs, cfg=opt_cfg)
            fit = fit_ransac_optimized(cf["d_midas"], reg, cfg=opt_cfg)
            for st in CAVITY_STRATEGIES:
                o = extract_cavity_with_strategy(
                    cf["d_midas"], bx, strategy=st, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"],
                    opt_cfg=opt_cfg, core_cfg=core_cfg, precomputed_region=reg, precomputed_fit=fit
                )
                cav_ablation[st]["cov"].append(o["interior_coverage_ratio"])
                cav_ablation[st]["ms"].append(o["total_ms"])
                if o["is_valid"]:
                    cav_ablation[st]["valid"] += 1
                    cav_ablation[st]["sn_p75"].append(o["core_cavity_stats"]["scale_normalized_p75"])
                else:
                    for rs in o["failure_reasons"]:
                        cav_ablation[st]["reasons"][rs] += 1

    cav_summary = {
        s: {
            "valid_count": d["valid"],
            "valid_pct": round(100.0 * d["valid"] / max(1, len(d["cov"])), 2),
            "mean_coverage_ratio": round(float(np.mean(d["cov"])), 4),
            "median_scale_norm_p75": round(float(np.median(d["sn_p75"])), 4) if d["sn_p75"] else None,
            "mean_ms": round(float(np.mean(d["ms"])), 4),
            "failure_reasons": dict(d["reasons"]),
            "selection_biased": (s == "spatially_coherent_residual"),
        }
        for s, d in cav_ablation.items()
    }

    # Part B: 4 Controlled arms + ±8% box perturbation on cached dev frames
    arms = {
        "midas_full": ("d_midas", "full_box"),
        "midas_elliptical": ("d_midas", "elliptical_core"),
        "dav2_full": ("d_dav2", "full_box"),
        "dav2_elliptical": ("d_dav2", "elliptical_core"),
    }
    arm_outputs: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {a: {} for a in arms}
    arm_pert_stats = {}

    for aname, (dkey, cstrat) in arms.items():
        rng_p = np.random.default_rng(42)
        p_ret, p_base_v, p_abs, p_rel = 0, 0, [], []
        for cf in cached:
            H, W = cf["img_bgr"].shape[:2]
            dmap = cf[dkey]
            for i, bx in enumerate(cf["boxes"]):
                nbrs = [cf["boxes"][j] for j in range(len(cf["boxes"])) if j != i]
                o = extract_cavity_with_strategy(
                    dmap, bx, strategy=cstrat, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], opt_cfg=opt_cfg, core_cfg=core_cfg
                )
                arm_outputs[aname][(cf["entry"]["frame_id"], i)] = o

                bw, bh = bx[2] - bx[0], bx[3] - bx[1]
                dx1, dx2 = rng_p.uniform(-0.08 * bw, 0.08 * bw, size=2)
                dy1, dy2 = rng_p.uniform(-0.08 * bh, 0.08 * bh, size=2)
                pbx = (max(0.0, bx[0] + dx1), max(0.0, bx[1] + dy1), min(float(W), bx[2] + dx2), min(float(H), bx[3] + dy2))
                op = extract_cavity_with_strategy(
                    dmap, pbx, strategy=cstrat, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], opt_cfg=opt_cfg, core_cfg=core_cfg
                )
                if o["is_valid"]:
                    p_base_v += 1
                    if op["is_valid"]:
                        p_ret += 1
                        v0, v1 = o["core_cavity_stats"]["p75_residual"], op["core_cavity_stats"]["p75_residual"]
                        p_abs.append(abs(v1 - v0))
                        p_rel.append(_guarded_rel_diff_pct(v1, v0))
        arm_pert_stats[aname] = {
            "retained": f"{p_ret}/{p_base_v}",
            "median_abs_shift": round(float(np.median(p_abs)), 4) if p_abs else None,
            "median_rel_shift_pct": round(float(np.median(p_rel)), 2) if p_rel else None,
        }

    keys = list(arm_outputs["midas_full"].keys())
    common_all4 = [k for k in keys if all(arm_outputs[a][k]["is_valid"] for a in arms)]
    arm_summary = {}
    for aname in arms:
        vals = [arm_outputs[aname][k] for k in keys if arm_outputs[aname][k]["is_valid"]]
        inval = [arm_outputs[aname][k] for k in keys if not arm_outputs[aname][k]["is_valid"]]
        arm_summary[aname] = {
            "total_regions": len(keys),
            "valid_count": len(vals),
            "valid_pct": round(100.0 * len(vals) / max(1, len(keys)), 2),
            "failure_reasons": dict(Counter(r for o in inval for r in o["failure_reasons"])),
            "water_warnings": sum(1 for o in arm_outputs[aname].values() if o["is_water_suspected"]),
            "inlier_ratio_median": round(float(np.median([o["ref_diagnostics"]["inlier_ratio"] for o in arm_outputs[aname].values()])), 4),
            "ref_mad_median": round(float(np.median([o["ref_diagnostics"]["ref_fit_mad"] for o in arm_outputs[aname].values()])), 4),
            "admitted_scale_norm_p75_median": round(float(np.median([o["core_cavity_stats"]["scale_normalized_p75"] for o in vals])), 4) if vals else None,
            "common_all4_scale_norm_p75_median": round(float(np.median([arm_outputs[aname][k]["core_cavity_stats"]["scale_normalized_p75"] for k in common_all4])), 4) if common_all4 else None,
            "perturbation_8pct": arm_pert_stats[aname],
        }

    return {
        "total_dev_regions": len(keys),
        "common_all4_valid": len(common_all4),
        "cavity_strategy_ablation_midas": cav_summary,
        "arms_2x2_dev": arm_summary,
    }, cached


def run_exploratory_video_and_approach(
    det_model: YOLO,
    midas: MiDaSSmallAdapter,
    dav2: DepthAnythingV2SmallAdapter,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Task 7 (Exploratory Grouped Video Evaluation), Task 8 (Approach-Distance), & Task 9 (RTX 2050 Timing).
    Uses strictly identical YOLO+ByteTrack proposals and timestamps per frame.
    """
    opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    core_cfg = CoreExtractionConfig()

    arms = {
        "midas_full": ("midas", "full_box"),
        "midas_elliptical": ("midas", "elliptical_core"),
        "dav2_full": ("dav2", "full_box"),
        "dav2_elliptical": ("dav2", "elliptical_core"),
    }

    per_region_csv = []
    per_track_obs: Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]] = {a: {} for a in arms}
    per_source_stats: Dict[str, Any] = {}
    all_arm_recs: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}

    # Benchmark accumulators cleanly separated per model & proposal bucket
    t_midas_256 = {"pre": [], "inf": [], "res": [], "tot": [], "scale": {"0": [], "1": [], "2": [], "3": [], "4+": []}}
    t_dav2_518 = {"pre": [], "inf": [], "res": [], "tot": [], "scale": {"0": [], "1": [], "2": [], "3": [], "4+": []}}
    t_det_track = []
    # Keep first 60 ruralRoad active frames for DAv2 266 smaller-input runtime ablation
    small_input_test_frames = []
    # Also keep representative urban frames for 10-panel plots
    viz_samples: List[Dict[str, Any]] = []

    for src in EXPLORATORY_VIDEO_SOURCES:
        sid = src["source_id"]
        cap = cv2.VideoCapture(os.path.join(BASE_DIR, src["path"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src["start_frame"])
        det_model.predictor = None

        s_counts = {a: Counter() for a in arms}
        s_reasons = {a: Counter() for a in arms}
        s_dets = 0

        for off in range(src["count"]):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            f_idx = src["start_frame"] + off
            ts = round(f_idx / fps, 4)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            td0 = time.perf_counter()
            res = det_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.50, imgsz=640, verbose=False)[0]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_dt = (time.perf_counter() - td0) * 1000.0
            t_det_track.append(t_dt)

            boxes, confs, tids = [], [], []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy().tolist()]
                confs = [float(c) for c in res.boxes.conf.cpu().numpy().tolist()]
                tids = [int(t) for t in res.boxes.id.int().cpu().tolist()] if res.boxes.id is not None else [i + 1 for i in range(len(boxes))]

            nb_key = "4+" if len(boxes) >= 4 else str(len(boxes))
            if not boxes:
                t_midas_256["scale"]["0"].append(t_dt)
                t_dav2_518["scale"]["0"].append(t_dt)
                continue

            pm = midas.predict(frame)
            pd = dav2.predict(frame)

            for bucket, pr in ((t_midas_256, pm), (t_dav2_518, pd)):
                bucket["pre"].append(pr.timings_ms["preprocess_ms"])
                bucket["inf"].append(pr.timings_ms["inference_ms"])
                bucket["res"].append(pr.timings_ms["resize_ms"])
                bucket["tot"].append(pr.timings_ms["total_ms"])

            if sid == "ruralRoad_f0_119" and len(small_input_test_frames) < 40:
                small_input_test_frames.append(frame)

            dmap_m, dmap_d = pm.depth_map, pd.depth_map
            q25_m, q75_m = np.percentile(dmap_m, [25.0, 75.0])
            iqr_m = max(1e-4, float(q75_m - q25_m))
            q25_d, q75_d = np.percentile(dmap_d, [25.0, 75.0])
            iqr_d = max(1e-4, float(q75_d - q25_d))

            ext_sum_m, ext_sum_d = 0.0, 0.0
            for idx_b, (bx, cf, tid) in enumerate(zip(boxes, confs, tids)):
                s_dets += 1
                nbrs = [boxes[j] for j in range(len(boxes)) if j != idx_b]
                bw, bh = max(1.0, bx[2] - bx[0]), max(1.0, bx[3] - bx[1])
                box_area = bw * bh
                area_frac = max(1e-5, box_area / float(H * W))
                y2_ratio = bx[3] / float(H)

                # Precompute ANNULAR region & fit once per depth model (fair constant reference!)
                reg_m = build_and_sample_annular_region(dmap_m, bx, neighbor_boxes=nbrs, cfg=opt_cfg)
                fit_m = fit_ransac_optimized(dmap_m, reg_m, cfg=opt_cfg)
                reg_d = build_and_sample_annular_region(dmap_d, bx, neighbor_boxes=nbrs, cfg=opt_cfg)
                fit_d = fit_ransac_optimized(dmap_d, reg_d, cfg=opt_cfg)

                arm_box_res = {}
                for aname, (mtag, cstrat) in arms.items():
                    dmap_cur = dmap_m if mtag == "midas" else dmap_d
                    reg_cur = reg_m if mtag == "midas" else reg_d
                    fit_cur = fit_m if mtag == "midas" else fit_d
                    fiqr_cur = iqr_m if mtag == "midas" else iqr_d

                    want_arr = (
                        (sid == "rawSurvey_720p_f0_119" and f_idx in (0, 21, 81) and idx_b == 0)
                        or (sid == "ruralRoad_f0_119" and f_idx == 45 and idx_b == 0)
                    )
                    o = extract_cavity_with_strategy(
                        dmap_cur, bx, strategy=cstrat, neighbor_boxes=nbrs, img_bgr=frame,
                        opt_cfg=opt_cfg, core_cfg=core_cfg, precomputed_region=reg_cur, precomputed_fit=fit_cur,
                        return_arrays=want_arr
                    )
                    arm_box_res[aname] = o
                    if aname == "midas_elliptical":
                        ext_sum_m += o["total_ms"]
                    elif aname == "dav2_elliptical":
                        ext_sum_d += o["total_ms"]

                    s_counts[aname]["total"] += 1
                    if o["is_valid"]:
                        s_counts[aname]["valid"] += 1
                    else:
                        for rs in o["failure_reasons"]:
                            s_reasons[aname][rs] += 1

                    st = o["core_cavity_stats"]
                    p75_raw = st["p75_residual"]
                    sn_p75 = st["scale_normalized_p75"]
                    fiqr_p75 = (p75_raw / fiqr_cur) if (p75_raw is not None) else None
                    area_norm = (fiqr_p75 / np.sqrt(area_frac)) if (fiqr_p75 is not None) else None
                    y2_norm = (fiqr_p75 / max(0.05, y2_ratio - 0.15)) if (fiqr_p75 is not None) else None

                    rec = {
                        "source_id": sid, "frame_idx": f_idx, "ts": ts, "track_id": tid,
                        "conf": round(cf, 4), "box_area": round(box_area, 1), "area_frac": round(area_frac, 6),
                        "y2_ratio": round(y2_ratio, 4), "arm": aname, "is_valid": o["is_valid"],
                        "failure_reasons": "|".join(o["failure_reasons"]),
                        "ref_mad": o["ref_diagnostics"]["ref_fit_mad"],
                        "inlier_ratio": o["ref_diagnostics"]["inlier_ratio"],
                        "raw_p75": p75_raw, "raw_diag_p75": st["raw_diag_p75"],
                        "scale_norm_p75": sn_p75, "frame_iqr_norm_p75": round(fiqr_p75, 6) if fiqr_p75 is not None else None,
                        "area_weighted_iqr_p75": round(area_norm, 6) if area_norm is not None else None,
                        "y2_weighted_iqr_p75": round(y2_norm, 6) if y2_norm is not None else None,
                        "ms": o["total_ms"],
                    }
                    all_arm_recs[aname].append(rec)
                    per_track_obs[aname].setdefault((sid, tid), []).append(rec)
                    per_region_csv.append(rec)

                if (sid == "rawSurvey_720p_f0_119" and f_idx in (0, 21, 81) and idx_b == 0) or (sid == "ruralRoad_f0_119" and f_idx == 45 and idx_b == 0):
                    viz_samples.append({
                        "tag": f"{sid}_f{f_idx:04d}_r{idx_b}",
                        "img_bgr": frame.copy(),
                        "box": bx,
                        "d_midas": dmap_m.copy(),
                        "d_dav2": dmap_d.copy(),
                        "arms": arm_box_res,
                    })

            t_midas_256["scale"][nb_key].append(t_dt + pm.timings_ms["total_ms"] + ext_sum_m)
            t_dav2_518["scale"][nb_key].append(t_dt + pd.timings_ms["total_ms"] + ext_sum_d)

        cap.release()
        per_source_stats[sid] = {
            "prior_exposure": src["prior_exposure"],
            "resolution": f"{W}x{H}",
            "total_detections": s_dets,
            "arms": {
                a: {
                    "valid": s_counts[a]["valid"],
                    "valid_pct": round(100.0 * s_counts[a]["valid"] / max(1, s_counts[a]["total"]), 2),
                    "reasons": dict(s_reasons[a]),
                }
                for a in arms
            },
        }

    # Exploratory video aggregate across all 360 frames
    n_tot = len(all_arm_recs["midas_full"])
    common_all4_idx = [i for i in range(n_tot) if all(all_arm_recs[a][i]["is_valid"] for a in arms)]
    common_ell_pair = [i for i in range(n_tot) if all_arm_recs["midas_elliptical"][i]["is_valid"] and all_arm_recs["dav2_elliptical"][i]["is_valid"]]

    video_arm_summary = {}
    for a in arms:
        recs = all_arm_recs[a]
        vals = [r for r in recs if r["is_valid"]]
        inval = [r for r in recs if not r["is_valid"]]
        video_arm_summary[a] = {
            "total_regions": len(recs),
            "valid_count": len(vals),
            "valid_pct": round(100.0 * len(vals) / max(1, len(recs)), 2),
            "unexpected_polarity_count": sum(1 for r in inval if "unexpected_residual_polarity" in r["failure_reasons"]),
            "low_inlier_count": sum(1 for r in inval if "low_inlier_ratio" in r["failure_reasons"]),
            "water_count": sum(1 for r in inval if "suspected_water_reflection" in r["failure_reasons"]),
            "inlier_ratio_median": round(float(np.median([r["inlier_ratio"] for r in recs])), 4),
            "ref_mad_median": round(float(np.median([r["ref_mad"] for r in recs])), 4),
            "admitted_scale_norm_p75_med": round(float(np.median([r["scale_norm_p75"] for r in vals])), 4) if vals else None,
            "common_all4_scale_norm_p75_med": round(float(np.median([recs[i]["scale_norm_p75"] for i in common_all4_idx])), 4) if common_all4_idx else None,
        }

    # Task 8: Approach-Distance Analysis (Within-Track vs Pooled across all 5 normalizations)
    approach_analysis: Dict[str, Any] = {}
    per_track_csv = []
    norm_fields = ("raw_p75", "scale_norm_p75", "frame_iqr_norm_p75", "area_weighted_iqr_p75", "y2_weighted_iqr_p75")

    for a in ("midas_elliptical", "dav2_elliptical"):
        within_rhos_area = {nf: [] for nf in norm_fields}
        within_rhos_y2 = {nf: [] for nf in norm_fields}
        within_cvs = {nf: [] for nf in norm_fields}
        pooled_vecs = {nf: [] for nf in norm_fields}
        pooled_areas, pooled_y2 = [], []
        tracks_ge5 = 0

        for (sid, tid), tlist in sorted(per_track_obs[a].items()):
            vlist = sorted([x for x in tlist if x["is_valid"] and x["raw_p75"] is not None], key=lambda z: z["frame_idx"])
            for r in vlist:
                pooled_areas.append(r["box_area"])
                pooled_y2.append(r["y2_ratio"])
                for nf in norm_fields:
                    pooled_vecs[nf].append(r[nf])
            if len(vlist) >= 5:
                tracks_ge5 += 1
                areas = np.asarray([r["box_area"] for r in vlist], dtype=np.float64)
                y2s = np.asarray([r["y2_ratio"] for r in vlist], dtype=np.float64)
                tr_row = {"arm": a, "source_id": sid, "track_id": tid, "valid_obs": len(vlist)}
                for nf in norm_fields:
                    vals = np.asarray([r[nf] for r in vlist], dtype=np.float64)
                    ra = _spearman_rho(areas, vals)
                    ry = _spearman_rho(y2s, vals)
                    cv = float(np.std(vals, ddof=1) / max(1e-6, np.mean(vals))) if (np.mean(vals) > 1e-4) else None
                    if ra is not None: within_rhos_area[nf].append(ra)
                    if ry is not None: within_rhos_y2[nf].append(ry)
                    if cv is not None: within_cvs[nf].append(cv)
                    tr_row[f"{nf}_rho_area"] = round(ra, 4) if ra is not None else None
                    tr_row[f"{nf}_cv"] = round(cv, 4) if cv is not None else None
                per_track_csv.append(tr_row)

        approach_analysis[a] = {
            "tracks_with_ge5_valid_obs": tracks_ge5,
            "pooled_valid_obs": len(pooled_areas),
            "normalizations": {
                nf: {
                    "within_track_mean_rho_vs_area": round(float(np.mean(within_rhos_area[nf])), 4) if within_rhos_area[nf] else None,
                    "within_track_mean_rho_vs_y2": round(float(np.mean(within_rhos_y2[nf])), 4) if within_rhos_y2[nf] else None,
                    "within_track_mean_cv": round(float(np.mean(within_cvs[nf])), 4) if within_cvs[nf] else None,
                    "pooled_rho_vs_area": round(_spearman_rho(np.asarray(pooled_areas), np.asarray(pooled_vecs[nf])), 4) if len(pooled_areas) >= 5 else None,
                    "pooled_rho_vs_y2": round(_spearman_rho(np.asarray(pooled_y2), np.asarray(pooled_vecs[nf])), 4) if len(pooled_y2) >= 5 else None,
                }
                for nf in norm_fields
            },
        }

    # Test strict affine invariance (`5.0 * d + 100.0`) of all 5 normalizations on urban sample frame 0
    aff_frame = small_input_test_frames[0] if small_input_test_frames else None
    aff_check = {}
    if aff_frame is not None:
        H0, W0 = aff_frame.shape[:2]
        dm0 = midas.predict(aff_frame).depth_map
        dm1 = 5.0 * dm0 + 100.0
        iqr0 = max(1e-4, float(np.percentile(dm0, 75) - np.percentile(dm0, 25)))
        iqr1 = max(1e-4, float(np.percentile(dm1, 75) - np.percentile(dm1, 25)))
        bx0 = (W0 * 0.30, H0 * 0.55, W0 * 0.65, H0 * 0.85)
        o0 = extract_cavity_with_strategy(dm0, bx0, strategy="elliptical_core")
        o1 = extract_cavity_with_strategy(dm1, bx0, strategy="elliptical_core")
        if o0["is_valid"] and o1["is_valid"]:
            r0, r1 = o0["core_cavity_stats"]["p75_residual"], o1["core_cavity_stats"]["p75_residual"]
            sn0, sn1 = o0["core_cavity_stats"]["scale_normalized_p75"], o1["core_cavity_stats"]["scale_normalized_p75"]
            fq0, fq1 = r0 / iqr0, r1 / iqr1
            ar0, ar1 = fq0 / np.sqrt((0.35 * 0.30)), fq1 / np.sqrt((0.35 * 0.30))
            aff_check = {
                "scale_factor": 5.0, "shift": 100.0,
                "raw_p75_ratio_5x": round(r1 / max(1e-9, r0), 4),
                "scale_norm_diff": round(abs(sn1 - sn0), 5),
                "frame_iqr_norm_diff": round(abs(fq1 - fq0), 6),
                "area_weighted_iqr_diff": round(abs(ar1 - ar0), 6),
            }
    approach_analysis["affine_invariance_5x_plus_100"] = aff_check

    # Task 9: DAv2 Smaller-Input Runtime Ablation (266 vs 518) + Separate Synchronized Memory Benchmark
    dav2_266 = DepthAnythingV2SmallAdapter(device=dav2.device, input_size=266, use_fp16=dav2.use_fp16)
    dav2_266.warmup(360, 640)
    t_dav2_266 = {"pre": [], "inf": [], "res": [], "tot": []}
    for frm in small_input_test_frames[:30]:
        pr2 = dav2_266.predict(frm)
        t_dav2_266["pre"].append(pr2.timings_ms["preprocess_ms"])
        t_dav2_266["inf"].append(pr2.timings_ms["inference_ms"])
        t_dav2_266["res"].append(pr2.timings_ms["resize_ms"])
        t_dav2_266["tot"].append(pr2.timings_ms["total_ms"])

    # Dedicated GPU peak memory measurement per model in isolation
    vram_stats = {}
    if torch.cuda.is_available() and small_input_test_frames:
        test_frm = small_input_test_frames[0]
        for label, adp in (("midas_256", midas), ("dav2_518", dav2), ("dav2_266", dav2_266)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            base_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
            _ = adp.predict(test_frm)
            torch.cuda.synchronize()
            vram_stats[label] = {
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2),
                "delta_inference_mb": round((torch.cuda.max_memory_allocated(0) / (1024 ** 2)) - base_alloc, 2),
            }

    perf_summary = {
        "detector_tracking_640_ms": compute_latency_stats(t_det_track),
        "models": {
            "midas_small_256": {
                "load_ms": midas.load_ms, "params": midas.param_count, "precision": "fp16_autocast" if midas.use_fp16 else "fp32",
                "preprocess_ms": compute_latency_stats(t_midas_256["pre"]),
                "inference_ms": compute_latency_stats(t_midas_256["inf"]),
                "resize_ms": compute_latency_stats(t_midas_256["res"]),
                "total_depth_ms": compute_latency_stats(t_midas_256["tot"]),
                "scaling_e2e_ms": {k: compute_latency_stats(v) for k, v in t_midas_256["scale"].items() if v},
                "vram": vram_stats.get("midas_256"),
            },
            "dav2_small_518_native": {
                "load_ms": dav2.load_ms, "params": dav2.param_count, "ckpt_bytes": dav2.ckpt_bytes, "ckpt_sha256": dav2.ckpt_sha256,
                "precision": "fp16_autocast" if dav2.use_fp16 else "fp32",
                "preprocess_ms": compute_latency_stats(t_dav2_518["pre"]),
                "inference_ms": compute_latency_stats(t_dav2_518["inf"]),
                "resize_ms": compute_latency_stats(t_dav2_518["res"]),
                "total_depth_ms": compute_latency_stats(t_dav2_518["tot"]),
                "scaling_e2e_ms": {k: compute_latency_stats(v) for k, v in t_dav2_518["scale"].items() if v},
                "vram": vram_stats.get("dav2_518"),
            },
            "dav2_small_266_ablation_note_speed_only": {
                "preprocess_ms": compute_latency_stats(t_dav2_266["pre"]),
                "inference_ms": compute_latency_stats(t_dav2_266["inf"]),
                "resize_ms": compute_latency_stats(t_dav2_266["res"]),
                "total_depth_ms": compute_latency_stats(t_dav2_266["tot"]),
                "vram": vram_stats.get("dav2_266"),
            },
        },
    }

    # Write CSV exports
    with open(os.path.join(BENCH_DIR, "phase6a_per_region.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_region_csv[0].keys()))
        w.writeheader()
        w.writerows(per_region_csv)

    if per_track_csv:
        with open(os.path.join(BENCH_DIR, "phase6a_per_track.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_track_csv[0].keys()))
            w.writeheader()
            w.writerows(per_track_csv)

    # Render Task 10 diagnostic panels + benchmark charts
    viz_paths = _render_phase6a_visuals(viz_samples, perf_summary, approach_analysis)

    return {
        "note": "EXPLORATORY ONLY — uses shared dev video sessions without independent field physical GT.",
        "total_matched_regions": n_tot,
        "common_valid_all4": len(common_all4_idx),
        "common_valid_elliptical_pair": len(common_ell_pair),
        "per_source_breakdown": per_source_stats,
        "aggregate_arms": video_arm_summary,
    }, approach_analysis, perf_summary, viz_paths


def _render_phase6a_visuals(
    viz_samples: List[Dict[str, Any]],
    perf_summary: Dict[str, Any],
    approach_analysis: Dict[str, Any],
) -> List[str]:
    paths = []

    # 1. Task 10 10-Panel Comparative Panels (Original+Box, MiDaS Depth, DAv2 Depth, Annular Ring,
    #    Full vs Elliptical Core, MiDaS ResMap, DAv2 ResMap, Verdicts)
    for item in viz_samples[:4]:
        rgb = cv2.cvtColor(item["img_bgr"], cv2.COLOR_BGR2RGB)
        arm_mf = item["arms"]["midas_full"]
        arm_me = item["arms"]["midas_elliptical"]
        arm_df = item["arms"]["dav2_full"]
        arm_de = item["arms"]["dav2_elliptical"]

        reg = arm_me["arrays"]["region"]
        x1, y1, x2, y2 = reg["box_xyxy"]
        rx1, ry1, rx2, ry2 = reg["outer_xyxy"]
        outer_rgb = rgb[ry1:ry2, rx1:rx2]

        fig = plt.figure(figsize=(17.5, 9.6), dpi=130)
        gs = GridSpec(2, 5, figure=fig, hspace=0.28, wspace=0.24)

        # Panel 1: Original Image + Detector Box
        a1 = fig.add_subplot(gs[0, 0])
        vis_b = rgb.copy()
        cv2.rectangle(vis_b, (rx1, ry1), (rx2, ry2), (56, 189, 248), 2)
        cv2.rectangle(vis_b, (x1, y1), (x2, y2), (34, 197, 94), 3)
        a1.imshow(vis_b); a1.set_title("1. ROI Box (Green) & Outer", fontsize=9); a1.axis("off")

        # Panel 2: MiDaS_small Depth Map
        a2 = fig.add_subplot(gs[0, 1])
        i2 = a2.imshow(item["d_midas"], cmap="inferno")
        a2.set_title("2. MiDaS Small (Rel. Inv-Depth)", fontsize=9); plt.colorbar(i2, ax=a2, fraction=0.046); a2.axis("off")

        # Panel 3: Depth Anything V2 Small Depth Map
        a3 = fig.add_subplot(gs[0, 2])
        i3 = a3.imshow(item["d_dav2"], cmap="inferno")
        a3.set_title("3. Depth Anything V2 Small (Rel.)", fontsize=9); plt.colorbar(i3, ax=a3, fraction=0.046); a3.axis("off")

        # Panel 4: Annular Reference Ring Overlay
        a4 = fig.add_subplot(gs[0, 3])
        ring_v = (outer_rgb * 0.45).astype(np.uint8)
        ring_v[reg["valid_ring_mask"]] = [56, 189, 248]
        a4.imshow(ring_v); a4.set_title(f"4. Annular Ring (n={reg['sampled_ref_samples']}/{reg['available_ref_samples']})", fontsize=9); a4.axis("off")

        # Panel 5: Full Cavity vs Elliptical Core Geometry
        a5 = fig.add_subplot(gs[0, 4])
        box_rgb = rgb[y1:y2, x1:x2].copy()
        cmask = arm_me["arrays"]["core_mask"]
        core_vis = (box_rgb * 0.45).astype(np.uint8)
        core_vis[~cmask] = [239, 68, 68]
        core_vis[cmask] = box_rgb[cmask]
        a5.imshow(core_vis); a5.set_title(f"5. Full Box vs Core ({arm_me['interior_coverage_ratio']*100:.0f}%)", fontsize=9); a5.axis("off")

        # Panel 6 & 7: MiDaS Signed Residual Full vs Core Hist/Map
        a6 = fig.add_subplot(gs[1, 0])
        rm_m = arm_mf["arrays"]["signed_residual_map"]
        vm = max(1e-2, float(np.nanpercentile(np.abs(rm_m), 98)))
        i6 = a6.imshow(rm_m, cmap="coolwarm", vmin=-vm, vmax=vm)
        a6.set_title("6. MiDaS Signed Res (+ = Deeper)", fontsize=9); plt.colorbar(i6, ax=a6, fraction=0.046); a6.axis("off")

        a7 = fig.add_subplot(gs[1, 1])
        rm_d = arm_df["arrays"]["signed_residual_map"]
        vd = max(1e-3, float(np.nanpercentile(np.abs(rm_d), 98)))
        i7 = a7.imshow(rm_d, cmap="coolwarm", vmin=-vd, vmax=vd)
        a7.set_title("7. DAv2 Signed Res (+ = Deeper)", fontsize=9); plt.colorbar(i7, ax=a7, fraction=0.046); a7.axis("off")

        # Panel 8: Core Residual Distributions Comparison (Z-normalized vs ref MAD for visual comparison ONLY!)
        a8 = fig.add_subplot(gs[1, 2])
        fin_m = rm_m[np.isfinite(rm_m)] / max(1e-4, 1.4826 * arm_me["ref_diagnostics"]["ref_fit_mad"])
        fin_d = rm_d[np.isfinite(rm_d)] / max(1e-4, 1.4826 * arm_de["ref_diagnostics"]["ref_fit_mad"])
        a8.hist(fin_m.ravel(), bins=25, alpha=0.55, color="#2563eb", label="MiDaS / RefNoise")
        a8.hist(fin_d.ravel(), bins=25, alpha=0.55, color="#16a34a", label="DAv2 / RefNoise")
        a8.axvline(0.0, color="black", linestyle="--", linewidth=1)
        a8.set_title("8. Noise-Normalized Residuals", fontsize=9); a8.legend(fontsize=7.5)

        # Panel 9: Reference Fit Diagnostics Text
        a9 = fig.add_subplot(gs[1, 3]); a9.axis("off")
        t9 = [
            "MiDaS Ref Diagnostics:",
            f"  Inlier: {arm_me['ref_diagnostics']['inlier_ratio']:.3f}",
            f"  RefMAD: {arm_me['ref_diagnostics']['ref_fit_mad']:.4f}",
            f"  FullP75: {arm_mf['core_cavity_stats']['raw_diag_p75']:.3f}",
            f"  CoreP75: {arm_me['core_cavity_stats']['raw_diag_p75']:.3f}",
            "",
            "DAv2 Small Ref Diagnostics:",
            f"  Inlier: {arm_de['ref_diagnostics']['inlier_ratio']:.3f}",
            f"  RefMAD: {arm_de['ref_diagnostics']['ref_fit_mad']:.4f}",
            f"  FullP75: {arm_df['core_cavity_stats']['raw_diag_p75']:.3f}",
            f"  CoreP75: {arm_de['core_cavity_stats']['raw_diag_p75']:.3f}",
            "(Different scales! Relative ONLY)",
        ]
        a9.text(0.02, 0.95, "\n".join(t9), va="top", family="monospace", fontsize=8.0, bbox=dict(boxstyle="round", facecolor="#f8fafc"))
        a9.set_title("9. Fit Quality & Scale Diff", fontsize=9)

        # Panel 10: Validity & Failure Verdicts
        a10 = fig.add_subplot(gs[1, 4]); a10.axis("off")
        t10 = [
            f"MiDaS Full : {'VALID' if arm_mf['is_valid'] else arm_mf['failure_reasons']}",
            f"MiDaS Core : {'VALID' if arm_me['is_valid'] else arm_me['failure_reasons']}",
            f"DAv2 Full  : {'VALID' if arm_df['is_valid'] else arm_df['failure_reasons']}",
            f"DAv2 Core  : {'VALID' if arm_de['is_valid'] else arm_de['failure_reasons']}",
            "",
            f"Water Flag : {arm_me['is_water_suspected']}",
            f"Warnings   : {arm_me['warnings'] or 'None'}",
        ]
        a10.text(0.02, 0.95, "\n".join(t10), va="top", family="monospace", fontsize=7.8, bbox=dict(boxstyle="round", facecolor="#fefce8"))
        a10.set_title("10. Quality & Rejection Verdicts", fontsize=9)

        fig.suptitle(f"Phase 6A Comparative Diagnostic | {item['tag']} (Relative Units Only)", fontsize=11.5, fontweight="bold")
        op = os.path.join(VIZ_DIR, f"panel10_{item['tag']}.png")
        plt.savefig(op, bbox_inches="tight"); plt.close(fig)
        paths.append(op)

    # 2. RTX 2050 Scaling & Approach-Distance Normalization Chart
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), dpi=130)
    cats = ["0", "1", "2", "3", "4+"]
    ms_sc = perf_summary["models"]["midas_small_256"]["scaling_e2e_ms"]
    dv_sc = perf_summary["models"]["dav2_small_518_native"]["scaling_e2e_ms"]
    axes[0].plot(cats, [ms_sc.get(c, {}).get("mean_ms", np.nan) for c in cats], "o-", color="#2563eb", linewidth=2, label="MiDaS Small (256) E2E")
    axes[0].plot(cats, [dv_sc.get(c, {}).get("mean_ms", np.nan) for c in cats], "s-", color="#16a34a", linewidth=2, label="Depth Anything V2 Small (518) E2E")
    axes[0].set_xlabel("Proposals per Frame"); axes[0].set_ylabel("Mean Active Frame Time (ms)")
    axes[0].set_title("RTX 2050 End-to-End Scaling (YOLO+ByteTrack + Depth + RANSAC)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    nfs = ["raw_p75", "scale_norm_p75", "frame_iqr_norm_p75", "area_weighted_iqr_p75", "y2_weighted_iqr_p75"]
    n_lbl = ["Raw P75", "Local MAD", "Frame IQR", "Area+IQR", "Y2+IQR"]
    r_m = [approach_analysis["midas_elliptical"]["normalizations"][nf]["within_track_mean_rho_vs_area"] or 0.0 for nf in nfs]
    r_d = [approach_analysis["dav2_elliptical"]["normalizations"][nf]["within_track_mean_rho_vs_area"] or 0.0 for nf in nfs]
    x = np.arange(len(nfs))
    axes[1].bar(x - 0.18, r_m, width=0.35, color="#2563eb", label="MiDaS Elliptical")
    axes[1].bar(x + 0.18, r_d, width=0.35, color="#16a34a", label="DAv2 Elliptical")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels(n_lbl, fontsize=8.5)
    axes[1].set_ylabel("Mean Within-Track Spearman ρ vs Box Area")
    axes[1].set_title("Approach-Distance Drift Across Normalizations")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)

    p_chart = os.path.join(VIZ_DIR, "phase6a_perf_and_approach_drift.png")
    plt.savefig(p_chart, bbox_inches="tight"); plt.close(fig)
    paths.append(p_chart)
    return paths


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = YOLO(DEFAULT_DET_WEIGHTS)
    midas = MiDaSSmallAdapter(device=device_str)
    dav2 = DepthAnythingV2SmallAdapter(device=device_str, input_size=518)

    # Synchronized warmup for all models before timing
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    _ = det_model.predict(dummy, imgsz=640, conf=0.25, iou=0.70, device=0 if device_str == "cuda" else "cpu", verbose=False)
    midas.warmup(360, 640)
    dav2.warmup(360, 640)

    synth_core = run_synthetic_core_suite()
    dev_comp, _ = run_dev_36_comparison(det_model, midas, dav2)
    video_comp, approach_res, perf_res, viz_paths = run_exploratory_video_and_approach(det_model, midas, dav2)

    out_json = {
        "experiment": "Phase 6A Core-Aware Cavity Extraction & MiDaS vs Depth Anything V2 Small",
        "synthetic_core_suite_task4": synth_core,
        "dev_36f_comparison_task3_7": dev_comp,
        "exploratory_video_360f_task7": video_comp,
        "approach_distance_analysis_task8": approach_res,
        "rtx2050_performance_task9": perf_res,
        "visualization_paths": viz_paths,
    }
    json_path = os.path.join(BENCH_DIR, "phase6a_depth_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print("=== PHASE 6A BENCHMARK COMPLETE ===")
    print("Synthetic core summary:", json.dumps(synth_core["summary"], indent=2))
    print("Exploratory 360f arms:", json.dumps(video_comp["aggregate_arms"], indent=2))


if __name__ == "__main__":
    main()
