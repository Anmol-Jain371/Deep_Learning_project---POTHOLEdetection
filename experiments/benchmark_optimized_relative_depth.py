"""
========================================================================================
RoadVision AI -- Phase 5B: Optimized Adaptive RANSAC & Matched-Track Stability Benchmark
========================================================================================
Evaluates:
  1. Reproducible CPU profiling & RTX 2050 latency scaling (Task 1 & Task 8).
  2. Sampling strategy & point-budget ablation (`256, 512, 1024, 2048, full`) (Task 2 & 3).
  3. Synthetic regression suite: 12 Phase 5A scenarios + 36 randomized multi-seed scenarios
     (`seed in [101, 202, 303]`) comparing baseline vs optimized RANSAC (Task 4).
  4. Matched-region 36-frame dashcam comparison (`existing_annular`, `ols_plane`,
     `original_ransac`, `optimized_ransac`), strictly separating common valid regions vs
     each method's own admitted population (Task 5).
  5. Strictly matched-track temporal stability on `ruralRoad_potHoles.mp4` (`frames 0..119`)
     with identical proposals, track IDs, cached MiDaS maps, and timestamps, plus affine vs
     vehicle-approach normalization checks (Task 6).
  6. Validity-aware temporal filter evaluation on real tracks and synthetic sequences (Task 7).
  7. CSV exports, JSON artifact, performance plots, and visual case comparisons (Task 9).

Strict adherence: zero production changes, zero training, relative units ONLY.
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
    RobustDepthConfig,
    MiDaSDepthRunner,
    extract_relative_depth_for_box,
)
from experiments.optimized_relative_depth import (
    SAMPLING_MODES,
    OptimizedDepthConfig,
    extract_optimized_relative_depth_for_box,
)
from experiments.test_relative_depth_synthetic import (
    build_synthetic_suite,
    make_plane,
    elliptical_cavity,
    irregular_cavity,
)
from experiments.validity_aware_temporal_filter import (
    FILTER_MODES,
    TemporalFilterConfig,
    ValidityAwareTrackFilter,
    build_synthetic_temporal_sequences,
    evaluate_temporal_filter_suite,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "optimized_depth_viz")


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None

    def _avg_ranks(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr, kind="mergesort")
        svals = arr[order]
        n = len(arr)
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and svals[j] == svals[i]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
            i = j
        return ranks

    rx = _avg_ranks(x) - 0.5 * (len(x) + 1.0)
    ry = _avg_ranks(y) - 0.5 * (len(y) + 1.0)
    den = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if den <= 1e-12:
        return None
    return float(np.sum(rx * ry) / den)


def _guarded_rel_diff_pct(new_val: float, base_val: float, floor: float = 1e-3) -> float:
    return 100.0 * abs(new_val - base_val) / max(abs(base_val), floor)


def _summarize_vals(arr: List[float]) -> Dict[str, Any]:
    if not arr:
        return {"count": 0, "mean": None, "median": None, "iqr": None, "mad": None, "p95": None, "max": None}
    a = np.asarray(arr, dtype=np.float64)
    med = float(np.median(a))
    q25, q75, q95 = [float(v) for v in np.percentile(a, [25.0, 75.0, 95.0])]
    return {
        "count": int(len(a)),
        "mean": round(float(np.mean(a)), 4),
        "median": round(med, 4),
        "iqr": round(q75 - q25, 4),
        "mad": round(float(np.median(np.abs(a - med))), 4),
        "p95": round(q95, 4),
        "max": round(float(np.max(a)), 4),
    }


def _build_randomized_regression_scenarios(seeds: List[int]) -> List[Tuple[str, int, str, np.ndarray, np.ndarray, Tuple[int, int, int, int], List[Tuple[int, int, int, int]], bool, float]]:
    """Generate 12 randomized controlled test types per fixed seed (`36` scenarios total)."""
    h, w = 240, 320
    cases = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        a = float(rng.uniform(-0.08, 0.08))
        b = float(rng.uniform(-0.05, 0.05))
        c = float(rng.uniform(6.0, 20.0))
        depth_mag = float(rng.uniform(2.2, 4.8))
        box = (115, 105, 205, 195)
        plane = make_plane(h, w, a=a, b=b, c=c)
        cav = elliptical_cavity(h, w, cy=150.0, cx=160.0, ry=25.0, rx=28.0, max_depth=depth_mag)

        # Ring outer helper mask for random corruption
        ring_m = np.zeros((h, w), dtype=bool)
        ring_m[65:230, 75:245] = True
        ring_m[100:200, 110:210] = False
        ry, rx = np.where(ring_m)

        # 1. Randomized steep plane + irregular cavity
        p1 = make_plane(h, w, a=float(rng.choice([-1, 1])) * float(rng.uniform(0.06, 0.11)), b=b, c=c)
        cases.append((f"rand_steep_irregular_s{sd}", sd, "steep_irregular", p1, p1 - irregular_cavity(h, w, max_depth=depth_mag), box, [], True, 1.0))

        # 2. High Gaussian noise (sigma=0.30..0.40)
        sig = float(rng.uniform(0.30, 0.40))
        d2 = plane - cav + rng.normal(0.0, sig, size=(h, w))
        cases.append((f"rand_high_noise_s{sd}", sd, "gaussian_noise", plane, d2, box, [], True, 1.0))

        # 3. Random outlier fraction 25%..35%
        frac = float(rng.uniform(0.25, 0.35))
        d3 = (plane - cav).copy()
        pick3 = rng.choice(len(ry), size=int(frac * len(ry)), replace=False)
        d3[ry[pick3], rx[pick3]] += float(rng.uniform(8.0, 16.0)) * rng.choice([-1.0, 1.0], size=len(pick3))
        cases.append((f"rand_outliers_s{sd}", sd, "outliers", plane, d3, box, [], True, 1.0))

        # 4. Contiguous curb/edge contamination block (28%..36% outer strip)
        d4 = (plane - cav).copy()
        strip_h = int(rng.integers(24, 34))
        d4[66:66 + strip_h, 76:244] += float(rng.uniform(9.0, 16.0))
        cases.append((f"rand_curb_block_s{sd}", sd, "curb_block", plane, d4, box, [], True, 1.0))

        # 5. Neighboring deep cavity inside annular apron (with neighbor box)
        d5 = (plane - cav).copy()
        d5 -= elliptical_cavity(h, w, cy=148.0, cx=234.0, ry=26.0, rx=20.0, max_depth=11.0)
        cases.append((f"rand_neighbor_s{sd}", sd, "neighbor_cavity", plane, d5, box, [(210, 118, 258, 178)], True, 1.0))

        # 6. Sparse missing NaNs in ring (6%..9%)
        d6 = (plane - cav).copy()
        pick6 = rng.choice(len(ry), size=int(float(rng.uniform(0.06, 0.09)) * len(ry)), replace=False)
        d6[ry[pick6], rx[pick6]] = np.nan
        cases.append((f"rand_sparse_nan_s{sd}", sd, "missing_pixels", plane, d6, box, [], True, 1.0))

        # 7. Corner boundary clipping (near bottom-right edge)
        d7 = plane - elliptical_cavity(h, w, cy=216.0, cx=284.0, ry=19.0, rx=22.0, max_depth=depth_mag)
        cases.append((f"rand_corner_clip_s{sd}", sd, "boundary_clip", plane, d7, (254, 194, 319, 239), [], True, 1.0))

        # 8. Small reference region & small box (28x28) -> sparse pixel budget test
        d8 = plane - elliptical_cavity(h, w, cy=150.0, cx=160.0, ry=10.0, rx=11.0, max_depth=depth_mag)
        cases.append((f"rand_small_box_s{sd}", sd, "small_region", plane, d8, (146, 136, 174, 164), [], True, 1.0))

        # 9. Inverted polarity bump (expected INVALID)
        d9 = plane + cav
        cases.append((f"rand_invalid_bump_s{sd}", sd, "invalid_bump", plane, d9, box, [], False, 1.0))

        # 10. Fatal missing pixels inside crater (>55% NaN -> expected INVALID)
        d10 = (plane - cav).copy()
        d10[105:162, 115:205] = np.nan
        cases.append((f"rand_fatal_nan_s{sd}", sd, "invalid_nan", plane, d10, box, [], False, 1.0))

        # 11. Tiny occluded box starved by 3 large neighbors -> poor retention INVALID
        d11 = plane - elliptical_cavity(h, w, cy=150.0, cx=160.0, ry=10.0, rx=10.0, max_depth=depth_mag)
        nbr11 = [(100, 100, 146, 200), (174, 100, 220, 200), (140, 100, 180, 138)]
        cases.append((f"rand_starved_ring_s{sd}", sd, "invalid_starved", plane, d11, (147, 140, 173, 172), nbr11, False, 1.0))

        # 12. Affine scaled + shifted scene (scale in [2.5, 6.0])
        sc = float(rng.uniform(2.5, 6.0))
        sh = float(rng.uniform(20.0, 120.0))
        cases.append((f"rand_affine_s{sd}", sd, "affine_scale", sc * plane + sh, sc * (plane - cav) + sh, box, [], True, sc))

    return cases


def run_synthetic_regression(opt_cfg_default: OptimizedDepthConfig) -> Dict[str, Any]:
    """Task 4: Re-run 12 Phase 5A scenarios + 36 multi-seed randomized scenarios."""
    base_cfg = RobustDepthConfig()
    suite_12 = build_synthetic_suite()
    rand_36 = _build_randomized_regression_scenarios([101, 202, 303])

    configs_to_compare = {
        "original_ransac_full": ("orig", None),
        "optimized_ransac_strat_512": ("opt", opt_cfg_default),
        "optimized_ransac_strat_256": ("opt", OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=256)),
        "optimized_ransac_full": ("opt", OptimizedDepthConfig(sampling_mode="full", max_ref_samples=None)),
    }

    all_scenarios: List[Tuple[str, str, Optional[int], str, np.ndarray, np.ndarray, Tuple[int, int, int, int], List[Tuple[int, int, int, int]], bool, float]] = []
    for sc_name, true_p, d_map, box, nbrs, exp_v in suite_12:
        sc_div = 5.0 if sc_name == "affine_scale_shift_5x" else 1.0
        all_scenarios.append(("phase5a_12", sc_name, None, "phase5a", true_p, d_map, box, nbrs, exp_v, sc_div))
    for sc_name, sd, grp, true_p, d_map, box, nbrs, exp_v, sc_div in rand_36:
        all_scenarios.append(("randomized_36", sc_name, sd, grp, true_p, d_map, box, nbrs, exp_v, sc_div))

    out_summary: Dict[str, Any] = {}
    per_scenario_rows: Dict[str, Any] = {}

    for label, (kind, cfg_obj) in configs_to_compare.items():
        agg = {
            "phase5a_12": {"match": 0, "total": 0, "inv_match": 0, "inv_tot": 0, "rmse": [], "p75_err": [], "ms": []},
            "randomized_36": {"match": 0, "total": 0, "inv_match": 0, "inv_tot": 0, "rmse": [], "p75_err": [], "ms": []},
            "all_48": {"match": 0, "total": 0, "inv_match": 0, "inv_tot": 0, "rmse": [], "p75_err": [], "ms": []},
        }
        seed_acc: Dict[int, List[float]] = {101: [], 202: [], 303: []}

        for suite_tag, sc_name, sd, grp, true_p, d_map, box, nbrs, exp_v, sc_div in all_scenarios:
            x1, y1, x2, y2 = box
            t_plane = true_p[y1:y2, x1:x2]
            t_res = t_plane - d_map[y1:y2, x1:x2]
            fin = np.isfinite(t_res)
            t_p75 = float(np.percentile(t_res[fin], 75.0)) if fin.any() else 0.0

            if kind == "orig":
                res = extract_relative_depth_for_box(d_map, box, method="ransac_plane", neighbor_boxes=nbrs, cfg=base_cfg, return_arrays=True)
            else:
                res = extract_optimized_relative_depth_for_box(d_map, box, neighbor_boxes=nbrs, cfg=cfg_obj, return_arrays=True)

            pred_p = res["arrays"]["ref_plane_crater"]
            rmse = float(np.sqrt(np.mean((pred_p - t_plane) ** 2))) / sc_div
            est_p75 = res["relative_cavity_stats"]["raw_diagnostic_p75"]
            p75_err = abs(est_p75 - t_p75) / sc_div
            matched = (res["is_valid"] == exp_v)
            ms = res["timing_ms"]["extraction_total_ms"]

            for bucket in (suite_tag, "all_48"):
                agg[bucket]["total"] += 1
                agg[bucket]["match"] += int(matched)
                agg[bucket]["ms"].append(ms)
                if not exp_v:
                    agg[bucket]["inv_tot"] += 1
                    agg[bucket]["inv_match"] += int(not res["is_valid"])
                if exp_v:
                    agg[bucket]["rmse"].append(rmse)
                    agg[bucket]["p75_err"].append(p75_err)

            if sd is not None and exp_v:
                seed_acc[sd].append(p75_err)

            if label in ("original_ransac_full", "optimized_ransac_strat_512"):
                per_scenario_rows.setdefault(sc_name, {"suite": suite_tag, "expected_valid": exp_v})[label] = {
                    "is_valid": res["is_valid"],
                    "match": matched,
                    "failure_reasons": res["failure_reasons"],
                    "norm_ref_rmse": round(rmse, 5),
                    "norm_p75_abs_err": round(p75_err, 5),
                    "time_ms": ms,
                }

        def _fmt_bucket(b: Dict[str, Any]) -> Dict[str, Any]:
            r_arr = np.asarray(b["rmse"], dtype=np.float64)
            e_arr = np.asarray(b["p75_err"], dtype=np.float64)
            return {
                "validity_accuracy": f"{b['match']}/{b['total']}",
                "invalid_case_detection": f"{b['inv_match']}/{b['inv_tot']}",
                "mean_norm_ref_rmse": round(float(np.mean(r_arr)), 5),
                "median_norm_ref_rmse": round(float(np.median(r_arr)), 5),
                "max_norm_ref_rmse": round(float(np.max(r_arr)), 5),
                "mean_norm_p75_err": round(float(np.mean(e_arr)), 5),
                "median_norm_p75_err": round(float(np.median(e_arr)), 5),
                "max_norm_p75_err": round(float(np.max(e_arr)), 5),
                "mean_ms": round(float(np.mean(b["ms"])), 4),
            }

        out_summary[label] = {
            "phase5a_12": _fmt_bucket(agg["phase5a_12"]),
            "randomized_36": _fmt_bucket(agg["randomized_36"]),
            "all_48": _fmt_bucket(agg["all_48"]),
            "randomized_per_seed_mean_p75_err": {str(sd): round(float(np.mean(v)), 5) for sd, v in seed_acc.items()},
        }

    return {"summary": out_summary, "per_scenario_head_to_head": per_scenario_rows}


def run_sampling_ablation_dashcam(cached_frames: List[Dict[str, Any]], orig_outs: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    """Task 2: Compare uniform_stride, stratified_quadrant, deterministic_random x {256, 512, 1024, 2048} + full."""
    variants: List[Tuple[str, str, Optional[int]]] = [("full_budget_none", "full", None)]
    for mode in ("uniform_stride", "stratified_quadrant", "deterministic_random"):
        for b in (256, 512, 1024, 2048):
            variants.append((f"{mode}_{b}", mode, b))

    ablation = {}
    for vname, mode, budget in variants:
        cfg = OptimizedDepthConfig(sampling_mode=mode, max_ref_samples=budget)
        valid_cnt = 0
        matched_orig_valid = 0
        poor_support_cnt = 0
        insuff_cnt = 0
        times_ms = []
        mad_matched, p75_abs_diff_orig, p75_rel_diff_orig = [], [], []
        early_stop_cnt = 0
        iters_eval = []

        for loop_i in range(3):
            for cf in cached_frames:
                if not cf["boxes"] or cf["depth_map"] is None:
                    continue
                for idx, box in enumerate(cf["boxes"]):
                    nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
                    out = extract_optimized_relative_depth_for_box(
                        cf["depth_map"], tuple(box), neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=cfg
                    )
                    times_ms.append(out["timing_ms"]["extraction_total_ms"])
                    if loop_i == 0:
                        if out["is_valid"]:
                            valid_cnt += 1
                        if "poor_spatial_support" in out["failure_reasons"]:
                            poor_support_cnt += 1
                        if "insufficient_reference_samples" in out["failure_reasons"]:
                            insuff_cnt += 1
                        if out["ref_diagnostics"]["early_stopped"]:
                            early_stop_cnt += 1
                        iters_eval.append(out["ref_diagnostics"]["iters_evaluated"])

                        ob = orig_outs[(cf["entry"]["frame_id"], idx)]
                        if ob["is_valid"] and out["is_valid"]:
                            matched_orig_valid += 1
                            mad_matched.append(out["ref_diagnostics"]["ref_fit_mad"])
                            p_base = ob["relative_cavity_stats"]["p75_residual"]
                            p_new = out["relative_cavity_stats"]["p75_residual"]
                            p75_abs_diff_orig.append(abs(p_new - p_base))
                            p75_rel_diff_orig.append(_guarded_rel_diff_pct(p_new, p_base))

        orig_mean_ms = float(np.mean([v["timing_ms"]["extraction_total_ms"] for v in orig_outs.values()]))
        st_t = compute_latency_stats(times_ms)
        ablation[vname] = {
            "sampling_mode": mode,
            "budget": budget,
            "valid_count": valid_cnt,
            "total_regions": len(orig_outs),
            "valid_pct": round(100.0 * valid_cnt / max(1, len(orig_outs)), 2),
            "common_valid_with_orig_ransac": matched_orig_valid,
            "poor_spatial_support_rejections": poor_support_cnt,
            "insufficient_samples_rejections": insuff_cnt,
            "early_stopped_count": early_stop_cnt,
            "mean_iters_evaluated": round(float(np.mean(iters_eval)), 2) if iters_eval else 0.0,
            "matched_ref_mad_median": round(float(np.median(mad_matched)), 4) if mad_matched else None,
            "matched_p75_abs_diff_vs_orig_median": round(float(np.median(p75_abs_diff_orig)), 4) if p75_abs_diff_orig else None,
            "matched_p75_rel_diff_vs_orig_median_pct": round(float(np.median(p75_rel_diff_orig)), 2) if p75_rel_diff_orig else None,
            "latency_ms": st_t,
            "speedup_vs_orig_mean": round(orig_mean_ms / max(1e-6, st_t["mean_ms"]), 2),
        }
    return ablation


def run_matched_track_evaluation(
    det_model: YOLO,
    midas: MiDaSDepthRunner,
    opt_cfg: OptimizedDepthConfig,
) -> Dict[str, Any]:
    """
    Tasks 6 & 7: Cache ByteTrack trajectories (`frames 0..119`) once and evaluate all 4 methods
    + temporal filter modes on 100% identical frames, track IDs, boxes, depth maps, and timestamps.
    """
    vpath = os.path.join(BASE_DIR, "uploads", "upload_1794847200_ruralRoad_potHoles.mp4")
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cached_video = []
    try:
        frame_idx = 0
        while frame_idx < 120:
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_idx / fps
            res = det_model.track(
                frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.5, imgsz=640, verbose=False
            )[0]
            obs = []
            dmap = None
            frame_iqr = 1.0
            if res.boxes is not None and len(res.boxes) > 0 and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy().tolist()
                tids = res.boxes.id.int().cpu().tolist()
                dmap, _ = midas.predict(frame)
                q25, q75 = np.percentile(dmap, [25.0, 75.0])
                frame_iqr = max(1e-4, float(q75 - q25))
                for idx, (box, tid) in enumerate(zip(boxes, tids)):
                    nbrs = [tuple(boxes[j]) for j in range(len(boxes)) if j != idx]
                    obs.append({"track_id": int(tid), "box": tuple(box), "nbrs": nbrs})
            cached_video.append({
                "frame_idx": frame_idx,
                "ts": round(ts, 4),
                "img_bgr": frame,
                "depth_map": dmap,
                "frame_iqr": frame_iqr,
                "obs": obs,
            })
            frame_idx += 1
    finally:
        cap.release()

    methods = ("existing_annular", "ols_plane", "original_ransac", "optimized_ransac")
    base_cfg = RobustDepthConfig()

    # Record per-(frame_idx, track_id) result identically for all 4 methods
    all_keys: List[Tuple[int, int]] = []
    track_ids_seen = set()
    obs_records: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {m: {} for m in methods}
    obs_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}
    csv_obs_rows = []

    for fv in cached_video:
        if fv["depth_map"] is None:
            continue
        f_idx = fv["frame_idx"]
        for item in fv["obs"]:
            tid = item["track_id"]
            key = (f_idx, tid)
            all_keys.append(key)
            track_ids_seen.add(tid)
            bx = item["box"]
            box_area = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
            obs_meta[key] = {"frame_idx": f_idx, "ts": fv["ts"], "track_id": tid, "box_area": box_area, "frame_iqr": fv["frame_iqr"]}

            for m in methods:
                if m == "optimized_ransac":
                    out = extract_optimized_relative_depth_for_box(
                        fv["depth_map"], bx, neighbor_boxes=item["nbrs"], img_bgr=fv["img_bgr"], cfg=opt_cfg
                    )
                else:
                    rm_name = "ransac_plane" if m == "original_ransac" else m
                    out = extract_relative_depth_for_box(
                        fv["depth_map"], bx, method=rm_name, neighbor_boxes=item["nbrs"], img_bgr=fv["img_bgr"], cfg=base_cfg
                    )
                p75 = out["relative_cavity_stats"]["p75_residual"]
                sn_p75 = out["relative_cavity_stats"]["scale_normalized_p75"]
                fiqr_p75 = round(p75 / fv["frame_iqr"], 6) if (p75 is not None) else None
                rec = {
                    "is_valid": out["is_valid"],
                    "p75": p75,
                    "scale_norm_p75": sn_p75,
                    "frame_iqr_norm_p75": fiqr_p75,
                    "ref_mad": out["ref_diagnostics"]["ref_fit_mad"],
                    "failure_reasons": "|".join(out["failure_reasons"]),
                    "ms": out["timing_ms"]["extraction_total_ms"],
                }
                obs_records[m][key] = rec
                csv_obs_rows.append({
                    "frame_idx": f_idx,
                    "ts_sec": fv["ts"],
                    "track_id": tid,
                    "box_area_px": round(box_area, 1),
                    "frame_depth_iqr": round(fv["frame_iqr"], 4),
                    "method": m,
                    "is_valid": rec["is_valid"],
                    "p75_residual": rec["p75"],
                    "scale_normalized_p75": rec["scale_norm_p75"],
                    "frame_iqr_normalized_p75": rec["frame_iqr_norm_p75"],
                    "ref_fit_mad": rec["ref_mad"],
                    "failure_reasons": rec["failure_reasons"],
                    "extraction_ms": rec["ms"],
                })

    total_obs = len(all_keys)
    common_all4_keys = [k for k in all_keys if all(obs_records[m][k]["is_valid"] for m in methods)]
    common_ransac_keys = [k for k in all_keys if obs_records["original_ransac"][k]["is_valid"] and obs_records["optimized_ransac"][k]["is_valid"]]

    # Group observations by track ID
    keys_by_track: Dict[int, List[Tuple[int, int]]] = {}
    for k in sorted(all_keys):
        keys_by_track.setdefault(k[1], []).append(k)

    common_track_ids_all4 = [tid for tid, tkeys in keys_by_track.items() if sum(1 for k in tkeys if k in common_all4_keys) >= 4]
    common_track_ids_ransac = [tid for tid, tkeys in keys_by_track.items() if sum(1 for k in tkeys if k in common_ransac_keys) >= 4]

    def _track_metrics(vals: List[float], areas: List[float], frames: List[int]) -> Dict[str, Any]:
        arr = np.asarray(vals, dtype=np.float64)
        mean_v = float(np.mean(arr))
        med_v = float(np.median(arr))
        diffs = np.abs(np.diff(arr))
        med_abs_delta = float(np.median(diffs)) if len(diffs) > 0 else 0.0
        iqr_disp = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        mad_disp = float(np.median(np.abs(arr - med_v)))
        # Guard CV mathematically when track mean is near zero
        cv = float(np.std(arr, ddof=1) / mean_v) if (len(arr) > 1 and mean_v > 0.05) else None
        rho_area = _spearman_rho(np.asarray(areas, dtype=np.float64), arr)
        rho_frame = _spearman_rho(np.asarray(frames, dtype=np.float64), arr)
        return {
            "med_abs_delta": med_abs_delta,
            "mad_disp": mad_disp,
            "iqr_disp": iqr_disp,
            "cv": cv,
            "rho_area": rho_area,
            "rho_frame": rho_frame,
        }

    # Summarize both (a) Admitted-only vs (b) Common-All-4 Matched Tracks vs (c) Unmatched extra obs
    method_temporal_summary: Dict[str, Any] = {}
    per_track_csv_rows = []
    common_all4_set = set(common_all4_keys)
    common_ransac_set = set(common_ransac_keys)

    for m in methods:
        # 1. Own admitted population
        own_valid_cnt = sum(1 for k in all_keys if obs_records[m][k]["is_valid"])
        dropout_freq = 1.0 - (own_valid_cnt / float(max(1, total_obs)))

        # 2. Matched Common-4 tracks (`tid in common_track_ids_all4`) & Unmatched extra observations
        matched_stats = {"med_delta": [], "mad": [], "iqr": [], "cv": [], "rho_area": [], "rho_frame": [], "cv_norm_frame": [], "rho_area_norm_frame": []}
        matched_ransac_stats = {"med_delta": [], "mad": [], "cv": []}
        unmatched_vals = []
        own_tracks_ge4 = 0

        for tid, tkeys in sorted(keys_by_track.items()):
            own_t = [k for k in tkeys if obs_records[m][k]["is_valid"]]
            if len(own_t) >= 4:
                own_tracks_ge4 += 1

            # Common-All-4 matched subset within this track
            c4_t = [k for k in tkeys if k in common_all4_set]
            if tid in common_track_ids_all4:
                vals = [obs_records[m][k]["p75"] for k in c4_t]
                norms = [obs_records[m][k]["frame_iqr_norm_p75"] for k in c4_t]
                areas = [obs_meta[k]["box_area"] for k in c4_t]
                frms = [obs_meta[k]["frame_idx"] for k in c4_t]
                tm = _track_metrics(vals, areas, frms)
                tm_n = _track_metrics(norms, areas, frms)
                matched_stats["med_delta"].append(tm["med_abs_delta"])
                matched_stats["mad"].append(tm["mad_disp"])
                matched_stats["iqr"].append(tm["iqr_disp"])
                if tm["cv"] is not None:
                    matched_stats["cv"].append(tm["cv"])
                if tm["rho_area"] is not None:
                    matched_stats["rho_area"].append(tm["rho_area"])
                if tm["rho_frame"] is not None:
                    matched_stats["rho_frame"].append(tm["rho_frame"])
                if tm_n["cv"] is not None:
                    matched_stats["cv_norm_frame"].append(tm_n["cv"])
                if tm_n["rho_area"] is not None:
                    matched_stats["rho_area_norm_frame"].append(tm_n["rho_area"])

                per_track_csv_rows.append({
                    "track_id": tid,
                    "population": "common_all4",
                    "method": m,
                    "obs_count": len(c4_t),
                    "p75_median": round(float(np.median(vals)), 4),
                    "median_abs_frame_delta": round(tm["med_abs_delta"], 4),
                    "mad_dispersion": round(tm["mad_disp"], 4),
                    "iqr_dispersion": round(tm["iqr_disp"], 4),
                    "cv_raw": round(tm["cv"], 4) if tm["cv"] is not None else None,
                    "cv_frame_iqr_norm": round(tm_n["cv"], 4) if tm_n["cv"] is not None else None,
                    "approach_spearman_rho_area": round(tm["rho_area"], 4) if tm["rho_area"] is not None else None,
                    "approach_spearman_rho_area_norm": round(tm_n["rho_area"], 4) if tm_n["rho_area"] is not None else None,
                })

            # Common RANSAC vs Optimized RANSAC subset
            cr_t = [k for k in tkeys if k in common_ransac_set]
            if tid in common_track_ids_ransac and m in ("original_ransac", "optimized_ransac"):
                vals_r = [obs_records[m][k]["p75"] for k in cr_t]
                areas_r = [obs_meta[k]["box_area"] for k in cr_t]
                frms_r = [obs_meta[k]["frame_idx"] for k in cr_t]
                tm_r = _track_metrics(vals_r, areas_r, frms_r)
                matched_ransac_stats["med_delta"].append(tm_r["med_abs_delta"])
                matched_ransac_stats["mad"].append(tm_r["mad_disp"])
                if tm_r["cv"] is not None:
                    matched_ransac_stats["cv"].append(tm_r["cv"])

            for k in own_t:
                if k not in common_all4_set:
                    unmatched_vals.append(obs_records[m][k]["p75"])

        method_temporal_summary[m] = {
            "total_observations_seen": total_obs,
            "valid_observations": own_valid_cnt,
            "valid_coverage_pct": round(100.0 * own_valid_cnt / max(1, total_obs), 2),
            "dropout_frequency_pct": round(100.0 * dropout_freq, 2),
            "own_tracks_ge4_valid": own_tracks_ge4,
            "unmatched_extra_valid_obs": len(unmatched_vals),
            "unmatched_p75_median": round(float(np.median(unmatched_vals)), 4) if unmatched_vals else None,
            "matched_common4_metrics": {
                "tracks_count": len(common_track_ids_all4),
                "obs_count": len(common_all4_keys),
                "median_abs_frame_delta_mean": round(float(np.mean(matched_stats["med_delta"])), 4) if matched_stats["med_delta"] else None,
                "mad_dispersion_mean": round(float(np.mean(matched_stats["mad"])), 4) if matched_stats["mad"] else None,
                "iqr_dispersion_mean": round(float(np.mean(matched_stats["iqr"])), 4) if matched_stats["iqr"] else None,
                "cv_mean": round(float(np.mean(matched_stats["cv"])), 4) if matched_stats["cv"] else None,
                "cv_median": round(float(np.median(matched_stats["cv"])), 4) if matched_stats["cv"] else None,
                "approach_rho_box_area_mean": round(float(np.mean(matched_stats["rho_area"])), 4) if matched_stats["rho_area"] else None,
                "approach_rho_frame_mean": round(float(np.mean(matched_stats["rho_frame"])), 4) if matched_stats["rho_frame"] else None,
                "frame_iqr_norm_cv_mean": round(float(np.mean(matched_stats["cv_norm_frame"])), 4) if matched_stats["cv_norm_frame"] else None,
                "frame_iqr_norm_rho_area_mean": round(float(np.mean(matched_stats["rho_area_norm_frame"])), 4) if matched_stats["rho_area_norm_frame"] else None,
            },
            "matched_ransac_pair_metrics": {
                "tracks_count": len(common_track_ids_ransac),
                "obs_count": len(common_ransac_keys),
                "median_abs_frame_delta_mean": round(float(np.mean(matched_ransac_stats["med_delta"])), 4) if matched_ransac_stats["med_delta"] else None,
                "mad_dispersion_mean": round(float(np.mean(matched_ransac_stats["mad"])), 4) if matched_ransac_stats["mad"] else None,
                "cv_mean": round(float(np.mean(matched_ransac_stats["cv"])), 4) if matched_ransac_stats["cv"] else None,
            } if m in ("original_ransac", "optimized_ransac") else None,
        }

    # Affine invariance verification on normalization schemes using cached video frame 0 obs
    affine_norm_check = {"scale": 5.0, "shift": 100.0, "max_abs_diff_scale_norm_p75": None, "max_abs_diff_frame_iqr_norm_p75": None}
    for fv in cached_video:
        if fv["depth_map"] is not None and fv["obs"]:
            d_raw = fv["depth_map"]
            d_aff = 5.0 * d_raw + 100.0
            iqr_raw = fv["frame_iqr"]
            iqr_aff = float(np.percentile(d_aff, 75.0) - np.percentile(d_aff, 25.0))
            bx = fv["obs"][0]["box"]
            nb = fv["obs"][0]["nbrs"]
            o_raw = extract_optimized_relative_depth_for_box(d_raw, bx, neighbor_boxes=nb, img_bgr=fv["img_bgr"], cfg=opt_cfg)
            o_aff = extract_optimized_relative_depth_for_box(d_aff, bx, neighbor_boxes=nb, img_bgr=fv["img_bgr"], cfg=opt_cfg)
            if o_raw["is_valid"] and o_aff["is_valid"]:
                sn_d = abs(o_aff["relative_cavity_stats"]["scale_normalized_p75"] - o_raw["relative_cavity_stats"]["scale_normalized_p75"])
                fn_d = abs((o_aff["relative_cavity_stats"]["p75_residual"] / iqr_aff) - (o_raw["relative_cavity_stats"]["p75_residual"] / iqr_raw))
                affine_norm_check["max_abs_diff_scale_norm_p75"] = round(sn_d, 5)
                affine_norm_check["max_abs_diff_frame_iqr_norm_p75"] = round(fn_d, 5)
                break

    # Task 7 on real matched tracks (`optimized_ransac` stream over ByteTrack tracks)
    flt_cfg = TemporalFilterConfig(alpha=0.35, rolling_window=5, max_stale_frames=3, max_missed_frames_reset=8)
    real_filter_summary: Dict[str, Any] = {}
    sample_track_traces: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    # Pick longest active track ID for plotting
    longest_tid = max(keys_by_track.items(), key=lambda kv: len(kv[1]))[0] if keys_by_track else None

    for fmode in FILTER_MODES:
        flt = ValidityAwareTrackFilter(mode=fmode, cfg=flt_cfg)
        all_step_deltas = []
        valid_only_deltas = []
        stale_tot = 0
        expired_tot = 0
        emitted_tot = 0
        trace_by_tid: Dict[int, List[Dict[str, Any]]] = {}

        for fv in cached_video:
            f_idx = fv["frame_idx"]
            active_tids = [it["track_id"] for it in fv["obs"]]
            flt.step_missing_tracks(active_tids, f_idx)
            for it in fv["obs"]:
                tid = it["track_id"]
                rec = obs_records["optimized_ransac"][(f_idx, tid)]
                st_out = flt.update(tid, rec["p75"], rec["is_valid"], f_idx, fv["ts"])
                trace_by_tid.setdefault(tid, []).append(st_out)
                if st_out["filtered_value"] is not None:
                    emitted_tot += 1
                if st_out["is_stale"] and not st_out["is_expired"]:
                    stale_tot += 1
                if st_out["is_expired"]:
                    expired_tot += 1
                if tid == longest_tid:
                    sample_track_traces.setdefault(tid, {})[fmode] = trace_by_tid[tid]

        for tid, tr in trace_by_tid.items():
            vals_e = [t["filtered_value"] for t in tr if t["filtered_value"] is not None]
            vals_v = [t["filtered_value"] for t in tr if t["filtered_value"] is not None and t["is_valid_input"]]
            if len(vals_e) >= 2:
                all_step_deltas.extend(np.abs(np.diff(vals_e)).tolist())
            if len(vals_v) >= 2:
                valid_only_deltas.extend(np.abs(np.diff(vals_v)).tolist())

        real_filter_summary[fmode] = {
            "total_track_steps": total_obs,
            "emitted_non_none_steps": emitted_tot,
            "stale_held_steps": stale_tot,
            "expired_none_steps": expired_tot,
            "mean_abs_frame_delta_all_emitted": round(float(np.mean(all_step_deltas)), 4) if all_step_deltas else None,
            "median_abs_frame_delta_all_emitted": round(float(np.median(all_step_deltas)), 4) if all_step_deltas else None,
            "mean_abs_frame_delta_valid_steps": round(float(np.mean(valid_only_deltas)), 4) if valid_only_deltas else None,
        }

    # Write CSV exports
    obs_csv = os.path.join(BENCH_DIR, "matched_track_temporal_observations.csv")
    with open(obs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_obs_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_obs_rows)

    track_csv = os.path.join(BENCH_DIR, "matched_track_temporal_per_track.csv")
    with open(track_csv, "w", newline="", encoding="utf-8") as f:
        if per_track_csv_rows:
            w = csv.DictWriter(f, fieldnames=list(per_track_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(per_track_csv_rows)

    return {
        "video": "uploads/upload_1794847200_ruralRoad_potHoles.mp4 (frames 0..119)",
        "total_unique_tracks": len(track_ids_seen),
        "total_track_observations": total_obs,
        "common_valid_all4_tracks_ge4_obs": len(common_track_ids_all4),
        "common_valid_all4_observations": len(common_all4_keys),
        "common_valid_ransac_pair_tracks_ge4_obs": len(common_track_ids_ransac),
        "common_valid_ransac_pair_observations": len(common_ransac_keys),
        "affine_normalization_invariance_check": affine_norm_check,
        "methods": method_temporal_summary,
        "temporal_filters_on_real_optimized_tracks": real_filter_summary,
        "sample_plot_track_id": longest_tid,
        "sample_track_traces": sample_track_traces,
    }


def render_plots(
    dashcam_res: Dict[str, Any],
    ablation_res: Dict[str, Any],
    temporal_res: Dict[str, Any],
    cached_frames: List[Dict[str, Any]],
    opt_cfg: OptimizedDepthConfig,
) -> List[str]:
    """Generate benchmark comparison, temporal trajectory, and side-by-side failure/success plots."""
    paths = []

    # 1. Performance & Budget Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=130)
    m_keys = ["existing_annular", "ols_plane", "original_ransac", "optimized_ransac"]
    m_means = [dashcam_res["methods"][m]["latency_ms"]["mean_ms"] for m in m_keys]
    m_p95s = [dashcam_res["methods"][m]["latency_ms"]["p95_ms"] for m in m_keys]
    x = np.arange(len(m_keys))
    axes[0].bar(x - 0.18, m_means, width=0.35, label="Mean (ms)", color="#2563eb")
    axes[0].bar(x + 0.18, m_p95s, width=0.35, label="P95 (ms)", color="#f59e0b")
    axes[0].axhline(5.0, color="#16a34a", linestyle="--", linewidth=1.2, label="4-5 ms Target")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Existing\nAnnular", "OLS\nPlane", "Orig\nRANSAC", "Opt RANSAC\n(Strat 512)"], fontsize=8.5)
    axes[0].set_ylabel("Extraction Latency (ms / region)")
    axes[0].set_title("Method Extraction Latency (RTX 2050 Host CPU)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Budget vs Latency & Median P75 diff for stratified_quadrant
    budgets = [256, 512, 1024, 2048]
    b_labels = ["256", "512", "1024", "2048", "Full"]
    b_keys = [f"stratified_quadrant_{b}" for b in budgets] + ["full_budget_none"]
    b_ms = [ablation_res[k]["latency_ms"]["mean_ms"] for k in b_keys]
    b_diffs = [ablation_res[k]["matched_p75_abs_diff_vs_orig_median"] or 0.0 for k in b_keys]
    ax2 = axes[1]
    ax2_r = ax2.twinx()
    l1 = ax2.plot(b_labels, b_ms, "o-", color="#2563eb", linewidth=2, label="Mean Latency (ms)")
    l2 = ax2_r.plot(b_labels, b_diffs, "s--", color="#dc2626", linewidth=2, label="Median |ΔP75| vs Orig")
    ax2.set_ylabel("Latency (ms)", color="#2563eb")
    ax2_r.set_ylabel("Median |ΔP75| vs Orig (rel units)", color="#dc2626")
    ax2.set_title("Stratified Point Budget Trade-off")
    ax2.grid(alpha=0.3)
    lines_comb = l1 + l2
    ax2.legend(lines_comb, [l.get_label() for l in lines_comb], fontsize=8, loc="upper left")

    # Panel 3: Latency scaling vs available ring pixels
    scale_buckets = ["lt_2k", "2k_to_10k", "10k_to_30k", "gt_30k"]
    s_labels = ["<2k px", "2k-10k", "10k-30k", ">30k"]
    orig_s = [dashcam_res["sample_count_scaling"]["original_ransac"].get(b, {}).get("mean_ms", 0) or 0 for b in scale_buckets]
    opt_s = [dashcam_res["sample_count_scaling"]["optimized_ransac"].get(b, {}).get("mean_ms", 0) or 0 for b in scale_buckets]
    xs = np.arange(len(s_labels))
    axes[2].plot(xs, orig_s, "o-", color="#ef4444", linewidth=2, label="Original RANSAC")
    axes[2].plot(xs, opt_s, "o-", color="#16a34a", linewidth=2, label="Optimized RANSAC (512)")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(s_labels, fontsize=8.5)
    axes[2].set_ylabel("Mean Extraction Latency (ms)")
    axes[2].set_title("Latency vs Available Ring Pixel Count")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("RoadVision AI Phase 5B — Extraction Optimization & Scaling Benchmarks", fontsize=12, fontweight="bold")
    fig.subplots_adjust(wspace=0.45)
    p_perf = os.path.join(VIZ_DIR, "perf_and_budget_comparison.png")
    plt.savefig(p_perf, bbox_inches="tight")
    plt.close(fig)
    paths.append(p_perf)

    # 2. Temporal Filter Comparison Plot (Real ByteTrack Longest Track + Controlled Dropout Synthetic Trace)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8), dpi=130)
    tid = temporal_res["sample_plot_track_id"]
    if tid is not None and tid in temporal_res["sample_track_traces"]:
        tr_dict = temporal_res["sample_track_traces"][tid]
        colors = {"raw": "#64748b", "legacy_ema_zero_clamp": "#dc2626", "validity_aware_ema": "#2563eb", "rolling_median_validity_ema": "#16a34a"}
        for fm in FILTER_MODES:
            tr = tr_dict[fm]
            fs = [t["frame_idx"] for t in tr]
            vs = [t["filtered_value"] if t["filtered_value"] is not None else np.nan for t in tr]
            style = "o:" if fm == "raw" else "-"
            axes[0].plot(fs, vs, style, color=colors[fm], linewidth=1.8, markersize=3.5, label=fm)
        axes[0].set_title(f"Real Track #{tid} (ruralRoad_potHoles.mp4) — Relative P75")
        axes[0].set_xlabel("Frame Index")
        axes[0].set_ylabel("Relative Inverse-Depth P75 (units)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

    # Controlled `invalid_dropouts_with_glitches` trace
    seq = build_synthetic_temporal_sequences()["invalid_dropouts_with_glitches"]
    flt_cfg = TemporalFilterConfig(alpha=0.35, rolling_window=5, max_stale_frames=3, max_missed_frames_reset=8)
    colors = {"raw": "#64748b", "legacy_ema_zero_clamp": "#dc2626", "validity_aware_ema": "#2563eb", "rolling_median_validity_ema": "#16a34a"}
    for fm in FILTER_MODES:
        flt = ValidityAwareTrackFilter(mode=fm, cfg=flt_cfg)
        outs = [flt.update(1, s["obs"], s["valid"], s["frame"], s["ts"]) for s in seq]
        fs = [o["frame_idx"] for o in outs]
        vs = [o["filtered_value"] if o["filtered_value"] is not None else np.nan for o in outs]
        axes[1].plot(fs, vs, "-" if fm != "raw" else "o:", color=colors[fm], linewidth=1.8, markersize=3.5, label=fm)
    axes[1].axhline(12.0, color="black", linestyle="--", linewidth=1.0, label="True Relative Level (12.0)")
    axes[1].set_title("Controlled Sequence: Invalid Dropouts + Corrupted Glitches")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Filtered Relative Value")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Phase 5B — Validity-Aware Temporal Filter Behavior (Relative Units Only)", fontsize=12, fontweight="bold")
    p_tmp = os.path.join(VIZ_DIR, "matched_track_temporal_filters.png")
    plt.savefig(p_tmp, bbox_inches="tight")
    plt.close(fig)
    paths.append(p_tmp)

    # 3. Representative Side-by-Side Original vs Optimized RANSAC Visual Cases (Success, Water Invalid, Disagreement/Polarity)
    base_cfg = RobustDepthConfig()
    cases_found = {"valid": None, "invalid_water": None, "invalid_polarity": None}
    for cf in cached_frames:
        if not cf["boxes"] or cf["depth_map"] is None:
            continue
        for idx, box in enumerate(cf["boxes"]):
            nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
            o_orig = extract_relative_depth_for_box(cf["depth_map"], tuple(box), method="ransac_plane", neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=base_cfg, return_arrays=True)
            o_opt = extract_optimized_relative_depth_for_box(cf["depth_map"], tuple(box), neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=opt_cfg, return_arrays=True)
            if o_orig["is_valid"] and o_opt["is_valid"] and o_opt["ref_diagnostics"]["neighbor_excluded_px"] > 0 and cases_found["valid"] is None:
                cases_found["valid"] = (cf, idx, o_orig, o_opt)
            elif ("suspected_water_reflection" in o_opt["failure_reasons"]) and cases_found["invalid_water"] is None:
                cases_found["invalid_water"] = (cf, idx, o_orig, o_opt)
            elif ("unexpected_residual_polarity" in o_opt["failure_reasons"]) and cases_found["invalid_polarity"] is None:
                cases_found["invalid_polarity"] = (cf, idx, o_orig, o_opt)

    for tag, item in cases_found.items():
        if item is None:
            continue
        cf, idx, o_orig, o_opt = item
        rgb = cv2.cvtColor(cf["img_bgr"], cv2.COLOR_BGR2RGB)
        reg_o = o_opt["arrays"]["region"]
        rx1, ry1, rx2, ry2 = reg_o["outer_xyxy"]
        outer_rgb = rgb[ry1:ry2, rx1:rx2]

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=130)
        axes[0].imshow(outer_rgb)
        bx1, by1, bx2, by2 = o_opt["box_xyxy"]
        rect = plt.Rectangle((bx1 - rx1, by1 - ry1), bx2 - bx1, by2 - by1, fill=False, edgecolor="#22c55e", linewidth=2)
        axes[0].add_patch(rect)
        axes[0].set_title(f"ROI ({cf['entry']['frame_id']} r#{idx}) | Ring={reg_o['available_ref_samples']} px")
        axes[0].axis("off")

        # Stratified sampled points & inliers
        samp_vis = (outer_rgb * 0.45).astype(np.uint8)
        ys_s, xs_s = reg_o["ys_samp"], reg_o["xs_samp"]
        inl = o_opt["arrays"]["fit_out"]["inlier_mask"]
        for yy, xx, is_i in zip(ys_s, xs_s, inl):
            cv2.circle(samp_vis, (int(xx), int(yy)), 2, (34, 197, 94) if is_i else (239, 68, 68), -1)
        axes[1].imshow(samp_vis)
        axes[1].set_title(f"Opt Sampled ({len(ys_s)} pts, inl={o_opt['ref_diagnostics']['inlier_ratio']*100:.1f}%, iters={o_opt['ref_diagnostics']['iters_evaluated']})")
        axes[1].axis("off")

        axes[2].axis("off")
        st_b, st_o = o_orig["relative_cavity_stats"], o_opt["relative_cavity_stats"]
        txt = [
            f"Case Category: {tag.upper()}",
            f"Orig Valid={o_orig['is_valid']} | Reasons={o_orig['failure_reasons'] or 'None'}",
            f"Opt  Valid={o_opt['is_valid']} | Reasons={o_opt['failure_reasons'] or 'None'}",
            f"Orig P75={st_b['raw_diagnostic_p75']:.3f} | MAD={o_orig['ref_diagnostics']['ref_fit_mad']:.3f}",
            f"Opt  P75={st_o['raw_diagnostic_p75']:.3f} | MAD={o_opt['ref_diagnostics']['ref_fit_mad']:.3f}",
            f"Orig Time={o_orig['timing_ms']['extraction_total_ms']:.2f} ms",
            f"Opt  Time={o_opt['timing_ms']['extraction_total_ms']:.2f} ms",
            "(Relative MiDaS inverse-depth units ONLY)",
        ]
        axes[2].text(0.05, 0.90, "\n".join(txt), va="top", family="monospace", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#16a34a" if o_opt["is_valid"] else "#dc2626", linewidth=2))
        p_case = os.path.join(VIZ_DIR, f"case_compare_{tag}.png")
        plt.savefig(p_case, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_case)

    return paths


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = YOLO(DEFAULT_DET_WEIGHTS)
    midas = MiDaSDepthRunner(device=device_str)
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    _ = det_model.predict(dummy, imgsz=640, conf=0.25, iou=0.70, device=0 if device_str == "cuda" else "cpu", verbose=False)
    midas.warmup(360, 640)

    opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    base_cfg = RobustDepthConfig()

    # Cache 36 dev frames identically once
    catalog = build_dev_frame_catalog()
    cached_frames = []
    det_times, midas_times = [], []
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)

    for loop_i in range(3):
        for entry in catalog:
            img_bgr, ts = load_frame(entry)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            td0 = time.perf_counter()
            r = det_model.predict(img_bgr, imgsz=640, conf=0.25, iou=0.70, device=0 if device_str == "cuda" else "cpu", verbose=False)[0]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            det_ms = (time.perf_counter() - td0) * 1000.0
            det_times.append(det_ms)
            boxes = r.boxes.xyxy.cpu().numpy().tolist() if (r.boxes is not None and len(r.boxes) > 0) else []
            confs = r.boxes.conf.cpu().numpy().tolist() if (r.boxes is not None and len(r.boxes) > 0) else []
            if boxes:
                dmap, m_ms = midas.predict(img_bgr)
                midas_times.append(m_ms)
            else:
                dmap, m_ms = None, 0.0
                midas_times.append(0.0)
            if loop_i == 0:
                cached_frames.append({"entry": entry, "ts": ts, "img_bgr": img_bgr, "boxes": boxes, "confs": confs, "depth_map": dmap})

    peak_gpu_alloc = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0
    peak_gpu_res = round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0

    # Run all 4 comparison methods across 3 CPU passes + measure tracemalloc peak CPU memory
    methods = ("existing_annular", "ols_plane", "original_ransac", "optimized_ransac")
    dash_outputs: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {m: {} for m in methods}
    stage_timings = {m: {"annular": [], "ref_fit": [], "residual": [], "qa": [], "total": []} for m in methods}
    sub_opt_timings = {"local_sigma": [], "ransac_loop": [], "refit_plane": []}
    prop_scaling = {m: {"1": [], "2": [], "3": [], "4+": []} for m in methods}
    sample_scaling: Dict[str, Dict[str, List[float]]] = {
        m: {"lt_2k": [], "2k_to_10k": [], "10k_to_30k": [], "gt_30k": []} for m in ("original_ransac", "optimized_ransac")
    }
    cpu_peak_kb = {}

    for m in methods:
        for loop_i in range(3):
            for cf in cached_frames:
                if not cf["boxes"] or cf["depth_map"] is None:
                    continue
                ft_ms = 0.0
                for idx, box in enumerate(cf["boxes"]):
                    nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
                    if m == "optimized_ransac":
                        out = extract_optimized_relative_depth_for_box(
                            cf["depth_map"], tuple(box), neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=opt_cfg, return_arrays=(loop_i == 0)
                        )
                        if loop_i == 0:
                            st_sub = out["arrays"]["fit_out"]["sub_timing_ms"]
                            sub_opt_timings["local_sigma"].append(st_sub["local_sigma_ms"])
                            sub_opt_timings["ransac_loop"].append(st_sub["ransac_loop_ms"])
                            sub_opt_timings["refit_plane"].append(st_sub["refit_plane_ms"])
                    else:
                        rm = "ransac_plane" if m == "original_ransac" else m
                        out = extract_relative_depth_for_box(
                            cf["depth_map"], tuple(box), method=rm, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=base_cfg
                        )
                    tm = out["timing_ms"]
                    stage_timings[m]["annular"].append(tm["annular_region_ms"])
                    stage_timings[m]["ref_fit"].append(tm["ref_fit_ms"])
                    stage_timings[m]["residual"].append(tm["residual_compute_ms"])
                    stage_timings[m]["qa"].append(tm["quality_eval_ms"])
                    stage_timings[m]["total"].append(tm["extraction_total_ms"])
                    ft_ms += tm["extraction_total_ms"]

                    if m in ("original_ransac", "optimized_ransac"):
                        n_avail = out["ref_diagnostics"].get("available_ref_samples", out["ref_diagnostics"]["valid_ref_samples"])
                        sb = "lt_2k" if n_avail < 2000 else ("2k_to_10k" if n_avail < 10000 else ("10k_to_30k" if n_avail < 30000 else "gt_30k"))
                        sample_scaling[m][sb].append(tm["extraction_total_ms"])

                    if loop_i == 0:
                        dash_outputs[m][(cf["entry"]["frame_id"], idx)] = out
                bkey = "4+" if len(cf["boxes"]) >= 4 else str(len(cf["boxes"]))
                prop_scaling[m][bkey].append(ft_ms)

        # Isolated single memory-tracing pass OUTSIDE timing loops so allocation hooks never inflate latency
        tracemalloc.start()
        for cf in cached_frames:
            if not cf["boxes"] or cf["depth_map"] is None:
                continue
            for idx, box in enumerate(cf["boxes"]):
                nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]
                if m == "optimized_ransac":
                    _ = extract_optimized_relative_depth_for_box(cf["depth_map"], tuple(box), neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=opt_cfg)
                else:
                    rm = "ransac_plane" if m == "original_ransac" else m
                    _ = extract_relative_depth_for_box(cf["depth_map"], tuple(box), method=rm, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=base_cfg)
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_peak_kb[m] = round(peak_b / 1024.0, 2)

    all_region_keys = list(dash_outputs["optimized_ransac"].keys())
    common_valid_all4 = [k for k in all_region_keys if all(dash_outputs[m][k]["is_valid"] for m in methods)]
    common_valid_ransac_pair = [k for k in all_region_keys if dash_outputs["original_ransac"][k]["is_valid"] and dash_outputs["optimized_ransac"][k]["is_valid"]]

    # Sensitivity & Contamination on cached dev frames
    dash_method_stats = {}
    orig_mean_ms = compute_latency_stats(stage_timings["original_ransac"]["total"])["mean_ms"]

    per_region_csv = []
    for m in methods:
        own_valid_keys = [k for k in all_region_keys if dash_outputs[m][k]["is_valid"]]
        fails = Counter()
        for k in all_region_keys:
            if not dash_outputs[m][k]["is_valid"]:
                for r in dash_outputs[m][k]["failure_reasons"]:
                    fails[r] += 1

        # Perturbation (±8%) & curb contamination sensitivity
        rng_p = np.random.default_rng(42)
        p_abs, p_rel, p_ret = [], [], 0
        c_abs, c_rel, c_ret, c_tot = [], [], 0, 0
        for cf in cached_frames:
            if not cf["boxes"] or cf["depth_map"] is None:
                continue
            H, W = cf["depth_map"].shape[:2]
            for idx, box in enumerate(cf["boxes"]):
                k = (cf["entry"]["frame_id"], idx)
                ob = dash_outputs[m][k]
                bw, bh = box[2] - box[0], box[3] - box[1]
                dx1, dx2 = rng_p.uniform(-0.08 * bw, 0.08 * bw, size=2)
                dy1, dy2 = rng_p.uniform(-0.08 * bh, 0.08 * bh, size=2)
                pbox = (max(0, box[0] + dx1), max(0, box[1] + dy1), min(W, box[2] + dx2), min(H, box[3] + dy2))
                nbrs = [tuple(cf["boxes"][j]) for j in range(len(cf["boxes"])) if j != idx]

                if m == "optimized_ransac":
                    op = extract_optimized_relative_depth_for_box(cf["depth_map"], pbox, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=opt_cfg)
                else:
                    rm = "ransac_plane" if m == "original_ransac" else m
                    op = extract_relative_depth_for_box(cf["depth_map"], pbox, method=rm, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=base_cfg)

                if ob["is_valid"] and op["is_valid"]:
                    p_ret += 1
                    b0 = ob["relative_cavity_stats"]["p75_residual"]
                    b1 = op["relative_cavity_stats"]["p75_residual"]
                    p_abs.append(abs(b1 - b0))
                    p_rel.append(_guarded_rel_diff_pct(b1, b0))

                if ob["is_valid"]:
                    c_tot += 1
                    d_cor = cf["depth_map"].copy()
                    ox1, oy1, ox2, oy2 = ob["outer_xyxy"]
                    bx1, by1, bx2, by2 = ob["box_xyxy"]
                    top_h = max(0, by1 - oy1)
                    if top_h >= 4:
                        d_cor[oy1 : oy1 + max(2, int(0.8 * top_h)), ox1:ox2] += float(np.nanstd(d_cor) * 4.0 + 50.0)
                    if m == "optimized_ransac":
                        oc = extract_optimized_relative_depth_for_box(d_cor, tuple(box), neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=opt_cfg)
                    else:
                        rm = "ransac_plane" if m == "original_ransac" else m
                        oc = extract_relative_depth_for_box(d_cor, tuple(box), method=rm, neighbor_boxes=nbrs, img_bgr=cf["img_bgr"], cfg=base_cfg)
                    if oc["is_valid"]:
                        c_ret += 1
                    b0 = ob["relative_cavity_stats"]["p75_residual"]
                    b_c = oc["relative_cavity_stats"]["raw_diagnostic_p75"]
                    c_abs.append(abs(b_c - b0))
                    c_rel.append(_guarded_rel_diff_pct(b_c, b0))

                per_region_csv.append({
                    "frame_id": k[0],
                    "region_idx": k[1],
                    "method": m,
                    "is_valid": ob["is_valid"],
                    "in_common_all4": k in common_valid_all4,
                    "in_common_ransac_pair": k in common_valid_ransac_pair,
                    "failure_reasons": "|".join(ob["failure_reasons"]),
                    "available_ref_samples": ob["ref_diagnostics"].get("available_ref_samples", ob["ref_diagnostics"]["valid_ref_samples"]),
                    "used_ref_samples": ob["ref_diagnostics"]["used_ref_samples"],
                    "inlier_ratio": ob["ref_diagnostics"]["inlier_ratio"],
                    "ref_fit_mad": ob["ref_diagnostics"]["ref_fit_mad"],
                    "p75_residual_valid": ob["relative_cavity_stats"]["p75_residual"],
                    "raw_diag_p75": ob["relative_cavity_stats"]["raw_diagnostic_p75"],
                    "extraction_ms": ob["timing_ms"]["extraction_total_ms"],
                })

        lat = compute_latency_stats(stage_timings[m]["total"])
        dash_method_stats[m] = {
            "total_regions": len(all_region_keys),
            "admitted_population": {
                "valid_count": len(own_valid_keys),
                "valid_pct": round(100.0 * len(own_valid_keys) / max(1, len(all_region_keys)), 2),
                "ref_mad_stats": _summarize_vals([dash_outputs[m][k]["ref_diagnostics"]["ref_fit_mad"] for k in own_valid_keys]),
                "p75_stats": _summarize_vals([dash_outputs[m][k]["relative_cavity_stats"]["p75_residual"] for k in own_valid_keys]),
            },
            "common_valid_all4_population": {
                "count": len(common_valid_all4),
                "ref_mad_stats": _summarize_vals([dash_outputs[m][k]["ref_diagnostics"]["ref_fit_mad"] for k in common_valid_all4]),
                "p75_stats": _summarize_vals([dash_outputs[m][k]["relative_cavity_stats"]["p75_residual"] for k in common_valid_all4]),
                "p75_abs_diff_vs_orig_ransac": _summarize_vals([abs(dash_outputs[m][k]["relative_cavity_stats"]["p75_residual"] - dash_outputs["original_ransac"][k]["relative_cavity_stats"]["p75_residual"]) for k in common_valid_all4]),
            },
            "common_valid_ransac_pair_population": {
                "count": len(common_valid_ransac_pair),
                "ref_mad_stats": _summarize_vals([dash_outputs[m][k]["ref_diagnostics"]["ref_fit_mad"] for k in common_valid_ransac_pair]),
                "p75_abs_diff_vs_orig_ransac": _summarize_vals([abs(dash_outputs[m][k]["relative_cavity_stats"]["p75_residual"] - dash_outputs["original_ransac"][k]["relative_cavity_stats"]["p75_residual"]) for k in common_valid_ransac_pair]),
                "p75_rel_diff_pct_vs_orig_ransac": _summarize_vals([_guarded_rel_diff_pct(dash_outputs[m][k]["relative_cavity_stats"]["p75_residual"], dash_outputs["original_ransac"][k]["relative_cavity_stats"]["p75_residual"]) for k in common_valid_ransac_pair]),
            } if m in ("original_ransac", "optimized_ransac") else None,
            "failure_reason_counts": dict(fails),
            "box_perturbation_8pct": {
                "retained_valid": f"{p_ret}/{len(own_valid_keys)}",
                "abs_p75_shift_median": round(float(np.median(p_abs)), 4) if p_abs else None,
                "rel_p75_shift_median_pct": round(float(np.median(p_rel)), 2) if p_rel else None,
            },
            "curb_contamination": {
                "retained_valid": f"{c_ret}/{c_tot}",
                "abs_p75_err_median": round(float(np.median(c_abs)), 4) if c_abs else None,
                "rel_p75_err_median_pct": round(float(np.median(c_rel)), 2) if c_rel else None,
            },
            "latency_ms": lat,
            "speedup_vs_orig_ransac_mean": round(orig_mean_ms / max(1e-6, lat["mean_ms"]), 2),
            "speedup_vs_orig_ransac_median": round(compute_latency_stats(stage_timings["original_ransac"]["total"])["median_ms"] / max(1e-6, lat["median_ms"]), 2),
            "stage_mean_ms": {sk: round(float(np.mean(sv)), 4) for sk, sv in stage_timings[m].items()},
            "proposal_scaling_ms": {bk: compute_latency_stats(bv) for bk, bv in prop_scaling[m].items() if bv},
            "cpu_peak_traced_kb": cpu_peak_kb[m],
        }

    dash_csv_path = os.path.join(BENCH_DIR, "optimized_relative_depth_per_region.csv")
    with open(dash_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_region_csv[0].keys()))
        w.writeheader()
        w.writerows(per_region_csv)

    dashcam_summary = {
        "common_valid_all4_count": len(common_valid_all4),
        "common_valid_ransac_pair_count": len(common_valid_ransac_pair),
        "methods": dash_method_stats,
        "sample_count_scaling": {
            m: {bk: compute_latency_stats(bv) for bk, bv in bdict.items() if bv}
            for m, bdict in sample_scaling.items()
        },
        "optimized_ransac_sub_stages_ms": {
            k: round(float(np.mean(v)), 4) for k, v in sub_opt_timings.items()
        },
    }

    sampling_ablation = run_sampling_ablation_dashcam(cached_frames, dash_outputs["original_ransac"])
    synth_regression = run_synthetic_regression(opt_cfg)
    matched_temporal = run_matched_track_evaluation(det_model, midas, opt_cfg)
    synth_temporal_filters = evaluate_temporal_filter_suite()

    viz_paths = render_plots(
        dashcam_summary,
        sampling_ablation,
        matched_temporal,
        cached_frames,
        opt_cfg,
    )

    # Strip large sample_track_traces from JSON before saving
    matched_temporal_clean = {k: v for k, v in matched_temporal.items() if k != "sample_track_traces"}

    final_json = {
        "experiment": "Phase 5B Optimized Adaptive RANSAC & Matched-Track Relative Depth Benchmark",
        "backbone_rtx2050": {
            "detector_640_ms": compute_latency_stats(det_times),
            "midas_small_active_ms": compute_latency_stats([t for t in midas_times if t > 0]),
            "peak_gpu_allocated_mb": peak_gpu_alloc,
            "peak_gpu_reserved_mb": peak_gpu_res,
        },
        "sampling_ablation_dashcam": sampling_ablation,
        "synthetic_regression_48": synth_regression,
        "dashcam_matched_comparison_36f": dashcam_summary,
        "matched_track_temporal_rural_video": matched_temporal_clean,
        "synthetic_temporal_filter_suite": synth_temporal_filters,
        "visualization_paths": viz_paths,
    }

    out_json = os.path.join(BENCH_DIR, "optimized_relative_depth_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    print("=== PHASE 5B BENCHMARK COMPLETE ===")
    print(f"Orig RANSAC mean={dash_method_stats['original_ransac']['latency_ms']['mean_ms']:.2f} ms (med={dash_method_stats['original_ransac']['latency_ms']['median_ms']:.2f}, p95={dash_method_stats['original_ransac']['latency_ms']['p95_ms']:.2f})")
    print(f"Opt  RANSAC mean={dash_method_stats['optimized_ransac']['latency_ms']['mean_ms']:.2f} ms (med={dash_method_stats['optimized_ransac']['latency_ms']['median_ms']:.2f}, p95={dash_method_stats['optimized_ransac']['latency_ms']['p95_ms']:.2f}) | Speedup={dash_method_stats['optimized_ransac']['speedup_vs_orig_ransac_mean']}x")
    print(f"Matched common_all4 regions={len(common_valid_all4)}, common_ransac_pair={len(common_valid_ransac_pair)}")
    print(f"Synthetic 48 regression: Orig={synth_regression['summary']['original_ransac_full']['all_48']} | Opt512={synth_regression['summary']['optimized_ransac_strat_512']['all_48']}")


if __name__ == "__main__":
    main()
