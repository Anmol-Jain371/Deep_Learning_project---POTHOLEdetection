"""
Phase 13B: Corrected Reference-Plane Estimation Ablation & Relative Geometry Evaluation.

Fixes ISS-008: eliminates duplicate Method D call so every method executes ONCE per timed trial
across equivalent full per-region processing scope, plus separate sub-stage breakdown for
reference-region construction, plane fitting, and cavity extraction.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np

from experiments.core_aware_cavity_extraction import extract_cavity_with_strategy
from experiments.optimized_relative_depth import (
    OptimizedDepthConfig,
    build_and_sample_annular_region,
    fit_ransac_optimized,
)
from experiments.robust_relative_depth import RobustDepthConfig, extract_relative_depth_for_box
from roadvision_engine import AnnularDonutDepthExtractor
from roadvision_v2.geometry import build_relative_surface_snapshot, compute_cross_section_profiles


def _make_synthetic_case(
    scenario: str,
    idx: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    H, W = 180, 240
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if scenario == "flat":
        a, b, c = 0.0, 0.0, 2.5
        cav_amp = 0.22
    elif scenario == "sloped":
        a, b, c = 0.008, -0.004, 2.2
        cav_amp = 0.24
    elif scenario == "raised":
        a, b, c = 0.003, 0.001, 2.0
        cav_amp = -0.20  # Protrusion / speedbump (must be rejected)
    elif scenario == "boundary_truncated":
        a, b, c = 0.004, -0.002, 2.1
        cav_amp = 0.20
    else:
        a, b, c = 0.006, -0.003, 2.3
        cav_amp = 0.24

    gt_plane = a * yy + b * xx + c
    depth_map = gt_plane.copy()

    if scenario == "boundary_truncated":
        box = (188.0, 128.0, 238.0, 178.0)
        cx, cy = 213.0, 153.0
    else:
        box = (80.0, 65.0, 160.0, 125.0)
        cx, cy = 120.0, 95.0

    ell = ((xx - cx) / 28.0) ** 2 + ((yy - cy) / 20.0) ** 2
    if cav_amp != 0.0:
        depth_map[ell <= 1.0] -= cav_amp * (1.0 - ell[ell <= 1.0])

    if scenario == "structured_noise":
        depth_map += 0.015 * np.sin(xx * 0.22) + rng.normal(0.0, 0.006, size=(H, W))
    else:
        depth_map += rng.normal(0.0, 0.003, size=(H, W))

    outlier_frac = {"outliers_10pct": 0.10, "outliers_25pct": 0.25, "outliers_40pct": 0.40}.get(scenario, 0.0)
    if outlier_frac > 0.0:
        ring_cand = (ell > 1.05) & (ell < 3.2)
        ys_r, xs_r = np.where(ring_cand)
        k = int(round(len(ys_r) * outlier_frac))
        if k > 0:
            pick = rng.choice(len(ys_r), size=k, replace=False)
            depth_map[ys_r[pick], xs_r[pick]] += rng.choice([-0.8, 0.9], size=k)

    neighbors = []
    if scenario == "low_support":
        neighbors = [
            (30.0, 20.0, 210.0, 82.0),
            (30.0, 82.0, 108.0, 170.0),
            (132.0, 82.0, 210.0, 170.0),
        ]

    expect_valid = scenario not in ("raised", "low_support")
    return {
        "scenario": scenario,
        "case_id": f"{scenario}_{idx}",
        "depth_map": depth_map.astype(np.float32),
        "gt_plane": gt_plane,
        "box": box,
        "neighbors": neighbors,
        "expect_valid": expect_valid,
    }


def _evaluate_method_d_once(
    dmap: np.ndarray,
    box: Any,
    neighbors: Any,
    opt_cfg: OptimizedDepthConfig,
) -> Dict[str, Any]:
    """Single-call Method D execution wrapper returning arrays without duplicate plane fit."""
    return extract_cavity_with_strategy(
        dmap,
        box,
        strategy="elliptical_core",
        opt_cfg=opt_cfg,
        neighbor_boxes=neighbors,
        return_arrays=True,
    )


def run_geometry_ablation(
    *,
    smoke: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    scenarios = [
        "flat",
        "sloped",
        "structured_noise",
        "outliers_10pct",
        "outliers_25pct",
        "outliers_40pct",
        "raised",
        "low_support",
        "boundary_truncated",
    ]
    repeats = 2 if smoke else 5
    timing_trials = 1 if smoke else 3
    cases: List[Dict[str, Any]] = []
    for sc in scenarios:
        for r in range(repeats):
            cases.append(_make_synthetic_case(sc, r, rng))

    rob_cfg = RobustDepthConfig(seed=seed)
    opt_cfg = OptimizedDepthConfig(seed=seed)

    # Consistent untimed warm-up on the first case
    if cases:
        w = cases[0]
        wbx = [int(v) for v in w["box"]]
        for _ in range(3):
            AnnularDonutDepthExtractor.extract_depth_disparity(w["depth_map"], wbx[0], wbx[1], wbx[2], wbx[3])
            extract_relative_depth_for_box(w["depth_map"], w["box"], method="ols_plane", cfg=rob_cfg, neighbor_boxes=w["neighbors"], return_arrays=True)
            extract_relative_depth_for_box(w["depth_map"], w["box"], method="ransac_plane", cfg=rob_cfg, neighbor_boxes=w["neighbors"], return_arrays=True)
            _evaluate_method_d_once(w["depth_map"], w["box"], w["neighbors"], opt_cfg)

    methods = ("A_legacy_annular", "B_ols_plane", "C_ransac_baseline", "D_optimized_msac_ransac")
    per_method_metrics: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
    all_trial_records: List[Dict[str, Any]] = []
    d_substage_region_ms: List[float] = []
    d_substage_fit_ms: List[float] = []
    d_substage_total_ms: List[float] = []

    for trial_idx in range(timing_trials):
        for case in cases:
            dmap = case["depth_map"]
            gt = case["gt_plane"]
            bx1, by1, bx2, by2 = [int(v) for v in case["box"]]
            gt_crater = gt[by1:by2, bx1:bx2]

            # A: Legacy annular scalar-only extractor (smaller scope noted explicitly)
            t0 = time.perf_counter()
            raw_disp = AnnularDonutDepthExtractor.extract_depth_disparity(dmap, bx1, by1, bx2, by2)
            dt_a = (time.perf_counter() - t0) * 1000.0
            row_a = {
                "trial": trial_idx,
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "is_valid": bool(raw_disp > 0.0),
                "expect_valid": case["expect_valid"],
                "plane_rmse": None,
                "inlier_ratio": None,
                "score": float(raw_disp),
                "latency_ms": dt_a,
            }
            if trial_idx == 0:
                per_method_metrics["A_legacy_annular"].append(row_a)
            all_trial_records.append({"method": "A_legacy_annular", **row_a})

            # B: OLS full per-region processing
            t0 = time.perf_counter()
            out_ols = extract_relative_depth_for_box(
                dmap, case["box"], method="ols_plane", cfg=rob_cfg, neighbor_boxes=case["neighbors"], return_arrays=True
            )
            dt_b = (time.perf_counter() - t0) * 1000.0
            ols_plane = (out_ols.get("arrays") or {}).get("ref_plane_crater")
            ols_gt_rmse = (
                float(np.sqrt(np.mean((ols_plane - gt_crater) ** 2)))
                if (ols_plane is not None and ols_plane.shape == gt_crater.shape)
                else None
            )
            row_b = {
                "trial": trial_idx,
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "is_valid": bool(out_ols["is_valid"]),
                "expect_valid": case["expect_valid"],
                "plane_rmse": ols_gt_rmse,
                "inlier_ratio": (out_ols.get("ref_diagnostics") or {}).get("inlier_ratio"),
                "score": (out_ols.get("relative_cavity_stats") or {}).get("p75_residual"),
                "latency_ms": dt_b,
            }
            if trial_idx == 0:
                per_method_metrics["B_ols_plane"].append(row_b)
            all_trial_records.append({"method": "B_ols_plane", **row_b})

            # C: Baseline RANSAC full per-region processing
            t0 = time.perf_counter()
            out_ran = extract_relative_depth_for_box(
                dmap, case["box"], method="ransac_plane", cfg=rob_cfg, neighbor_boxes=case["neighbors"], return_arrays=True
            )
            dt_c = (time.perf_counter() - t0) * 1000.0
            ran_plane = (out_ran.get("arrays") or {}).get("ref_plane_crater")
            ran_gt_rmse = (
                float(np.sqrt(np.mean((ran_plane - gt_crater) ** 2)))
                if (ran_plane is not None and ran_plane.shape == gt_crater.shape)
                else None
            )
            row_c = {
                "trial": trial_idx,
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "is_valid": bool(out_ran["is_valid"]),
                "expect_valid": case["expect_valid"],
                "plane_rmse": ran_gt_rmse,
                "inlier_ratio": (out_ran.get("ref_diagnostics") or {}).get("inlier_ratio"),
                "score": (out_ran.get("relative_cavity_stats") or {}).get("p75_residual"),
                "latency_ms": dt_c,
            }
            if trial_idx == 0:
                per_method_metrics["C_ransac_baseline"].append(row_c)
            all_trial_records.append({"method": "C_ransac_baseline", **row_c})

            # D: Optimized MSAC/RANSAC full per-region processing (SINGLE execution per trial!)
            t0 = time.perf_counter()
            cav_out = _evaluate_method_d_once(dmap, case["box"], case["neighbors"], opt_cfg)
            dt_d = (time.perf_counter() - t0) * 1000.0
            fit_out = (cav_out.get("arrays") or {}).get("fit_out") or {}
            ref_crater = fit_out.get("ref_plane_crater")
            gt_err_rmse = (
                float(np.sqrt(np.mean((ref_crater - gt_crater) ** 2)))
                if (ref_crater is not None and ref_crater.shape == gt_crater.shape)
                else None
            )
            row_d = {
                "trial": trial_idx,
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "is_valid": bool(cav_out["is_valid"]),
                "expect_valid": case["expect_valid"],
                "plane_rmse": gt_err_rmse,
                "inlier_ratio": fit_out.get("inlier_ratio"),
                "score": cav_out.get("raw_relative_cavity_score"),
                "latency_ms": dt_d,
            }
            if trial_idx == 0:
                per_method_metrics["D_optimized_msac_ransac"].append(row_d)
            all_trial_records.append({"method": "D_optimized_msac_ransac", **row_d})

            # Separate sub-stage micro-benchmark (region construction vs plane fit) outside Method D main timer
            tr0 = time.perf_counter()
            reg_sub = build_and_sample_annular_region(dmap, case["box"], neighbor_boxes=case["neighbors"], cfg=opt_cfg)
            tr1 = time.perf_counter()
            fit_ransac_optimized(dmap, reg_sub, cfg=opt_cfg)
            tr2 = time.perf_counter()
            d_substage_region_ms.append((tr1 - tr0) * 1000.0)
            d_substage_fit_ms.append((tr2 - tr1) * 1000.0)
            d_substage_total_ms.append(dt_d)

    ref_plane_summary: Dict[str, Any] = {}
    for m in methods:
        rows = per_method_metrics[m]
        rmses = [r["plane_rmse"] for r in rows if r["plane_rmse"] is not None]
        outlier_rmses = [r["plane_rmse"] for r in rows if r["plane_rmse"] is not None and r["scenario"].startswith("outliers_")]
        inliers = [r["inlier_ratio"] for r in rows if r["inlier_ratio"] is not None]
        lats = [r["latency_ms"] for r in all_trial_records if r["method"] == m]
        correct_gate = sum(1 for r in rows if r["is_valid"] == r["expect_valid"])
        invalid_rej = sum(1 for r in rows if (not r["expect_valid"]) and (not r["is_valid"]))
        invalid_tot = sum(1 for r in rows if not r["expect_valid"])
        scope_note = (
            "scalar_donut_delta_only"
            if m == "A_legacy_annular"
            else "full_per_region_extraction (region_build + plane_fit + cavity_extract)"
        )
        ref_plane_summary[m] = {
            "scope": scope_note,
            "unique_cases": len(rows),
            "timed_trials": len(lats),
            "valid_count": sum(1 for r in rows if r["is_valid"]),
            "gate_concordance_pct": round((correct_gate / len(rows)) * 100.0, 2),
            "invalid_rejection_rate_pct": round((invalid_rej / max(1, invalid_tot)) * 100.0, 2),
            "mean_plane_rmse": round(float(np.mean(rmses)), 6) if rmses else None,
            "outlier_mean_plane_rmse": round(float(np.mean(outlier_rmses)), 6) if outlier_rmses else None,
            "mean_inlier_ratio": round(float(np.mean(inliers)), 4) if inliers else None,
            "mean_latency_ms": round(float(np.mean(lats)), 3),
            "median_latency_ms": round(float(np.median(lats)), 3),
            "p95_latency_ms": round(float(np.percentile(lats, 95.0)), 3),
        }

    c_mean = ref_plane_summary["C_ransac_baseline"]["mean_latency_ms"]
    d_mean = ref_plane_summary["D_optimized_msac_ransac"]["mean_latency_ms"]
    c_med = ref_plane_summary["C_ransac_baseline"]["median_latency_ms"]
    d_med = ref_plane_summary["D_optimized_msac_ransac"]["median_latency_ms"]

    speedup_summary = {
        "speedup_D_vs_C_mean": round(c_mean / d_mean, 3) if d_mean > 0 else None,
        "speedup_D_vs_C_median": round(c_med / d_med, 3) if d_med > 0 else None,
        "d_substage_breakdown_ms": {
            "region_construction_mean_ms": round(float(np.mean(d_substage_region_ms)), 3),
            "plane_fitting_mean_ms": round(float(np.mean(d_substage_fit_ms)), 3),
            "cavity_extraction_remainder_mean_ms": round(max(0.0, float(np.mean(d_substage_total_ms) - np.mean(d_substage_region_ms) - np.mean(d_substage_fit_ms))), 3),
            "full_region_processing_mean_ms": round(float(np.mean(d_substage_total_ms)), 3),
        },
        "duplicate_call_fixed": True,
    }

    # Phase 10 Relative Geometry Visualization Fidelity Check
    synth_case = _make_synthetic_case("sloped", 0, rng)
    d_synth = synth_case["depth_map"].copy()
    d_synth[88:98, 112:124] = np.nan
    ext = extract_cavity_with_strategy(d_synth, synth_case["box"], strategy="elliptical_core", opt_cfg=opt_cfg, return_arrays=True)
    t0 = time.perf_counter()
    snap = build_relative_surface_snapshot(
        snapshot_id="phase13b_fidelity_test",
        session_id="phase13b",
        frame_idx=0,
        timestamp_sec=0.0,
        track_id=1,
        track_generation_id=0,
        depth_map=d_synth,
        extract_out=ext,
        img_bgr=np.full((180, 240, 3), 100, dtype=np.uint8),
        max_grid_size=48,
        include_source_crop=True,
    )
    snap_ms = (time.perf_counter() - t0) * 1000.0
    reg = ext["arrays"]["region"]
    fit_out = ext["arrays"]["fit_out"]
    rx1, ry1, rx2, ry2 = [int(v) for v in reg["outer_xyxy"]]
    h_out, w_out = ry2 - ry1, rx2 - rx1
    coeffs = fit_out["coeffs"]
    cy_vec = (np.arange(h_out) * (2.0 / max(1.0, h_out - 1)) - 1.0)[:, None]
    cx_vec = (np.arange(w_out) * (2.0 / max(1.0, w_out - 1)) - 1.0)[None, :]
    raw_res = (coeffs[0] * cy_vec + coeffs[1] * cx_vec + coeffs[2]) - d_synth[ry1:ry2, rx1:rx2]
    raw_valid = np.isfinite(raw_res)
    ds_grid = np.array([[np.nan if v is None else float(v) for v in row] for row in snap["signed_residual_grid"]], dtype=np.float64)
    ds_valid = np.array(snap["validity_mask_grid"], dtype=bool)
    ref_true_cells = int(np.sum(np.array(snap["reference_mask_grid"], dtype=bool)))
    cav_true_cells = int(np.sum(np.array(snap["cavity_mask_grid"], dtype=bool)))
    prof = compute_cross_section_profiles(snap["signed_residual_grid"], snap["validity_mask_grid"], row_idx=snap["grid_shape"][0] // 2, col_idx=snap["grid_shape"][1] // 2)
    json_bytes = len(json.dumps(snap, allow_nan=False).encode("utf-8"))

    phase10_eval = {
        "raw_mean_residual": round(float(np.mean(raw_res[raw_valid])), 6),
        "downsampled_mean_residual": round(float(np.mean(ds_grid[ds_valid])), 6),
        "mean_resampling_abs_diff": round(abs(float(np.mean(raw_res[raw_valid])) - float(np.mean(ds_grid[ds_valid]))), 6),
        "raw_min_max": [round(float(np.min(raw_res[raw_valid])), 6), round(float(np.max(raw_res[raw_valid])), 6)],
        "downsampled_min_max": [snap["min_residual"], snap["max_residual"]],
        "reference_mask_true_cells": ref_true_cells,
        "cavity_mask_true_cells": cav_true_cells,
        "null_cells_preserved": int(np.sum(~ds_valid)),
        "cross_section_gap_preserved": any(not pt["valid"] for pt in prof["horizontal_profile"]),
        "snapshot_json_kb": round(json_bytes / 1024.0, 2),
        "generation_overhead_ms": round(snap_ms, 2),
        "extra_depth_inference_calls": 0,
        "geometry_schema_version": snap.get("geometry_schema_version"),
        "reference_mask_status": snap.get("reference_mask_status"),
    }

    return {
        "status": "COMPLETED",
        "reference_plane_ablation": ref_plane_summary,
        "latency_speedup_summary": speedup_summary,
        "per_trial_timings": all_trial_records,
        "phase10_geometry_evaluation": phase10_eval,
        "synthetic_disclaimer": "Synthetic reference-plane RMSE and timings reflect analytic surface recovery and CPU latency ONLY; they do NOT establish physical metric pothole depth accuracy.",
    }
