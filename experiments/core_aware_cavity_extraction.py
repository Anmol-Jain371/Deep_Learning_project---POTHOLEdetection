"""
========================================================================================
RoadVision AI -- Phase 6A: Core-Aware Cavity Extraction Strategies
========================================================================================
Independent experimental module comparing five cavity-region selection strategies while
holding the fitted planar road reference surface (`fit_ransac_optimized`) fixed:
  A. full_box                    : entire bounding-box interior
  B. rect_trim_10 / 20 / 30      : symmetric 10%, 20%, 30% rectangular margin trimming
  C. elliptical_core             : inscribed elliptical core (shrink factor = 0.90)
  D. multi_scale_agreement       : evaluates full, trim_15, trim_30 consistency (flags unstable)
  E. spatially_coherent_residual : largest 8-connected positive residual component
                                   (explicitly marked selection_biased=True; diagnostic ONLY!)

SCIENTIFIC CONSTRAINTS:
  - Preserves ALL original full-box signed residuals (`full_p50`, `full_p75`, `full_pos_frac`).
  - Never reverses polarity per sample or discards contradictory residuals silently.
  - Scale-normalized quality gating works invariantly across MiDaS and Depth Anything V2.
  - Geometric cores are NOT segmentation masks; relative units ONLY (never cm/volume).
========================================================================================
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from roadvision_engine import SpecularAquaplaneDetector
from experiments.optimized_relative_depth import (
    OptimizedDepthConfig,
    build_and_sample_annular_region,
    fit_ransac_optimized,
)

CAVITY_STRATEGIES = (
    "full_box",
    "rect_trim_10",
    "rect_trim_20",
    "rect_trim_30",
    "elliptical_core",
    "multi_scale_agreement",
    "spatially_coherent_residual",
)


@dataclass
class CoreExtractionConfig:
    min_interior_px: int = 16
    ellipse_shrink: float = 0.90
    multiscale_max_cv: float = 0.85
    coherent_min_frac: float = 0.10
    min_positive_strong_frac: float = 0.22
    min_inlier_ratio: float = 0.45
    min_populated_quadrants: int = 2
    max_single_quadrant_frac: float = 0.80
    min_annular_retention: float = 0.25
    reject_water_patches: bool = True
    boundary_trunc_px: int = 2
    max_iqr_vs_p75_ratio: float = 3.0


def _build_geometric_core_mask(
    h: int,
    w: int,
    strategy: str,
    cfg: CoreExtractionConfig,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return boolean interior mask over (h, w) box patch and local bounds (y1, y2, x1, x2)."""
    if strategy in ("full_box", "spatially_coherent_residual"):
        m = np.ones((h, w), dtype=bool)
        return m, (0, h, 0, w)

    trim_map = {"rect_trim_10": 0.10, "rect_trim_20": 0.20, "rect_trim_30": 0.30, "multi_scale_agreement": 0.20}
    if strategy in trim_map:
        f = trim_map[strategy]
        dy = int(round(f * h))
        dx = int(round(f * w))
        y1 = min(h - 1, max(0, dy))
        y2 = max(y1 + 1, h - dy)
        x1 = min(w - 1, max(0, dx))
        x2 = max(x1 + 1, w - dx)
        m = np.zeros((h, w), dtype=bool)
        m[y1:y2, x1:x2] = True
        return m, (y1, y2, x1, x2)

    if strategy == "elliptical_core":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
        ry = max(1.0, (h * 0.5) * cfg.ellipse_shrink)
        rx = max(1.0, (w * 0.5) * cfg.ellipse_shrink)
        r2 = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        m = r2 <= 1.0
        ys, xs = np.where(m)
        if len(ys) == 0:
            return m, (0, 0, 0, 0)
        return m, (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)

    raise ValueError(f"Unknown cavity strategy '{strategy}'; expected one of {CAVITY_STRATEGIES}")


def _region_stats(vals: np.ndarray, ref_mad: float) -> Dict[str, Any]:
    if len(vals) == 0:
        return {
            "px": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p85": 0.0, "p90": 0.0,
            "iqr": 0.0, "mad": 0.0, "pos_frac": 0.0, "pos_strong_frac": 0.0,
            "neg_strong_frac": 0.0, "scale_norm_p75": 0.0,
        }
    p25, med, p75, p85, p90 = [float(v) for v in np.percentile(vals, [25.0, 50.0, 75.0, 85.0, 90.0])]
    mad = float(np.median(np.abs(vals - med)))
    noise_floor = max(1e-5, 1.4826 * ref_mad)
    noise_eps = 0.5 * noise_floor
    return {
        "px": int(len(vals)),
        "p25": round(p25, 6),
        "median": round(med, 6),
        "p75": round(p75, 6),
        "p85": round(p85, 6),
        "p90": round(p90, 6),
        "iqr": round(p75 - p25, 6),
        "mad": round(mad, 6),
        "pos_frac": round(float(np.mean(vals > 0.0)), 4),
        "pos_strong_frac": round(float(np.mean(vals > noise_eps)), 4),
        "neg_strong_frac": round(float(np.mean(vals < -noise_eps)), 4),
        "scale_norm_p75": round(p75 / noise_floor, 4),
    }


def extract_cavity_with_strategy(
    depth_map: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    strategy: str = "elliptical_core",
    neighbor_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    img_bgr: Optional[np.ndarray] = None,
    opt_cfg: Optional[OptimizedDepthConfig] = None,
    core_cfg: Optional[CoreExtractionConfig] = None,
    precomputed_region: Optional[Dict[str, Any]] = None,
    precomputed_fit: Optional[Dict[str, Any]] = None,
    return_arrays: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a core-aware cavity strategy while holding the fitted road reference plane
    identical. Allows passing `precomputed_region` and `precomputed_fit` to guarantee
    zero plane drift when comparing strategies A–E side-by-side.
    """
    if strategy not in CAVITY_STRATEGIES:
        raise ValueError(f"Unknown cavity strategy '{strategy}'")
    opt_cfg = opt_cfg or OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    core_cfg = core_cfg or CoreExtractionConfig()

    t0 = time.perf_counter()
    region = precomputed_region or build_and_sample_annular_region(depth_map, box_xyxy, neighbor_boxes=neighbor_boxes, cfg=opt_cfg)
    fit_out = precomputed_fit or fit_ransac_optimized(depth_map, region, cfg=opt_cfg)

    x1, y1, x2, y2 = region["box_xyxy"]
    crater_patch = depth_map[y1:y2, x1:x2].astype(np.float64)
    h, w = crater_patch.shape
    res_map = fit_out["ref_plane_crater"] - crater_patch
    finite_all = np.isfinite(res_map)

    ref_res = fit_out["ref_residuals"]
    ref_mad = float(np.median(np.abs(ref_res - np.median(ref_res)))) if len(ref_res) > 0 else 0.0
    ref_rmse = float(np.sqrt(np.mean(ref_res ** 2))) if len(ref_res) > 0 else 0.0

    # Preserve complete full-box diagnostics strictly
    full_stats = _region_stats(res_map[finite_all], ref_mad=ref_mad)

    # Select interior mask per strategy
    selection_biased = False
    multiscale_info = None
    unstable_multiscale = False

    if strategy == "spatially_coherent_residual":
        selection_biased = True
        noise_floor = max(1e-5, 1.4826 * ref_mad)
        cand = finite_all & (res_map > 0.25 * noise_floor)
        core_mask = np.zeros((h, w), dtype=bool)
        if np.any(cand):
            n_cc, lbls, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
            if n_cc > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                best_lbl = int(np.argmax(areas)) + 1
                best_area = int(areas[best_lbl - 1])
                if best_area >= max(core_cfg.min_interior_px, int(core_cfg.coherent_min_frac * h * w)):
                    core_mask = (lbls == best_lbl)
        ys_m, xs_m = np.where(core_mask)
        local_bounds = (int(ys_m.min()), int(ys_m.max()) + 1, int(xs_m.min()), int(xs_m.max()) + 1) if len(ys_m) > 0 else (0, 0, 0, 0)
    else:
        core_mask, local_bounds = _build_geometric_core_mask(h, w, strategy, core_cfg)

    valid_core_mask = core_mask & finite_all
    core_vals = res_map[valid_core_mask]
    core_stats = _region_stats(core_vals, ref_mad=ref_mad)
    coverage_ratio = round(float(core_mask.sum()) / float(max(1, h * w)), 4)

    # Multi-scale interior agreement check (D): inspect trim_10, trim_20, trim_30 & full_box together
    if strategy == "multi_scale_agreement":
        scale_p75s = {}
        scale_pos = {}
        for s_tag in ("full_box", "rect_trim_10", "rect_trim_20", "rect_trim_30"):
            sm, _ = _build_geometric_core_mask(h, w, s_tag, core_cfg)
            sv = res_map[sm & finite_all]
            if len(sv) >= core_cfg.min_interior_px:
                scale_p75s[s_tag] = float(np.percentile(sv, 75.0))
                scale_pos[s_tag] = float(np.mean(sv > 0.0))
        noise_eps = max(1e-5, 0.5 * 1.4826 * ref_mad)
        if len(scale_p75s) >= 2:
            vals_s = list(scale_p75s.values())
            med_s = float(np.median(vals_s))
            spread_s = float(max(vals_s) - min(vals_s))
            rel_spread = spread_s / max(abs(med_s), 2.0 * noise_eps)
            sign_flip = (min(vals_s) <= 0.0 and max(vals_s) > noise_eps)
            unstable_multiscale = sign_flip or (rel_spread > core_cfg.multiscale_max_cv)
            multiscale_info = {
                "scale_p75s": {k: round(v, 5) for k, v in scale_p75s.items()},
                "scale_pos_fracs": {k: round(v, 4) for k, v in scale_pos.items()},
                "sign_flip_across_scales": sign_flip,
                "rel_spread": round(rel_spread, 4),
            }
        else:
            unstable_multiscale = True

    is_water = False
    if img_bgr is not None and x2 > x1 and y2 > y1:
        is_water = SpecularAquaplaneDetector.inspect_patch(img_bgr[y1:y2, x1:x2])

    # Quality gating (scale-normalized across models; zero gate relaxation!)
    reasons: List[str] = []
    warnings: List[str] = []

    finite_core_ratio = float(valid_core_mask.sum()) / float(max(1, int(core_mask.sum())))
    if (
        finite_core_ratio < 0.90
        or core_stats["px"] < core_cfg.min_interior_px
        or region["nonfinite_ring_px"] > max(10, int(0.10 * region["ideal_ring_px"]))
    ):
        reasons.append("insufficient_or_nonfinite_interior")
    if fit_out["used_ref_samples"] < opt_cfg.min_ref_samples:
        reasons.append("insufficient_reference_samples")
    if region["populated_quadrants"] < core_cfg.min_populated_quadrants or region["max_quadrant_frac"] > core_cfg.max_single_quadrant_frac:
        reasons.append("poor_spatial_support")
    if fit_out["plane_degenerate"]:
        reasons.append("degenerate_reference_plane")
    if fit_out["inlier_ratio"] < core_cfg.min_inlier_ratio:
        reasons.append("low_inlier_ratio")
    if region["annular_retention_ratio"] < core_cfg.min_annular_retention:
        reasons.append("overlapping_neighbors_insufficient_pavement")
    elif region["neighbor_excluded_px"] > 0:
        warnings.append("neighbors_excluded_from_ring")

    # Polarity & instability checks evaluated on selected region (note: for selection-biased E,
    # also enforce that full-box is not overwhelmingly negative contradictory debris)
    noise_eps = max(1e-5, 0.5 * 1.4826 * ref_mad)
    if (
        core_stats["p75"] <= noise_eps
        or core_stats["pos_strong_frac"] < core_cfg.min_positive_strong_frac
        or (core_stats["neg_strong_frac"] > core_stats["pos_strong_frac"] and core_stats["p75"] <= 2.0 * noise_eps)
    ):
        reasons.append("unexpected_residual_polarity")
    else:
        # Scale-normalized IQR vs signal check (replaces fixed `1.0` floor so MiDaS & DAv2 match consistently)
        scale_floor = max(core_cfg.max_iqr_vs_p75_ratio * max(core_stats["p75"], 1e-5), 4.0 * 1.4826 * ref_mad, 1e-4)
        if core_stats["iqr"] > scale_floor and core_stats["pos_frac"] < 0.50:
            reasons.append("unstable_cavity_residuals")

    if strategy == "multi_scale_agreement" and unstable_multiscale:
        reasons.append("unstable_multiscale_interior")

    if region["truncated_edges"]:
        warnings.append(f"boundary_truncated:{'+'.join(region['truncated_edges'])}")
        if len(region["truncated_edges"]) >= 2 and region["populated_quadrants"] < core_cfg.min_populated_quadrants and "poor_spatial_support" not in reasons:
            reasons.append("frame_boundary_truncation")

    if is_water:
        warnings.append("suspected_water_reflection")
        if core_cfg.reject_water_patches:
            reasons.append("suspected_water_reflection")

    is_valid = len(reasons) == 0
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Global coordinates of the selected interior bounds
    gy1, gy2, gx1, gx2 = y1 + local_bounds[0], y1 + local_bounds[1], x1 + local_bounds[2], x1 + local_bounds[3]

    out = {
        "strategy": strategy,
        "selection_biased": selection_biased,
        "box_xyxy": [x1, y1, x2, y2],
        "selected_interior_xyxy": [gx1, gy1, gx2, gy2],
        "interior_coverage_ratio": coverage_ratio,
        "is_valid": is_valid,
        "failure_reasons": reasons,
        "warnings": warnings,
        "is_water_suspected": is_water,
        "core_cavity_stats": {
            "px": core_stats["px"],
            "p75_residual": core_stats["p75"] if is_valid else None,
            "median_residual": core_stats["median"] if is_valid else None,
            "iqr_residual": core_stats["iqr"] if is_valid else None,
            "scale_normalized_p75": core_stats["scale_norm_p75"] if is_valid else None,
            "positive_frac": core_stats["pos_frac"],
            "raw_diag_p75": core_stats["p75"],
            "raw_diag_median": core_stats["median"],
        },
        "full_box_preserved_stats": full_stats,
        "multiscale_agreement": multiscale_info,
        "ref_diagnostics": {
            "used_ref_samples": fit_out["used_ref_samples"],
            "available_ref_samples": region["available_ref_samples"],
            "populated_quadrants": region["populated_quadrants"],
            "inlier_ratio": fit_out["inlier_ratio"],
            "ref_fit_mad": round(ref_mad, 6),
            "ref_fit_rmse": round(ref_rmse, 6),
        },
        "total_ms": round(dt_ms, 4),
    }
    if return_arrays:
        out["arrays"] = {
            "region": region,
            "fit_out": fit_out,
            "signed_residual_map": res_map,
            "core_mask": core_mask,
        }
    return out
