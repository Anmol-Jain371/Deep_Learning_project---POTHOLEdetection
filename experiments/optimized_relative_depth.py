"""
========================================================================================
RoadVision AI -- Phase 5B: Optimized Adaptive RANSAC & Reference Point Budget Extraction
========================================================================================
Experimental research module optimizing Phase 5A's `ransac_plane` pipeline without
sacrificing robustness or altering any production files (`roadvision_engine.py`, `server.py`).

Key algorithmic optimizations over `experiments/robust_relative_depth.py`:
  1. Configurable reference-point budget sampling (`256, 512, 1024, 2048`, or `None` for
     full resolution) across four strategies:
       - `full`:                 all finite neighbor-excluded annular pixels
       - `uniform_stride`:       even linear-step subsampling
       - `stratified_quadrant`:  proportional quadrant-guaranteed sampling preserving 4-side support
       - `deterministic_random`: seeded pseudo-random permutation subsampling
  2. Vectorized batch hypothesis generation & MSAC consensus scoring (`32` 3-point planes
     solved simultaneously via batched `np.linalg.solve` + matrix-multiply consensus).
  3. Adaptive RANSAC/MSAC iteration budget (`N = ceil(log(1 - conf) / log(1 - w^3))`) with
     deterministic seeding and early termination after `ransac_min_iters`.
  4. Fast 3x3 normal-equation final OLS refit (`A^T A c = A^T z`) with singular-value /
     eigenvalue conditioning check instead of full-matrix SVD on N x 3 pixels.
  5. Vectorized local-surface finite-difference sigma estimation on sampled ring pairs and
     single-pass joint `np.percentile(..., [25, 50, 75, 85, 90])` cavity statistics.

SCIENTIFIC CONVENTIONS & CONSTRAINTS:
  - Preserves ALL 11 Phase 5A quality gates cleanly (no silent gate weakening).
  - Outputs represent MiDaS relative inverse-depth residuals ONLY — NEVER physical cm.
  - Heuristic reliability scores are NOT calibrated probabilities of physical correctness.
========================================================================================
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from roadvision_engine import SpecularAquaplaneDetector

SAMPLING_MODES = (
    "full",
    "uniform_stride",
    "stratified_quadrant",
    "deterministic_random",
)


@dataclass
class OptimizedDepthConfig:
    # Adaptive annular geometry (identical to Phase 5A RobustDepthConfig)
    outer_margin_ratio: float = 0.45
    min_margin_px: int = 16
    max_margin_px: int = 120
    inner_pad_ratio: float = 0.05
    exclude_neighbors: bool = True
    neighbor_pad_px: int = 4
    horizon_cutoff_ratio: float = 0.15

    # Optimized sampling controls
    sampling_mode: str = "stratified_quadrant"
    max_ref_samples: Optional[int] = 512

    # Vectorized adaptive RANSAC controls
    ransac_max_iters: int = 128
    ransac_min_iters: int = 16
    ransac_batch_size: int = 16
    ransac_confidence: float = 0.99
    enable_early_stop: bool = True
    ransac_mad_mult: float = 2.5
    seed: int = 42

    # Phase 5A quality assessment gates (explicitly preserved — NOT relaxed)
    min_ref_samples: int = 40
    min_quadrant_samples: int = 5
    min_populated_quadrants: int = 2
    max_single_quadrant_frac: float = 0.80
    max_cond_number: float = 1e4
    min_annular_retention: float = 0.25
    min_inlier_ratio: float = 0.45
    min_positive_depress_frac: float = 0.35
    max_ref_mad_abs: float = 1e6
    max_ref_mad_vs_iqr_ratio: float = 3.0
    reject_water_patches: bool = True
    boundary_trunc_px: int = 2


def _normalize_coords_fast(
    ys: np.ndarray,
    xs: np.ndarray,
    h_out: int,
    w_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    inv_h = 2.0 / max(1.0, float(h_out - 1))
    inv_w = 2.0 / max(1.0, float(w_out - 1))
    yn = ys.astype(np.float64) * inv_h - 1.0
    xn = xs.astype(np.float64) * inv_w - 1.0
    return yn, xn


def _estimate_local_sigma_fast(
    outer_patch: np.ndarray,
    valid_ring: np.ndarray,
    max_pairs: int = 4096,
) -> float:
    """Fast intrinsic surface-noise estimator from strided finite differences on valid ring pixels."""
    h, w = outer_patch.shape
    diffs = []
    if h >= 2:
        vm_y = valid_ring[1:, :] & valid_ring[:-1, :]
        vy, vx = np.where(vm_y)
        if len(vy) > 0:
            if len(vy) > max_pairs:
                step = len(vy) // max_pairs
                vy, vx = vy[::step][:max_pairs], vx[::step][:max_pairs]
            diffs.append(np.abs(outer_patch[vy + 1, vx] - outer_patch[vy, vx]))
    if w >= 2:
        vm_x = valid_ring[:, 1:] & valid_ring[:, :-1]
        hy, hx = np.where(vm_x)
        if len(hy) > 0:
            if len(hy) > max_pairs:
                step = len(hy) // max_pairs
                hy, hx = hy[::step][:max_pairs], hx[::step][:max_pairs]
            diffs.append(np.abs(outer_patch[hy, hx + 1] - outer_patch[hy, hx]))
    if not diffs:
        return 1e-3
    all_d = diffs[0] if len(diffs) == 1 else np.concatenate(diffs)
    return max(1e-4, 1.048 * float(np.median(all_d)))


def _compute_quadrant_labels(
    ys: np.ndarray,
    xs: np.ndarray,
    cy_local: float,
    cx_local: float,
) -> Tuple[np.ndarray, List[int]]:
    """Classify ring points into 4 axial quadrants: 0=Top, 1=Bottom, 2=Left, 3=Right."""
    dy = ys.astype(np.float64) - cy_local
    dx = xs.astype(np.float64) - cx_local
    horiz = np.abs(dx) > np.abs(dy)
    q_labels = np.empty(len(ys), dtype=np.int8)
    # Vertical sectors: dy < 0 -> 0 (Top), dy >= 0 -> 1 (Bottom)
    vert_m = ~horiz
    q_labels[vert_m] = np.where(dy[vert_m] < 0.0, 0, 1)
    # Horizontal sectors: dx < 0 -> 2 (Left), dx >= 0 -> 3 (Right)
    q_labels[horiz] = np.where(dx[horiz] < 0.0, 2, 3)
    counts = np.bincount(q_labels, minlength=4).astype(int).tolist()
    return q_labels, counts


def build_and_sample_annular_region(
    depth_map: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    neighbor_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    cfg: Optional[OptimizedDepthConfig] = None,
) -> Dict[str, Any]:
    """
    Construct the neighbor-excluded adaptive annular region AND apply structured reference
    point budgeting (`full`, `uniform_stride`, `stratified_quadrant`, `deterministic_random`).
    """
    cfg = cfg or OptimizedDepthConfig()
    if cfg.sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown sampling_mode '{cfg.sampling_mode}'; expected one of {SAMPLING_MODES}")

    H, W = depth_map.shape[:2]
    x1 = max(0, int(np.floor(box_xyxy[0])))
    y1 = max(0, int(np.floor(box_xyxy[1])))
    x2 = min(W, int(np.ceil(box_xyxy[2])))
    y2 = min(H, int(np.ceil(box_xyxy[3])))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    ip_x = max(0, int(round(bw * cfg.inner_pad_ratio)))
    ip_y = max(0, int(round(bh * cfg.inner_pad_ratio)))
    ex_x1, ex_y1 = max(0, x1 - ip_x), max(0, y1 - ip_y)
    ex_x2, ex_y2 = min(W, x2 + ip_x), min(H, y2 + ip_y)

    mx = max(cfg.min_margin_px, min(cfg.max_margin_px, int(round(bw * cfg.outer_margin_ratio))))
    my = max(cfg.min_margin_px, min(cfg.max_margin_px, int(round(bh * cfg.outer_margin_ratio))))

    horizon_y = int(round(H * cfg.horizon_cutoff_ratio))
    rx1 = max(0, x1 - mx)
    ry1 = max(horizon_y, y1 - my) if y1 >= horizon_y else max(0, y1 - my)
    rx2 = min(W, x2 + mx)
    ry2 = min(H, y2 + my)
    if rx2 <= rx1:
        rx2 = min(W, rx1 + 1)
    if ry2 <= ry1:
        ry2 = min(H, ry1 + 1)

    h_out, w_out = ry2 - ry1, rx2 - rx1
    ideal_outer = np.ones((h_out, w_out), dtype=bool)
    iy1_p, iy2_p = max(0, ex_y1 - ry1), min(h_out, ex_y2 - ry1)
    ix1_p, ix2_p = max(0, ex_x1 - rx1), min(w_out, ex_x2 - rx1)
    ideal_outer[iy1_p:iy2_p, ix1_p:ix2_p] = False
    ideal_ring_px = max(1, int(ideal_outer.sum()))

    valid_ring = ideal_outer.copy()
    neighbor_excluded_px = 0
    if cfg.exclude_neighbors and neighbor_boxes:
        for nb in neighbor_boxes:
            cx1 = max(0, int(np.floor(nb[0])) - cfg.neighbor_pad_px - rx1)
            cy1 = max(0, int(np.floor(nb[1])) - cfg.neighbor_pad_px - ry1)
            cx2 = min(w_out, int(np.ceil(nb[2])) + cfg.neighbor_pad_px - rx1)
            cy2 = min(h_out, int(np.ceil(nb[3])) + cfg.neighbor_pad_px - ry1)
            if cx2 > cx1 and cy2 > cy1:
                block = valid_ring[cy1:cy2, cx1:cx2]
                ex_cnt = int(block.sum())
                if ex_cnt > 0:
                    block[:] = False
                    neighbor_excluded_px += ex_cnt

    outer_patch = depth_map[ry1:ry2, rx1:rx2]
    finite_outer = np.isfinite(outer_patch)
    # Count non-finite inside active ring before masking
    pre_finite_cnt = int(valid_ring.sum())
    valid_ring &= finite_outer
    nonfinite_ring_px = pre_finite_cnt - int(valid_ring.sum())

    ys_full, xs_full = np.where(valid_ring)
    n_avail = len(ys_full)
    cy_local = ((y1 + y2) * 0.5) - ry1
    cx_local = ((x1 + x2) * 0.5) - rx1

    if n_avail > 0:
        q_labels_full, full_quad_counts = _compute_quadrant_labels(ys_full, xs_full, cy_local, cx_local)
        full_pop_quads = sum(1 for q in full_quad_counts if q >= cfg.min_quadrant_samples)
        full_max_quad_frac = max(full_quad_counts) / float(n_avail)
    else:
        q_labels_full = np.array([], dtype=np.int8)
        full_quad_counts = [0, 0, 0, 0]
        full_pop_quads = 0
        full_max_quad_frac = 1.0

    # Apply requested reference sampling strategy
    budget = cfg.max_ref_samples
    if n_avail == 0 or cfg.sampling_mode == "full" or budget is None or n_avail <= budget:
        sel_idx = np.arange(n_avail, dtype=np.int64)
    elif cfg.sampling_mode == "uniform_stride":
        # Evenly spaced across row-major valid ring coordinates
        sel_idx = np.linspace(0, n_avail - 1, num=budget, dtype=np.int64)
    elif cfg.sampling_mode == "deterministic_random":
        rng_s = np.random.default_rng(cfg.seed + 101)
        sel_idx = rng_s.choice(n_avail, size=budget, replace=False)
        sel_idx.sort()
    elif cfg.sampling_mode == "stratified_quadrant":
        # Proportional quadrant allocation with guaranteed floor per populated quadrant
        pop_qs = [q for q in range(4) if full_quad_counts[q] > 0]
        alloc = [0, 0, 0, 0]
        if pop_qs:
            # First give each populated quadrant a spatial-support floor
            floor_q = min(cfg.min_quadrant_samples, max(1, budget // len(pop_qs)))
            rem = budget
            for q in pop_qs:
                take = min(full_quad_counts[q], floor_q)
                alloc[q] = take
                rem -= take
            # Distribute remaining budget proportionally across unexhausted quadrants
            if rem > 0:
                spare = [max(0, full_quad_counts[q] - alloc[q]) for q in range(4)]
                tot_spare = sum(spare)
                if tot_spare > 0:
                    extra = [int(np.floor(rem * spare[q] / float(tot_spare))) for q in range(4)]
                    for q in range(4):
                        add = min(spare[q], extra[q])
                        alloc[q] += add
                        rem -= add
                    # Top off rounding remainder deterministically
                    for q in sorted(pop_qs, key=lambda qq: -(full_quad_counts[qq] - alloc[qq])):
                        if rem <= 0:
                            break
                        room = full_quad_counts[q] - alloc[q]
                        add = min(room, rem)
                        alloc[q] += add
                        rem -= add
        picks = []
        for q in range(4):
            if alloc[q] <= 0:
                continue
            q_idx = np.where(q_labels_full == q)[0]
            if alloc[q] >= len(q_idx):
                picks.append(q_idx)
            else:
                pos = np.linspace(0, len(q_idx) - 1, num=alloc[q], dtype=np.int64)
                picks.append(q_idx[pos])
        sel_idx = np.concatenate(picks) if picks else np.array([], dtype=np.int64)
        sel_idx.sort()
    else:
        sel_idx = np.arange(n_avail, dtype=np.int64)

    ys_samp = ys_full[sel_idx]
    xs_samp = xs_full[sel_idx]
    z_samp = outer_patch[ys_samp, xs_samp].astype(np.float64)
    n_samp = len(ys_samp)

    if n_samp > 0:
        if n_samp == n_avail:
            samp_quad_counts = list(full_quad_counts)
            samp_pop_quads = full_pop_quads
            samp_max_quad_frac = full_max_quad_frac
        else:
            q_s = q_labels_full[sel_idx]
            samp_quad_counts = np.bincount(q_s, minlength=4).astype(int).tolist()
            samp_pop_quads = sum(1 for q in samp_quad_counts if q >= cfg.min_quadrant_samples)
            samp_max_quad_frac = max(samp_quad_counts) / float(n_samp)
    else:
        samp_quad_counts = [0, 0, 0, 0]
        samp_pop_quads = 0
        samp_max_quad_frac = 1.0

    # Frame boundary clipping flags
    truncated_edges = []
    if x1 <= cfg.boundary_trunc_px:
        truncated_edges.append("left")
    if (W - x2) <= cfg.boundary_trunc_px:
        truncated_edges.append("right")
    if y1 <= horizon_y + cfg.boundary_trunc_px:
        truncated_edges.append("top_or_horizon")
    if (H - y2) <= cfg.boundary_trunc_px:
        truncated_edges.append("bottom")

    return {
        "box_xyxy": (x1, y1, x2, y2),
        "outer_xyxy": (rx1, ry1, rx2, ry2),
        "h_out": h_out,
        "w_out": w_out,
        "outer_patch": outer_patch,
        "valid_ring_mask": valid_ring,
        "ys_samp": ys_samp,
        "xs_samp": xs_samp,
        "z_samp": z_samp,
        "available_ref_samples": n_avail,
        "sampled_ref_samples": n_samp,
        "sampling_mode": cfg.sampling_mode,
        "sampling_budget": budget,
        "sampling_coverage_ratio": round(n_samp / float(max(1, n_avail)), 4) if n_avail > 0 else 0.0,
        "ideal_ring_px": ideal_ring_px,
        "annular_retention_ratio": round(n_avail / float(ideal_ring_px), 4),
        "neighbor_excluded_px": neighbor_excluded_px,
        "nonfinite_ring_px": nonfinite_ring_px,
        # Preserve full-ring quadrant diagnostics AND sampled support checks
        "full_quadrant_counts": full_quad_counts,
        "full_populated_quadrants": full_pop_quads,
        "full_max_quadrant_frac": round(full_max_quad_frac, 4),
        "quadrant_counts": samp_quad_counts,
        "populated_quadrants": samp_pop_quads,
        "max_quadrant_frac": round(samp_max_quad_frac, 4),
        "truncated_edges": truncated_edges,
    }


def _solve_3x3_normal_eq(
    A: np.ndarray,
    z: np.ndarray,
    max_cond: float,
) -> Tuple[np.ndarray, int, float, bool]:
    """Solve linear plane coefficients via fast 3x3 normal equations with condition validation."""
    AtA = A.T @ A
    Atz = A.T @ z
    eigs = np.linalg.eigvalsh(AtA)
    eig_min = float(eigs[0])
    eig_max = float(eigs[-1])
    if eig_min <= 1e-12 or eig_max <= 1e-12:
        med_z = float(np.median(z))
        return np.array([0.0, 0.0, med_z], dtype=np.float64), int(np.sum(eigs > 1e-12)), 1e9, True
    cond = float(np.sqrt(eig_max / eig_min))
    if cond > max_cond:
        med_z = float(np.median(z))
        return np.array([0.0, 0.0, med_z], dtype=np.float64), 3, cond, True
    try:
        coeffs = np.linalg.solve(AtA, Atz)
        return coeffs, 3, cond, False
    except np.linalg.LinAlgError:
        med_z = float(np.median(z))
        return np.array([0.0, 0.0, med_z], dtype=np.float64), 0, 1e9, True


def fit_ransac_optimized(
    depth_map: np.ndarray,
    region: Dict[str, Any],
    cfg: Optional[OptimizedDepthConfig] = None,
) -> Dict[str, Any]:
    """
    Vectorized adaptive RANSAC/MSAC planar reference fitter operating on the budgeted
    spatial reference samples and reusing outer-patch normalized coordinates.
    """
    cfg = cfg or OptimizedDepthConfig()
    rx1, ry1, rx2, ry2 = region["outer_xyxy"]
    x1, y1, x2, y2 = region["box_xyxy"]
    h_out, w_out = region["h_out"], region["w_out"]

    crater_patch = depth_map[y1:y2, x1:x2].astype(np.float64)
    ys_s = region["ys_samp"]
    xs_s = region["xs_samp"]
    z_s = region["z_samp"]
    n_pts = len(z_s)

    if n_pts == 0:
        return {
            "method": "optimized_ransac",
            "ref_plane_crater": np.zeros_like(crater_patch, dtype=np.float64),
            "ref_residuals": np.array([], dtype=np.float64),
            "inlier_mask": np.array([], dtype=bool),
            "inlier_ratio": 0.0,
            "rank": 0,
            "cond_number": 1e9,
            "plane_degenerate": True,
            "used_ref_samples": 0,
            "coeffs": [0.0, 0.0, 0.0],
            "iters_evaluated": 0,
            "valid_hypotheses_evaluated": 0,
            "degenerate_triplets_rejected": 0,
            "early_stopped": False,
            "local_sigma": 0.0,
        }

    if n_pts < 3:
        med_z = float(np.median(z_s))
        return {
            "method": "optimized_ransac",
            "ref_plane_crater": np.full_like(crater_patch, med_z, dtype=np.float64),
            "ref_residuals": z_s - med_z,
            "inlier_mask": np.ones(n_pts, dtype=bool),
            "inlier_ratio": 1.0,
            "rank": 1,
            "cond_number": 1e9,
            "plane_degenerate": True,
            "used_ref_samples": n_pts,
            "coeffs": [0.0, 0.0, round(med_z, 6)],
            "iters_evaluated": 0,
            "valid_hypotheses_evaluated": 0,
            "degenerate_triplets_rejected": 0,
            "early_stopped": False,
            "local_sigma": 0.0,
        }

    # Build 1D normalized coordinates for sample points; 1D crater axes for separable outer surface evaluation
    yn, xn = _normalize_coords_fast(ys_s, xs_s, h_out, w_out)
    A = np.column_stack([yn, xn, np.ones_like(yn)])

    coeffs_ols, rank_ols, cond_ols, degen_ols = _solve_3x3_normal_eq(A, z_s, cfg.max_cond_number)
    if degen_ols:
        med_z = float(np.median(z_s))
        return {
            "method": "optimized_ransac",
            "ref_plane_crater": np.full_like(crater_patch, med_z, dtype=np.float64),
            "ref_residuals": z_s - med_z,
            "inlier_mask": np.ones(n_pts, dtype=bool),
            "inlier_ratio": 1.0,
            "rank": int(rank_ols),
            "cond_number": round(cond_ols, 2),
            "plane_degenerate": True,
            "used_ref_samples": n_pts,
            "coeffs": [0.0, 0.0, round(med_z, 6)],
            "iters_evaluated": 0,
            "valid_hypotheses_evaluated": 0,
            "degenerate_triplets_rejected": 0,
            "early_stopped": False,
            "local_sigma": 0.0,
        }

    t_sigma0 = time.perf_counter()
    local_sigma = _estimate_local_sigma_fast(region["outer_patch"].astype(np.float64), region["valid_ring_mask"])
    t_sigma_ms = (time.perf_counter() - t_sigma0) * 1000.0

    q25, q75 = np.percentile(z_s, [25.0, 75.0])
    thresh = max(1e-3, 3.0 * local_sigma, 0.005 * float(q75 - q25))

    rng = np.random.default_rng(cfg.seed)
    t_loop0 = time.perf_counter()

    # If n_pts exceeds ransac sample pool (2048+), score against a capped sub-pool matching Phase 5A speed,
    # otherwise use the exact budgeted sample points directly.
    A_eval = A
    z_eval = z_s
    n_eval = n_pts

    best_count = -1
    best_err_sum = float("inf")
    best_coeffs = coeffs_ols
    best_inl_eval: Optional[np.ndarray] = None

    iters_done = 0
    valid_hyps = 0
    degen_triplets = 0
    early_stopped = False
    max_iters = max(1, cfg.ransac_max_iters)
    batch_sz = max(1, cfg.ransac_batch_size)
    dynamic_limit = max_iters
    At = A_eval.T  # (3, n_eval)

    while iters_done < max_iters and iters_done < dynamic_limit:
        cur_b = min(batch_sz, max_iters - iters_done)
        # Generate cur_b random 3-index hypotheses deterministically in one vectorized call
        # Using distinct index pairs by picking i0, i1, i2 mod n_eval with replacement probability O(3/n_eval) rejected cleanly
        tri = rng.integers(0, n_eval, size=(cur_b, 3), dtype=np.int64)
        distinct_mask = (tri[:, 0] != tri[:, 1]) & (tri[:, 0] != tri[:, 2]) & (tri[:, 1] != tri[:, 2])

        M3 = A_eval[tri]  # (cur_b, 3, 3)
        det2 = np.abs(
            (M3[:, 1, 1] - M3[:, 0, 1]) * (M3[:, 2, 0] - M3[:, 0, 0])
            - (M3[:, 2, 1] - M3[:, 0, 1]) * (M3[:, 1, 0] - M3[:, 0, 0])
        )
        valid_mask = distinct_mask & (det2 >= 1e-4)
        n_bad = int((~valid_mask).sum())
        degen_triplets += n_bad
        iters_done += cur_b

        if np.any(valid_mask):
            M_v = M3[valid_mask]        # (K, 3, 3)
            z_v = z_eval[tri[valid_mask]]  # (K, 3)
            try:
                c_batch = np.linalg.solve(M_v, z_v[..., None])[..., 0]  # (K, 3)
            except np.linalg.LinAlgError:
                # Fallback individually inside degenerate edge batch
                c_list = []
                for mk, zk in zip(M_v, z_v):
                    try:
                        c_list.append(np.linalg.solve(mk, zk))
                    except np.linalg.LinAlgError:
                        degen_triplets += 1
                if c_list:
                    c_batch = np.stack(c_list, axis=0)
                else:
                    c_batch = np.empty((0, 3), dtype=np.float64)

            if len(c_batch) > 0:
                valid_hyps += len(c_batch)
                # Vectorized batch scoring: (K, 3) @ (3, n_eval) -> (K, n_eval)
                abs_errs = np.abs(z_eval[None, :] - (c_batch @ At))
                inl_mat = abs_errs <= thresh
                counts = np.sum(inl_mat, axis=1)
                err_sums = np.sum(np.minimum(abs_errs, thresh), axis=1)

                max_cnt = int(counts.max())
                top_idx_candidates = np.where(counts == max_cnt)[0]
                best_b_idx = int(top_idx_candidates[int(np.argmin(err_sums[top_idx_candidates]))])
                b_cnt = int(counts[best_b_idx])
                b_err = float(err_sums[best_b_idx])

                if b_cnt > best_count or (b_cnt == best_count and b_err < best_err_sum):
                    best_count = b_cnt
                    best_err_sum = b_err
                    best_coeffs = c_batch[best_b_idx]
                    best_inl_eval = inl_mat[best_b_idx]

                    if cfg.enable_early_stop and best_count >= 6:
                        w_best = best_count / float(n_eval)
                        w3 = w_best ** 3
                        if w3 >= 1.0 - 1e-12:
                            dynamic_limit = max(cfg.ransac_min_iters, iters_done)
                        elif w3 > 1e-6:
                            est_n = int(np.ceil(np.log(1.0 - cfg.ransac_confidence) / np.log(1.0 - w3)))
                            dynamic_limit = max(cfg.ransac_min_iters, min(max_iters, est_n))

        if cfg.enable_early_stop and iters_done >= max(cfg.ransac_min_iters, dynamic_limit) and best_count >= 6:
            early_stopped = True
            break

    t_loop_ms = (time.perf_counter() - t_loop0) * 1000.0
    t_refit0 = time.perf_counter()

    if best_inl_eval is None or best_count < 6:
        coeffs_fin = coeffs_ols
        inl_fin = np.ones(n_pts, dtype=bool)
        degen_fin = True
        cond_fin = cond_ols
        rank_fin = rank_ols
    else:
        c_mid, _, _, deg_mid = _solve_3x3_normal_eq(A[best_inl_eval], z_s[best_inl_eval], cfg.max_cond_number)
        if deg_mid:
            c_mid = best_coeffs
        r_mid = np.abs(z_s - (A @ c_mid))
        inlier_r = r_mid[r_mid <= thresh * 1.5]
        mad_inl = float(np.median(inlier_r)) if len(inlier_r) > 0 else local_sigma
        ref_thresh = max(thresh, cfg.ransac_mad_mult * 1.4826 * mad_inl)
        inl_fin = r_mid <= ref_thresh
        if int(inl_fin.sum()) >= 6:
            coeffs_fin, rank_fin, cond_fin, degen_fin = _solve_3x3_normal_eq(
                A[inl_fin], z_s[inl_fin], cfg.max_cond_number
            )
            if degen_fin:
                coeffs_fin = coeffs_ols
        else:
            coeffs_fin = c_mid
            rank_fin = rank_ols
            cond_fin = cond_ols
            degen_fin = True

    t_refit_ms = (time.perf_counter() - t_refit0) * 1000.0

    # Separable 1D reference-plane broadcast across crater patch (avoids full 2D np.indices allocation)
    inv_h = 2.0 / max(1.0, float(h_out - 1))
    inv_w = 2.0 / max(1.0, float(w_out - 1))
    cy_vec = (np.arange(y1 - ry1, y2 - ry1, dtype=np.float64) * inv_h - 1.0)[:, None]
    cx_vec = (np.arange(x1 - rx1, x2 - rx1, dtype=np.float64) * inv_w - 1.0)[None, :]
    plane_crater = coeffs_fin[0] * cy_vec + coeffs_fin[1] * cx_vec + coeffs_fin[2]

    res_fin = z_s - (A @ coeffs_fin)
    return {
        "method": "optimized_ransac",
        "ref_plane_crater": plane_crater,
        "ref_residuals": res_fin,
        "inlier_mask": inl_fin,
        "inlier_ratio": round(float(inl_fin.mean()), 4),
        "rank": int(rank_fin),
        "cond_number": round(cond_fin, 2),
        "plane_degenerate": degen_fin,
        "used_ref_samples": n_pts,
        "coeffs": [round(float(c), 6) for c in coeffs_fin],
        "iters_evaluated": iters_done,
        "valid_hypotheses_evaluated": valid_hyps,
        "degenerate_triplets_rejected": degen_triplets,
        "early_stopped": early_stopped,
        "local_sigma": round(local_sigma, 6),
        "sub_timing_ms": {
            "local_sigma_ms": round(t_sigma_ms, 4),
            "ransac_loop_ms": round(t_loop_ms, 4),
            "refit_plane_ms": round(t_refit_ms, 4),
        },
    }


def compute_cavity_statistics_fast(
    crater_patch: np.ndarray,
    ref_plane_crater: np.ndarray,
    ref_mad_scale: float,
) -> Dict[str, Any]:
    """Single-pass quantile cavity residual statistics matching Phase 5A numerics identically."""
    finite_mask = np.isfinite(crater_patch) & np.isfinite(ref_plane_crater)
    valid_px = int(finite_mask.sum())
    total_px = int(crater_patch.size)
    if valid_px == 0:
        return {
            "valid_crater_px": 0,
            "finite_crater_ratio": 0.0,
            "signed_residual_map": np.full_like(crater_patch, np.nan, dtype=np.float64),
            "median_residual": 0.0,
            "iqr_residual": 0.0,
            "mad_residual": 0.0,
            "p75_residual": 0.0,
            "p85_residual": 0.0,
            "p90_residual": 0.0,
            "positive_depression_frac": 0.0,
            "above_noise_depression_frac": 0.0,
            "scale_normalized_p75": 0.0,
        }

    res_map = ref_plane_crater - crater_patch
    r_vals = res_map if valid_px == total_px else res_map[finite_mask]
    p25_r, med_r, p75_r, p85_r, p90_r = [
        float(v) for v in np.percentile(r_vals, [25.0, 50.0, 75.0, 85.0, 90.0])
    ]
    mad_r = float(np.median(np.abs(r_vals - med_r)))
    pos_frac = float(np.mean(r_vals > 0.0))
    noise_floor = max(1e-4, 1.4826 * ref_mad_scale)
    strong_frac = float(np.mean(r_vals > noise_floor))

    return {
        "valid_crater_px": valid_px,
        "finite_crater_ratio": round(valid_px / float(max(1, total_px)), 4),
        "signed_residual_map": res_map,
        "median_residual": round(med_r, 6),
        "iqr_residual": round(p75_r - p25_r, 6),
        "mad_residual": round(mad_r, 6),
        "p75_residual": round(p75_r, 6),
        "p85_residual": round(p85_r, 6),
        "p90_residual": round(p90_r, 6),
        "positive_depression_frac": round(pos_frac, 4),
        "above_noise_depression_frac": round(strong_frac, 4),
        "scale_normalized_p75": round(p75_r / noise_floor, 4),
    }


def assess_quality_fast(
    region: Dict[str, Any],
    fit_out: Dict[str, Any],
    cav_stats: Dict[str, Any],
    is_water: bool,
    cfg: Optional[OptimizedDepthConfig] = None,
) -> Dict[str, Any]:
    """Identical 11-point Phase 5A quality assessment gates without gate relaxation."""
    cfg = cfg or OptimizedDepthConfig()
    reasons: List[str] = []
    warnings: List[str] = []

    ref_res = fit_out["ref_residuals"]
    if len(ref_res) > 0:
        ref_med = float(np.median(ref_res))
        ref_mad = float(np.median(np.abs(ref_res - ref_med)))
        ref_rmse = float(np.sqrt(np.mean(ref_res ** 2)))
    else:
        ref_mad = 0.0
        ref_rmse = 0.0

    # 1. Invalid / non-finite depth
    if (
        cav_stats["finite_crater_ratio"] < 0.90
        or cav_stats["valid_crater_px"] < 16
        or region["nonfinite_ring_px"] > max(10, int(0.10 * region["ideal_ring_px"]))
    ):
        reasons.append("invalid_or_nonfinite_depth")

    # 2. Insufficient reference samples
    if fit_out["used_ref_samples"] < cfg.min_ref_samples:
        reasons.append("insufficient_reference_samples")

    # 3. Poor spatial support (checked on actual sampled reference points!)
    if region["populated_quadrants"] < cfg.min_populated_quadrants or region["max_quadrant_frac"] > cfg.max_single_quadrant_frac:
        reasons.append("poor_spatial_support")

    # 4. Degenerate reference plane
    if fit_out["plane_degenerate"]:
        reasons.append("degenerate_reference_plane")

    # 5. Low robust inlier ratio
    if fit_out["inlier_ratio"] < cfg.min_inlier_ratio:
        reasons.append("low_inlier_ratio")

    # 6. Overlapping neighbors starving pavement
    if region["annular_retention_ratio"] < cfg.min_annular_retention:
        reasons.append("overlapping_neighbors_insufficient_pavement")
    elif region["neighbor_excluded_px"] > 0:
        warnings.append("neighbors_excluded_from_ring")

    # 7. Excessive reference residual noise
    if ref_mad > cfg.max_ref_mad_abs:
        reasons.append("high_reference_fit_residual")

    # 8. Unexpected residual polarity
    noise_eps = max(1e-5, 0.5 * 1.4826 * ref_mad)
    r_map = cav_stats["signed_residual_map"]
    fin_r = r_map if cav_stats["valid_crater_px"] == r_map.size else r_map[np.isfinite(r_map)]
    pos_strong_frac = float(np.mean(fin_r > noise_eps)) if len(fin_r) > 0 else 0.0
    neg_strong_frac = float(np.mean(fin_r < -noise_eps)) if len(fin_r) > 0 else 0.0
    if (
        cav_stats["p75_residual"] <= noise_eps
        or pos_strong_frac < 0.22
        or (neg_strong_frac > pos_strong_frac and cav_stats["p75_residual"] <= 2.0 * noise_eps)
    ):
        reasons.append("unexpected_residual_polarity")
    # 9. Unstable cavity residuals
    elif (
        cav_stats["iqr_residual"] > max(cfg.max_ref_mad_vs_iqr_ratio * max(cav_stats["p75_residual"], 1e-4), 1.0)
        and cav_stats["positive_depression_frac"] < 0.50
    ):
        reasons.append("unstable_cavity_residuals")

    # 10. Frame boundary truncation
    if region["truncated_edges"]:
        warnings.append(f"boundary_truncated:{'+'.join(region['truncated_edges'])}")
        if (
            len(region["truncated_edges"]) >= 2
            and region["populated_quadrants"] < cfg.min_populated_quadrants
            and "poor_spatial_support" not in reasons
        ):
            reasons.append("frame_boundary_truncation")

    # 11. Specular water patch
    if is_water:
        warnings.append("suspected_water_reflection")
        if cfg.reject_water_patches:
            reasons.append("suspected_water_reflection")

    is_valid = len(reasons) == 0
    if not is_valid:
        heur_score = 0.0
    else:
        quad_factor = min(1.0, region["populated_quadrants"] / 4.0)
        inl_factor = float(fit_out["inlier_ratio"])
        pol_factor = min(1.0, cav_stats["positive_depression_frac"] / 0.80)
        snr_factor = min(1.0, cav_stats["scale_normalized_p75"] / 4.0)
        heur_score = round(0.25 * quad_factor + 0.25 * inl_factor + 0.25 * pol_factor + 0.25 * snr_factor, 4)

    return {
        "is_valid": is_valid,
        "failure_reasons": reasons,
        "warnings": warnings,
        "heuristic_reliability_score": heur_score,
        "ref_fit_mad": round(ref_mad, 6),
        "ref_fit_rmse": round(ref_rmse, 6),
    }


def extract_optimized_relative_depth_for_box(
    depth_map: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    neighbor_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    img_bgr: Optional[np.ndarray] = None,
    cfg: Optional[OptimizedDepthConfig] = None,
    return_arrays: bool = False,
) -> Dict[str, Any]:
    """Optimized full relative-depth extraction pipeline for one candidate pothole box."""
    cfg = cfg or OptimizedDepthConfig()
    t0 = time.perf_counter()
    region = build_and_sample_annular_region(depth_map, box_xyxy, neighbor_boxes=neighbor_boxes, cfg=cfg)
    t_reg = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    fit_out = fit_ransac_optimized(depth_map, region, cfg=cfg)
    t_fit = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    x1, y1, x2, y2 = region["box_xyxy"]
    crater_patch = depth_map[y1:y2, x1:x2].astype(np.float64)
    ref_res = fit_out["ref_residuals"]
    ref_mad = float(np.median(np.abs(ref_res - np.median(ref_res)))) if len(ref_res) > 0 else 0.0
    cav_stats = compute_cavity_statistics_fast(crater_patch, fit_out["ref_plane_crater"], ref_mad_scale=ref_mad)
    t_res = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    is_water = False
    if img_bgr is not None and x2 > x1 and y2 > y1:
        is_water = SpecularAquaplaneDetector.inspect_patch(img_bgr[y1:y2, x1:x2])
    qa = assess_quality_fast(region, fit_out, cav_stats, is_water=is_water, cfg=cfg)
    t_qa = (time.perf_counter() - t3) * 1000.0

    result = {
        "method": "optimized_ransac",
        "sampling_mode": cfg.sampling_mode,
        "sampling_budget": cfg.max_ref_samples,
        "box_xyxy": list(region["box_xyxy"]),
        "outer_xyxy": list(region["outer_xyxy"]),
        "is_valid": qa["is_valid"],
        "failure_reasons": qa["failure_reasons"],
        "warnings": qa["warnings"],
        "heuristic_reliability_score": qa["heuristic_reliability_score"],
        "is_water_suspected": is_water,
        "relative_cavity_stats": {
            "median_residual": cav_stats["median_residual"] if qa["is_valid"] else None,
            "iqr_residual": cav_stats["iqr_residual"] if qa["is_valid"] else None,
            "mad_residual": cav_stats["mad_residual"] if qa["is_valid"] else None,
            "p75_residual": cav_stats["p75_residual"] if qa["is_valid"] else None,
            "p85_residual": cav_stats["p85_residual"] if qa["is_valid"] else None,
            "p90_residual": cav_stats["p90_residual"] if qa["is_valid"] else None,
            "positive_depression_frac": cav_stats["positive_depression_frac"],
            "above_noise_depression_frac": cav_stats["above_noise_depression_frac"],
            "scale_normalized_p75": cav_stats["scale_normalized_p75"] if qa["is_valid"] else None,
            "raw_diagnostic_p75": cav_stats["p75_residual"],
            "raw_diagnostic_median": cav_stats["median_residual"],
        },
        "ref_diagnostics": {
            "available_ref_samples": region["available_ref_samples"],
            "valid_ref_samples": region["sampled_ref_samples"],
            "used_ref_samples": fit_out["used_ref_samples"],
            "sampling_coverage_ratio": region["sampling_coverage_ratio"],
            "annular_retention_ratio": region["annular_retention_ratio"],
            "neighbor_excluded_px": region["neighbor_excluded_px"],
            "quadrant_counts": region["quadrant_counts"],
            "full_quadrant_counts": region["full_quadrant_counts"],
            "populated_quadrants": region["populated_quadrants"],
            "inlier_ratio": fit_out["inlier_ratio"],
            "ref_fit_mad": qa["ref_fit_mad"],
            "ref_fit_rmse": qa["ref_fit_rmse"],
            "cond_number": fit_out["cond_number"],
            "plane_degenerate": fit_out["plane_degenerate"],
            "coeffs": fit_out["coeffs"],
            "iters_evaluated": fit_out["iters_evaluated"],
            "valid_hypotheses_evaluated": fit_out["valid_hypotheses_evaluated"],
            "degenerate_triplets_rejected": fit_out["degenerate_triplets_rejected"],
            "early_stopped": fit_out["early_stopped"],
        },
        "timing_ms": {
            "annular_region_ms": round(t_reg, 4),
            "ref_fit_ms": round(t_fit, 4),
            "residual_compute_ms": round(t_res, 4),
            "quality_eval_ms": round(t_qa, 4),
            "extraction_total_ms": round(t_reg + t_fit + t_res + t_qa, 4),
        },
    }
    if return_arrays:
        result["arrays"] = {
            "region": region,
            "fit_out": fit_out,
            "signed_residual_map": cav_stats["signed_residual_map"],
            "ref_plane_crater": fit_out["ref_plane_crater"],
        }
    return result
