"""
========================================================================================
RoadVision AI -- Phase 5A: Robust Relative Depth Extraction & Adaptive Road Reference
========================================================================================
Experimental research module (completely independent from production roadvision_engine.py
and server.py). Compares five relative inverse-depth reference-surface estimation strategies:
  A. median_baseline   : Surrounding-region scalar median
  B. existing_annular  : Exact legacy AnnularDonutDepthExtractor logic (diagnostic baseline)
  C. ols_plane         : Adaptive annular sampling + normalized OLS plane fit
  D. ransac_plane      : Adaptive annular sampling + deterministic RANSAC plane fit
  E. huber_irls_plane  : Adaptive annular sampling + Huber IRLS robust plane fit

SCIENTIFIC CONVENTIONS & DISCLAIMERS:
  - Input depth maps are MiDaS relative inverse depth (higher = closer, lower = deeper).
  - Fitted reference planes live in 2D normalized inverse-depth image space (z = a*y_n + b*x_n + c),
    NOT calibrated 3D metric coordinates.
  - Residuals and scale-normalized scores are relative units ONLY — NEVER converted to cm.
  - Heuristic reliability scores are NOT calibrated probabilities of physical correctness.
  - Invalid measurements keep explicit `is_valid=False` and failure reasons (never silent 0.0).
========================================================================================
"""

import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from roadvision_engine import AnnularDonutDepthExtractor, SpecularAquaplaneDetector

ALL_METHODS = (
    "median_baseline",
    "existing_annular",
    "ols_plane",
    "ransac_plane",
    "huber_irls_plane",
)


@dataclass
class RobustDepthConfig:
    outer_margin_ratio: float = 0.45
    min_margin_px: int = 16
    max_margin_px: int = 120
    inner_pad_ratio: float = 0.05
    exclude_neighbors: bool = True
    neighbor_pad_px: int = 4
    horizon_cutoff_ratio: float = 0.15

    # Reference fitting
    ransac_iters: int = 120
    ransac_sample_limit: int = 1500
    ransac_mad_mult: float = 2.5
    huber_max_iters: int = 20
    huber_k: float = 1.345
    seed: int = 42

    # Quality assessment gates
    min_ref_samples: int = 40
    min_quadrant_samples: int = 5
    min_populated_quadrants: int = 2
    max_single_quadrant_frac: float = 0.80
    max_cond_number: float = 1e4
    min_annular_retention: float = 0.25
    min_inlier_ratio: float = 0.45
    min_positive_depress_frac: float = 0.35
    max_ref_mad_abs: float = 1e6  # Optional upper bound in raw units; relative check used primarily
    max_ref_mad_vs_iqr_ratio: float = 3.0
    reject_water_patches: bool = True
    boundary_trunc_px: int = 2


class MiDaSDepthRunner:
    """Isolated MiDaS_small wrapper fixing BGR->RGB ordering and tracking CUDA latency."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        torch.hub._check_repo_is_trusted = lambda *args, **kwargs: True
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.startup_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def _sync(self) -> None:
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()

    def warmup(self, h: int = 360, w: int = 640) -> None:
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        _ = self.predict(dummy)

    def predict(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._sync()
        t0 = time.perf_counter()
        inp = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = self.model(inp)
            else:
                pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze()
        self._sync()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return pred.cpu().numpy().astype(np.float32), round(dt_ms, 3)


def build_adaptive_annular_region(
    depth_map: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    neighbor_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    cfg: Optional[RobustDepthConfig] = None,
) -> Dict[str, Any]:
    """Construct an adaptive annular road reference mask excluding crater core, neighbors, and invalid pixels."""
    cfg = cfg or RobustDepthConfig()
    H, W = depth_map.shape[:2]
    x1 = max(0, int(np.floor(box_xyxy[0])))
    y1 = max(0, int(np.floor(box_xyxy[1])))
    x2 = min(W, int(np.ceil(box_xyxy[2])))
    y2 = min(H, int(np.ceil(box_xyxy[3])))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Inner crater region + slight safety padding around rim for clean road sampling
    ip_x = max(0, int(round(bw * cfg.inner_pad_ratio)))
    ip_y = max(0, int(round(bh * cfg.inner_pad_ratio)))
    ex_x1 = max(0, x1 - ip_x)
    ex_y1 = max(0, y1 - ip_y)
    ex_x2 = min(W, x2 + ip_x)
    ex_y2 = min(H, y2 + ip_y)

    # Adaptive outer annular expansion (unbounded below max_margin_px instead of legacy 25px cap)
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

    # Exclude target crater box (plus inner pad)
    iy1_pad, iy2_pad = max(0, ex_y1 - ry1), min(h_out, ex_y2 - ry1)
    ix1_pad, ix2_pad = max(0, ex_x1 - rx1), min(w_out, ex_x2 - rx1)
    ideal_outer[iy1_pad:iy2_pad, ix1_pad:ix2_pad] = False
    ideal_ring_px = max(1, int(ideal_outer.sum()))

    valid_ring = ideal_outer.copy()
    neighbor_excluded_px = 0
    if cfg.exclude_neighbors and neighbor_boxes:
        for nb in neighbor_boxes:
            nx1 = max(0, int(np.floor(nb[0])) - cfg.neighbor_pad_px) - rx1
            ny1 = max(0, int(np.floor(nb[1])) - cfg.neighbor_pad_px) - ry1
            nx2 = min(W, int(np.ceil(nb[2])) + cfg.neighbor_pad_px) - rx1
            ny2 = min(H, int(np.ceil(nb[3])) + cfg.neighbor_pad_px) - ry1
            cx1, cy1 = max(0, nx1), max(0, ny1)
            cx2, cy2 = min(w_out, nx2), min(h_out, ny2)
            if cx2 > cx1 and cy2 > cy1:
                before = int(valid_ring.sum())
                valid_ring[cy1:cy2, cx1:cx2] = False
                neighbor_excluded_px += before - int(valid_ring.sum())

    outer_patch = depth_map[ry1:ry2, rx1:rx2]
    finite_outer = np.isfinite(outer_patch)
    nonfinite_ring_px = int(np.logical_and(valid_ring, ~finite_outer).sum())
    valid_ring &= finite_outer

    # Spatial distribution around target box center (Top, Bottom, Left, Right regions)
    cy_local = ((y1 + y2) * 0.5) - ry1
    cx_local = ((x1 + x2) * 0.5) - rx1
    ys_r, xs_r = np.where(valid_ring)
    n_valid = len(ys_r)

    if n_valid > 0:
        dy_c = ys_r - cy_local
        dx_c = xs_r - cx_local
        # Partition into 4 angular/axial sectors relative to box center
        q_top = int(np.sum((dy_c < 0) & (np.abs(dy_c) >= np.abs(dx_c))))
        q_bot = int(np.sum((dy_c >= 0) & (np.abs(dy_c) >= np.abs(dx_c))))
        q_left = int(np.sum((dx_c < 0) & (np.abs(dx_c) > np.abs(dy_c))))
        q_right = int(np.sum((dx_c >= 0) & (np.abs(dx_c) > np.abs(dy_c))))
        quad_counts = [q_top, q_bot, q_left, q_right]
        pop_quads = sum(1 for q in quad_counts if q >= cfg.min_quadrant_samples)
        max_quad_frac = max(quad_counts) / float(n_valid)
    else:
        quad_counts = [0, 0, 0, 0]
        pop_quads = 0
        max_quad_frac = 1.0

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
        "inner_local_xyxy": (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1),
        "margins_px": (mx, my),
        "ideal_outer_mask": ideal_outer,
        "valid_ring_mask": valid_ring,
        "ys_local": ys_r,
        "xs_local": xs_r,
        "valid_ref_samples": n_valid,
        "ideal_ring_px": ideal_ring_px,
        "annular_retention_ratio": round(n_valid / float(ideal_ring_px), 4),
        "neighbor_excluded_px": neighbor_excluded_px,
        "nonfinite_ring_px": nonfinite_ring_px,
        "quadrant_counts": quad_counts,
        "populated_quadrants": pop_quads,
        "max_quadrant_frac": round(max_quad_frac, 4),
        "truncated_edges": truncated_edges,
    }


def _estimate_local_surface_sigma(depth_map: np.ndarray, outer_xyxy: Tuple[int, int, int, int], valid_ring: np.ndarray) -> float:
    """Estimate intrinsic pixel-to-pixel surface noise from neighbor finite differences (immune to step blocks)."""
    rx1, ry1, rx2, ry2 = outer_xyxy
    patch = depth_map[ry1:ry2, rx1:rx2].astype(np.float64)
    diffs = []
    if patch.shape[0] >= 2:
        vm_y = valid_ring[1:, :] & valid_ring[:-1, :]
        if np.any(vm_y):
            diffs.append(np.abs(patch[1:, :][vm_y] - patch[:-1, :][vm_y]))
    if patch.shape[1] >= 2:
        vm_x = valid_ring[:, 1:] & valid_ring[:, :-1]
        if np.any(vm_x):
            diffs.append(np.abs(patch[:, 1:][vm_x] - patch[:, :-1][vm_x]))
    if not diffs:
        return 1e-3
    all_d = np.concatenate(diffs)
    # 1.4826 * median(|diff|) / sqrt(2) ~ sigma for additive Gaussian noise, immune to sparse step boundaries
    return max(1e-4, 1.048 * float(np.median(all_d)))


def _normalize_coords(ys: np.ndarray, xs: np.ndarray, h_out: int, w_out: int) -> Tuple[np.ndarray, np.ndarray]:
    yn = (2.0 * ys.astype(np.float64) / max(1.0, float(h_out - 1))) - 1.0
    xn = (2.0 * xs.astype(np.float64) / max(1.0, float(w_out - 1))) - 1.0
    return yn, xn


def fit_reference_surface(
    method: str,
    depth_map: np.ndarray,
    region: Dict[str, Any],
    cfg: Optional[RobustDepthConfig] = None,
) -> Dict[str, Any]:
    """
    Fit relative inverse-depth reference surface using one of 5 strategies.
    Returns fitted coefficients, reference prediction over target crater box,
    reference fit residuals, inlier mask, and numerical conditioning diagnostics.
    """
    cfg = cfg or RobustDepthConfig()
    rx1, ry1, rx2, ry2 = region["outer_xyxy"]
    x1, y1, x2, y2 = region["box_xyxy"]
    h_out, w_out = ry2 - ry1, rx2 - rx1

    crater_patch = depth_map[y1:y2, x1:x2].astype(np.float64)
    iy_local, ix_local = np.indices(crater_patch.shape)
    iy_outer = (y1 - ry1) + iy_local
    ix_outer = (x1 - rx1) + ix_local

    # Legacy Method B uses exact production AnnularDonutDepthExtractor geometry (no neighbor exclusion/norm)
    if method == "existing_annular":
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        mx = max(15, min(25, int(bw * 0.25)))
        my = max(15, min(25, int(bh * 0.25)))
        H, W = depth_map.shape[:2]
        lry1, lry2 = max(0, y1 - my), min(H, y2 + my)
        lrx1, lrx2 = max(0, x1 - mx), min(W, x2 + mx)
        outer_reg = depth_map[lry1:lry2, lrx1:lrx2].astype(np.float64)
        dmask = np.ones(outer_reg.shape, dtype=bool)
        dmask[(y1 - lry1):(y2 - lry1), (x1 - lrx1):(x2 - lrx1)] = False
        ys_l, xs_l = np.where(dmask)
        finite_l = np.isfinite(outer_reg[ys_l, xs_l])
        ys_l, xs_l = ys_l[finite_l], xs_l[finite_l]
        z_l = outer_reg[ys_l, xs_l]

        legacy_raw_val = AnnularDonutDepthExtractor.extract_depth_disparity(depth_map, x1, y1, x2, y2, margin=25)
        if len(ys_l) >= 20:
            A = np.column_stack([ys_l.astype(np.float64), xs_l.astype(np.float64), np.ones_like(ys_l, dtype=np.float64)])
            try:
                coeffs, _, rank, svals = np.linalg.lstsq(A, z_l, rcond=None)
                cond = float(svals[0] / max(1e-12, svals[-1])) if len(svals) == 3 and svals[-1] > 1e-12 else 1e9
                pred_ref_ring = A @ coeffs
                ref_res = z_l - pred_ref_ring
                ref_plane_crater = (
                    coeffs[0] * ((y1 - lry1) + iy_local) + coeffs[1] * ((x1 - lrx1) + ix_local) + coeffs[2]
                )
                return {
                    "method": method,
                    "ref_plane_crater": ref_plane_crater,
                    "ref_residuals": ref_res,
                    "inlier_mask": np.ones(len(z_l), dtype=bool),
                    "inlier_ratio": 1.0,
                    "rank": int(rank),
                    "cond_number": round(cond, 2),
                    "plane_degenerate": rank < 3 or cond > cfg.max_cond_number,
                    "used_ref_samples": len(z_l),
                    "coeffs": [round(float(c), 6) for c in coeffs],
                    "legacy_raw_p85": round(float(legacy_raw_val), 6),
                }
            except Exception:
                pass
        med_z = float(np.median(z_l)) if len(z_l) > 0 else 0.0
        return {
            "method": method,
            "ref_plane_crater": np.full_like(crater_patch, med_z, dtype=np.float64),
            "ref_residuals": (z_l - med_z) if len(z_l) > 0 else np.array([], dtype=np.float64),
            "inlier_mask": np.ones(len(z_l), dtype=bool),
            "inlier_ratio": 1.0 if len(z_l) > 0 else 0.0,
            "rank": 1,
            "cond_number": 1.0,
            "plane_degenerate": len(z_l) == 0,
            "used_ref_samples": len(z_l),
            "coeffs": [0.0, 0.0, round(med_z, 6)],
            "legacy_raw_p85": round(float(legacy_raw_val), 6),
        }

    # All modern adaptive methods (A, C, D, E) share the neighbor-excluded adaptive annular samples
    ys = region["ys_local"]
    xs = region["xs_local"]
    outer_patch = depth_map[ry1:ry2, rx1:rx2].astype(np.float64)
    z_vals = outer_patch[ys, xs]
    n_pts = len(z_vals)

    if n_pts == 0:
        return {
            "method": method,
            "ref_plane_crater": np.zeros_like(crater_patch, dtype=np.float64),
            "ref_residuals": np.array([], dtype=np.float64),
            "inlier_mask": np.array([], dtype=bool),
            "inlier_ratio": 0.0,
            "rank": 0,
            "cond_number": 1e9,
            "plane_degenerate": True,
            "used_ref_samples": 0,
            "coeffs": [0.0, 0.0, 0.0],
        }

    yn, xn = _normalize_coords(ys, xs, h_out, w_out)
    cy_n, cx_n = _normalize_coords(iy_outer, ix_outer, h_out, w_out)

    if method == "median_baseline":
        med_z = float(np.median(z_vals))
        res = z_vals - med_z
        mad = float(np.median(np.abs(res)))
        tol = max(1e-4, cfg.ransac_mad_mult * 1.4826 * mad)
        inl = np.abs(res) <= tol
        return {
            "method": method,
            "ref_plane_crater": np.full_like(crater_patch, med_z, dtype=np.float64),
            "ref_residuals": res,
            "inlier_mask": inl,
            "inlier_ratio": round(float(inl.mean()), 4),
            "rank": 1,
            "cond_number": 1.0,
            "plane_degenerate": False,
            "used_ref_samples": n_pts,
            "coeffs": [0.0, 0.0, round(med_z, 6)],
        }

    A = np.column_stack([yn, xn, np.ones_like(yn)])
    if n_pts < 3:
        med_z = float(np.median(z_vals))
        return {
            "method": method,
            "ref_plane_crater": np.full_like(crater_patch, med_z, dtype=np.float64),
            "ref_residuals": z_vals - med_z,
            "inlier_mask": np.ones(n_pts, dtype=bool),
            "inlier_ratio": 1.0,
            "rank": 1,
            "cond_number": 1e9,
            "plane_degenerate": True,
            "used_ref_samples": n_pts,
            "coeffs": [0.0, 0.0, round(med_z, 6)],
        }

    coeffs_ols, _, rank_ols, svals = np.linalg.lstsq(A, z_vals, rcond=None)
    cond_ols = float(svals[0] / max(1e-12, svals[-1])) if len(svals) == 3 and svals[-1] > 1e-12 else 1e9
    if rank_ols < 3 or cond_ols > cfg.max_cond_number:
        med_z = float(np.median(z_vals))
        return {
            "method": method,
            "ref_plane_crater": np.full_like(crater_patch, med_z, dtype=np.float64),
            "ref_residuals": z_vals - med_z,
            "inlier_mask": np.ones(n_pts, dtype=bool),
            "inlier_ratio": 1.0,
            "rank": int(rank_ols),
            "cond_number": round(cond_ols, 2),
            "plane_degenerate": True,
            "used_ref_samples": n_pts,
            "coeffs": [0.0, 0.0, round(med_z, 6)],
        }

    if method == "ols_plane":
        res = z_vals - (A @ coeffs_ols)
        mad = float(np.median(np.abs(res - np.median(res))))
        tol = max(1e-4, cfg.ransac_mad_mult * 1.4826 * mad)
        inl = np.abs(res) <= tol
        plane = coeffs_ols[0] * cy_n + coeffs_ols[1] * cx_n + coeffs_ols[2]
        return {
            "method": method,
            "ref_plane_crater": plane,
            "ref_residuals": res,
            "inlier_mask": inl,
            "inlier_ratio": round(float(inl.mean()), 4),
            "rank": int(rank_ols),
            "cond_number": round(cond_ols, 2),
            "plane_degenerate": False,
            "used_ref_samples": n_pts,
            "coeffs": [round(float(c), 6) for c in coeffs_ols],
        }

    if method == "ransac_plane":
        rng = np.random.default_rng(cfg.seed)
        if n_pts > cfg.ransac_sample_limit:
            sub_idx = rng.choice(n_pts, size=cfg.ransac_sample_limit, replace=False)
            A_sub, z_sub = A[sub_idx], z_vals[sub_idx]
        else:
            A_sub, z_sub = A, z_vals

        local_sigma = _estimate_local_surface_sigma(depth_map, (rx1, ry1, rx2, ry2), region["valid_ring_mask"])
        z_iqr = float(np.percentile(z_sub, 75) - np.percentile(z_sub, 25))
        thresh = max(1e-3, 3.0 * local_sigma, 0.005 * z_iqr)

        best_inl = None
        best_count = -1
        best_err_sum = float("inf")
        n_sub = len(z_sub)

        for _ in range(cfg.ransac_iters):
            tri = rng.choice(n_sub, size=3, replace=False)
            M3 = A_sub[tri]
            det2 = abs(float((M3[1, 1] - M3[0, 1]) * (M3[2, 0] - M3[0, 0]) - (M3[2, 1] - M3[0, 1]) * (M3[1, 0] - M3[0, 0])))
            if det2 < 1e-4:
                continue
            try:
                c_cand = np.linalg.solve(M3, z_sub[tri])
            except np.linalg.LinAlgError:
                continue
            r_cand = np.abs(z_sub - (A_sub @ c_cand))
            inl_sub = r_cand <= thresh
            cnt = int(inl_sub.sum())
            err_sum = float(np.minimum(r_cand, thresh).sum())
            if cnt > best_count or (cnt == best_count and err_sum < best_err_sum):
                best_count = cnt
                best_err_sum = err_sum
                best_inl = inl_sub

        if best_inl is None or best_count < 6:
            coeffs_fin = coeffs_ols
            inl_full = np.ones(n_pts, dtype=bool)
            degen = True
        else:
            M3_fin, _, _, _ = np.linalg.lstsq(A_sub[best_inl], z_sub[best_inl], rcond=None)
            r_full = np.abs(z_vals - (A @ M3_fin))
            inlier_r = r_full[r_full <= thresh * 1.5]
            mad_inl = float(np.median(inlier_r)) if len(inlier_r) > 0 else local_sigma
            ref_thresh = max(thresh, cfg.ransac_mad_mult * 1.4826 * mad_inl)
            inl_full = r_full <= ref_thresh
            if int(inl_full.sum()) >= 6:
                coeffs_fin, _, r_fin, svals_fin = np.linalg.lstsq(A[inl_full], z_vals[inl_full], rcond=None)
                cond_fin = float(svals_fin[0] / max(1e-12, svals_fin[-1])) if len(svals_fin) == 3 and svals_fin[-1] > 1e-12 else 1e9
                degen = r_fin < 3 or cond_fin > cfg.max_cond_number
                if degen:
                    coeffs_fin = coeffs_ols
            else:
                coeffs_fin = M3_fin
                degen = True

        res_fin = z_vals - (A @ coeffs_fin)
        plane = coeffs_fin[0] * cy_n + coeffs_fin[1] * cx_n + coeffs_fin[2]
        return {
            "method": method,
            "ref_plane_crater": plane,
            "ref_residuals": res_fin,
            "inlier_mask": inl_full,
            "inlier_ratio": round(float(inl_full.mean()), 4),
            "rank": int(rank_ols),
            "cond_number": round(cond_ols, 2),
            "plane_degenerate": degen,
            "used_ref_samples": n_pts,
            "coeffs": [round(float(c), 6) for c in coeffs_fin],
        }

    if method == "huber_irls_plane":
        # Fast L1-minimizing 3-point warm-start so step curb blocks cannot lever OLS initialization
        rng_h = np.random.default_rng(cfg.seed + 1)
        if n_pts > cfg.ransac_sample_limit:
            sub_i = rng_h.choice(n_pts, size=cfg.ransac_sample_limit, replace=False)
            A_s, z_s = A[sub_i], z_vals[sub_i]
        else:
            A_s, z_s = A, z_vals

        coeffs = coeffs_ols.copy()
        best_l1 = float(np.sum(np.abs(z_s - (A_s @ coeffs))))
        for _ in range(32):
            tri = rng_h.choice(len(z_s), size=3, replace=False)
            M3 = A_s[tri]
            det2 = abs(float((M3[1, 1] - M3[0, 1]) * (M3[2, 0] - M3[0, 0]) - (M3[2, 1] - M3[0, 1]) * (M3[1, 0] - M3[0, 0])))
            if det2 < 1e-4:
                continue
            try:
                c_cand = np.linalg.solve(M3, z_s[tri])
            except np.linalg.LinAlgError:
                continue
            l1 = float(np.sum(np.abs(z_s - (A_s @ c_cand))))
            if l1 < best_l1:
                best_l1 = l1
                coeffs = c_cand

        local_sigma = _estimate_local_surface_sigma(depth_map, (rx1, ry1, rx2, ry2), region["valid_ring_mask"])
        degen = False

        for _ in range(cfg.huber_max_iters):
            res = z_vals - (A @ coeffs)
            mad = float(np.median(np.abs(res - np.median(res))))
            sigma = max(local_sigma, 1e-4, 1.4826 * mad)
            delta = cfg.huber_k * sigma
            abs_r = np.abs(res)
            w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-9))

            sw = np.sqrt(w)[:, None]
            Aw = A * sw
            zw = z_vals * np.sqrt(w)
            new_coeffs, _, r_w, svals_w = np.linalg.lstsq(Aw, zw, rcond=None)
            cond_w = float(svals_w[0] / max(1e-12, svals_w[-1])) if len(svals_w) == 3 and svals_w[-1] > 1e-12 else 1e9
            if r_w < 3 or cond_w > cfg.max_cond_number:
                degen = True
                break
            diff = float(np.max(np.abs(new_coeffs - coeffs)))
            coeffs = new_coeffs
            if diff < 1e-5:
                break

        res_fin = z_vals - (A @ coeffs)
        mad_fin = float(np.median(np.abs(res_fin - np.median(res_fin))))
        tol_fin = max(1e-3, cfg.ransac_mad_mult * max(local_sigma, 1.4826 * mad_fin))
        inl_fin = np.abs(res_fin) <= tol_fin
        plane = coeffs[0] * cy_n + coeffs[1] * cx_n + coeffs[2]
        return {
            "method": method,
            "ref_plane_crater": plane,
            "ref_residuals": res_fin,
            "inlier_mask": inl_fin,
            "inlier_ratio": round(float(inl_fin.mean()), 4),
            "rank": int(rank_ols),
            "cond_number": round(cond_ols, 2),
            "plane_degenerate": degen,
            "used_ref_samples": n_pts,
            "coeffs": [round(float(c), 6) for c in coeffs],
        }

    raise ValueError(f"Unknown reference fitting method: {method}")


def compute_cavity_statistics(
    crater_patch: np.ndarray,
    ref_plane_crater: np.ndarray,
    ref_mad_scale: float,
) -> Dict[str, Any]:
    """
    Compute robust signed cavity depression statistics in relative inverse-depth units.
    Positive residual = ref_plane - depth > 0 (deeper than surrounding road plane).
    """
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
    r_vals = res_map[finite_mask]
    med_r = float(np.median(r_vals))
    p25_r = float(np.percentile(r_vals, 25))
    p75_r = float(np.percentile(r_vals, 75))
    p85_r = float(np.percentile(r_vals, 85))
    p90_r = float(np.percentile(r_vals, 90))
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


def assess_measurement_quality(
    method: str,
    region: Dict[str, Any],
    fit_out: Dict[str, Any],
    cav_stats: Dict[str, Any],
    is_water: bool,
    cfg: Optional[RobustDepthConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate all 11 geometric, numerical, and physical quality checks.
    Failed measurements remain strictly `is_valid=False` (never replaced with zero depth).
    Heuristic reliability score is NOT a calibrated probability.
    """
    cfg = cfg or RobustDepthConfig()
    reasons: List[str] = []
    warnings: List[str] = []

    ref_res = fit_out["ref_residuals"]
    if len(ref_res) > 0:
        ref_med_res = float(np.median(ref_res))
        ref_mad = float(np.median(np.abs(ref_res - ref_med_res)))
        ref_rmse = float(np.sqrt(np.mean(ref_res ** 2)))
    else:
        ref_mad = 0.0
        ref_rmse = 0.0

    # 1. Invalid / non-finite depth values
    if cav_stats["finite_crater_ratio"] < 0.90 or cav_stats["valid_crater_px"] < 16 or region["nonfinite_ring_px"] > max(10, int(0.10 * region["ideal_ring_px"])):
        reasons.append("invalid_or_nonfinite_depth")

    # 2. Insufficient reference samples
    min_samp = 20 if method == "existing_annular" else cfg.min_ref_samples
    if fit_out["used_ref_samples"] < min_samp:
        reasons.append("insufficient_reference_samples")

    # 3. Poor spatial support (skip strict quadrant rejection on legacy method comparison only if checking support)
    if method != "existing_annular":
        if region["populated_quadrants"] < cfg.min_populated_quadrants or region["max_quadrant_frac"] > cfg.max_single_quadrant_frac:
            reasons.append("poor_spatial_support")

    # 4. Degenerate reference-plane estimation
    if fit_out["plane_degenerate"] and method not in ("median_baseline",):
        reasons.append("degenerate_reference_plane")

    # 5. Low RANSAC/robust inlier ratio
    if method in ("ransac_plane", "huber_irls_plane") and fit_out["inlier_ratio"] < cfg.min_inlier_ratio:
        reasons.append("low_inlier_ratio")

    # 6. Overlapping candidate regions starving surrounding pavement
    if method != "existing_annular" and region["annular_retention_ratio"] < cfg.min_annular_retention:
        reasons.append("overlapping_neighbors_insufficient_pavement")
    elif region["neighbor_excluded_px"] > 0:
        warnings.append("neighbors_excluded_from_ring")

    # 7. High reference-fit residual noise overwhelming signal
    if ref_mad > cfg.max_ref_mad_abs:
        reasons.append("high_reference_fit_residual")

    # 8. Unexpected residual polarity (no depression signature)
    noise_eps = max(1e-5, 0.5 * 1.4826 * ref_mad)
    r_map = cav_stats["signed_residual_map"]
    fin_r = r_map[np.isfinite(r_map)]
    pos_strong_frac = float(np.mean(fin_r > noise_eps)) if len(fin_r) > 0 else 0.0
    neg_strong_frac = float(np.mean(fin_r < -noise_eps)) if len(fin_r) > 0 else 0.0
    if cav_stats["p75_residual"] <= noise_eps or pos_strong_frac < 0.22 or (neg_strong_frac > pos_strong_frac and cav_stats["p75_residual"] <= 2.0 * noise_eps):
        reasons.append("unexpected_residual_polarity")

    # 9. Unstable cavity residuals (extreme spread relative to P75 signal when positive)
    elif cav_stats["iqr_residual"] > max(cfg.max_ref_mad_vs_iqr_ratio * max(cav_stats["p75_residual"], 1e-4), 1.0) and cav_stats["positive_depression_frac"] < 0.50:
        reasons.append("unstable_cavity_residuals")

    # 10. Frame-boundary truncation
    if region["truncated_edges"]:
        warnings.append(f"boundary_truncated:{'+'.join(region['truncated_edges'])}")
        if len(region["truncated_edges"]) >= 2 and region["populated_quadrants"] < cfg.min_populated_quadrants and "poor_spatial_support" not in reasons:
            reasons.append("frame_boundary_truncation")

    # 11. Suspected water / reflective puddle region
    if is_water:
        warnings.append("suspected_water_reflection")
        if cfg.reject_water_patches:
            reasons.append("suspected_water_reflection")

    is_valid = len(reasons) == 0

    # Heuristic reliability score in [0, 1] (deterministic summary ONLY; NOT calibrated probability)
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


def extract_relative_depth_for_box(
    depth_map: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    method: str = "huber_irls_plane",
    neighbor_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    img_bgr: Optional[np.ndarray] = None,
    cfg: Optional[RobustDepthConfig] = None,
    return_arrays: bool = False,
) -> Dict[str, Any]:
    """Full modular relative-depth extraction pipeline for one candidate pothole box."""
    cfg = cfg or RobustDepthConfig()
    t0 = time.perf_counter()
    region = build_adaptive_annular_region(depth_map, box_xyxy, neighbor_boxes=neighbor_boxes, cfg=cfg)
    t_reg = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    fit_out = fit_reference_surface(method, depth_map, region, cfg=cfg)
    t_fit = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    x1, y1, x2, y2 = region["box_xyxy"]
    crater_patch = depth_map[y1:y2, x1:x2].astype(np.float64)
    ref_res = fit_out["ref_residuals"]
    ref_mad = float(np.median(np.abs(ref_res - np.median(ref_res)))) if len(ref_res) > 0 else 0.0
    cav_stats = compute_cavity_statistics(crater_patch, fit_out["ref_plane_crater"], ref_mad_scale=ref_mad)
    t_res = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    is_water = False
    if img_bgr is not None and x2 > x1 and y2 > y1:
        is_water = SpecularAquaplaneDetector.inspect_patch(img_bgr[y1:y2, x1:x2])
    qa = assess_measurement_quality(method, region, fit_out, cav_stats, is_water=is_water, cfg=cfg)
    t_qa = (time.perf_counter() - t3) * 1000.0

    # Expose structured output; invalid measurements do NOT emit fabricated zero valid depths
    result = {
        "method": method,
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
            "valid_ref_samples": region["valid_ref_samples"],
            "used_ref_samples": fit_out["used_ref_samples"],
            "annular_retention_ratio": region["annular_retention_ratio"],
            "neighbor_excluded_px": region["neighbor_excluded_px"],
            "quadrant_counts": region["quadrant_counts"],
            "populated_quadrants": region["populated_quadrants"],
            "inlier_ratio": fit_out["inlier_ratio"],
            "ref_fit_mad": qa["ref_fit_mad"],
            "ref_fit_rmse": qa["ref_fit_rmse"],
            "cond_number": fit_out["cond_number"],
            "plane_degenerate": fit_out["plane_degenerate"],
            "coeffs": fit_out["coeffs"],
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
