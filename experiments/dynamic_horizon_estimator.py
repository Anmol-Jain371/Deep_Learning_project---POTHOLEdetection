"""
========================================================================================
RoadVision AI -- Phase 7B: Dynamic Vanishing-Point & Horizon-Proxy Reliability Estimator
========================================================================================
Lightweight classical computer-vision vanishing-point estimation, geometric conditioning,
reliability scoring (`VALID`, `LOW_CONFIDENCE`, `UNAVAILABLE`, `STALE`, `EXPIRED`), and
fallback policies (`A_fixed`, `B_dynamic_raw`, `C_dynamic_smoothed`, `D_dynamic_fallback`).

SCIENTIFIC BOUNDARIES:
  - A road-direction vanishing point is ONLY an image-space 1-point horizon proxy; it does
    NOT uniquely determine physical camera pitch, roll, or complete 3D horizon line without
    known camera intrinsics (`K`) and extrinsics (`R, t`).
  - Zero production file modifications (`roadvision_engine.py`, `server.py`, dashboard).
  - Heuristic reliability diagnostics are NOT calibrated probabilities of physical accuracy.
========================================================================================
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

HORIZON_STATES = (
    "VALID",
    "LOW_CONFIDENCE",
    "UNAVAILABLE",
    "STALE",
    "EXPIRED",
)

FALLBACK_POLICIES = (
    "A_fixed",
    "B_dynamic_raw",
    "C_dynamic_smoothed",
    "D_dynamic_fallback",
)


@dataclass
class HorizonEstimatorConfig:
    detector_mode: str = "lsd"                 # "lsd", "hough", or "hybrid"
    process_width: Optional[int] = 320         # Reduced processing width for low-cost CPU execution
    roi_top_ratio: float = 0.22
    roi_bottom_ratio: float = 0.92
    roi_left_ratio: float = 0.02
    roi_right_ratio: float = 0.98

    # Edge / Hough / LSD params
    gaussian_ksize: int = 5
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 28
    hough_min_len_ratio: float = 0.045
    hough_max_gap_ratio: float = 0.025
    min_line_len_ratio: float = 0.045

    # Perspective orientation filtering (degrees from horizontal)
    min_horiz_angle_deg: float = 12.0          # Reject horizontal lane bars / cracks / horizon wires
    max_vert_angle_deg: float = 80.0           # Reject vertical poles / buildings / tree trunks
    min_pair_angle_deg: float = 12.0           # Reject near-parallel intersections (poor conditioning)

    # RANSAC / MSAC vanishing point consensus
    ransac_iters: int = 120
    inlier_dist_norm: float = 0.022            # Orthogonal line-to-VP normalized distance threshold
    inlier_angle_deg: float = 5.0              # Max ray angle error at segment midpoint
    seed: int = 42

    # Reliability gates (Task 3)
    min_inliers_low_conf: int = 3
    min_inliers_valid: int = 5
    min_left_valid: int = 1
    min_right_valid: int = 1
    max_dispersion_valid: float = 0.055
    max_cond_valid: float = 28.0
    max_cond_hard: float = 120.0
    vp_x_min: float = -0.20
    vp_x_max: float = 1.20
    vp_y_min: float = 0.05
    vp_y_max: float = 0.68
    max_temporal_jump_valid: float = 0.10


@dataclass
class HorizonPolicyConfig:
    policy: str = "D_dynamic_fallback"
    fixed_horizon_ratio: float = 0.40
    ema_alpha: float = 0.25
    accept_low_conf_in_ema: bool = False
    max_stale_sec: float = 0.50
    max_expired_sec: float = 1.20


def _order_segment_norm(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Ensure (xt, yt) is the upper endpoint (yt <= yb) in normalized image coords."""
    if y1 <= y2:
        return x1, y1, x2, y2
    return x2, y2, x1, y1


def _fit_line_eq(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    """Unit-normal line equation a*x + b*y + c = 0 (a^2 + b^2 = 1)."""
    dx = x2 - x1
    dy = y2 - y1
    a = -dy
    b = dx
    nrm = math.hypot(a, b)
    if nrm <= 1e-12:
        return 0.0, 0.0, 0.0
    a /= nrm
    b /= nrm
    c = -(a * x1 + b * y1)
    return a, b, c


def detect_and_filter_lines(
    img_bgr: np.ndarray,
    cfg: HorizonEstimatorConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
    """
    Task 2: Classical preprocessing, Canny/HoughP and/or LSD detection, and perspective
    orientation filtering returning normalized [0,1]^2 line segments & timing breakdown.
    """
    H0, W0 = img_bgr.shape[:2]
    t0 = time.perf_counter()

    # Optional downscaled processing preserving aspect ratio
    if cfg.process_width is not None and W0 > cfg.process_width:
        pw = int(cfg.process_width)
        ph = max(64, int(round(H0 * (pw / float(W0)))))
        work = cv2.resize(img_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        pw, ph = W0, H0
        work = img_bgr

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    ry1 = max(0, int(round(ph * cfg.roi_top_ratio)))
    ry2 = min(ph, int(round(ph * cfg.roi_bottom_ratio)))
    rx1 = max(0, int(round(pw * cfg.roi_left_ratio)))
    rx2 = min(pw, int(round(pw * cfg.roi_right_ratio)))
    roi_gray = gray[ry1:ry2, rx1:rx2]
    rh, rw = roi_gray.shape[:2]
    t_pre = (time.perf_counter() - t0) * 1000.0

    t_edge_s = time.perf_counter()
    ksize = cfg.gaussian_ksize if cfg.gaussian_ksize % 2 == 1 else cfg.gaussian_ksize + 1
    blurred = cv2.GaussianBlur(roi_gray, (ksize, ksize), 0)
    raw_segs: List[Tuple[float, float, float, float, str]] = []

    if cfg.detector_mode in ("hough", "hybrid"):
        edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)
    else:
        edges = None
    t_edge = (time.perf_counter() - t_edge_s) * 1000.0

    t_line_s = time.perf_counter()
    if cfg.detector_mode in ("hough", "hybrid") and edges is not None and rh > 8 and rw > 8:
        diag_roi = math.hypot(rw, rh)
        min_len = max(8.0, cfg.hough_min_len_ratio * diag_roi)
        max_gap = max(3.0, cfg.hough_max_gap_ratio * diag_roi)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=cfg.hough_threshold,
                                minLineLength=min_len, maxLineGap=max_gap)
        if lines is not None:
            for ln in np.asarray(lines).reshape(-1, 4):
                x1, y1, x2, y2 = float(ln[0] + rx1), float(ln[1] + ry1), float(ln[2] + rx1), float(ln[3] + ry1)
                raw_segs.append((x1 / pw, y1 / ph, x2 / pw, y2 / ph, "hough"))

    if cfg.detector_mode in ("lsd", "hybrid") and rh > 8 and rw > 8:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        dlines = lsd.detect(blurred)[0]
        if dlines is not None:
            for ln in np.asarray(dlines).reshape(-1, 4):
                x1, y1, x2, y2 = float(ln[0] + rx1), float(ln[1] + ry1), float(ln[2] + rx1), float(ln[3] + ry1)
                raw_segs.append((x1 / pw, y1 / ph, x2 / pw, y2 / ph, "lsd"))
    t_line = (time.perf_counter() - t_line_s) * 1000.0

    # Filter segments strictly by length, orientation, & road convergence bank
    accepted: List[Dict[str, Any]] = []
    rejected_reasons = {"too_short": 0, "too_horizontal": 0, "too_vertical": 0}
    for x1n, y1n, x2n, y2n, src_tag in raw_segs:
        xt, yt, xb, yb = _order_segment_norm(x1n, y1n, x2n, y2n)
        dx = xb - xt
        dy = yb - yt
        length_n = math.hypot(dx, dy)
        if length_n < cfg.min_line_len_ratio:
            rejected_reasons["too_short"] += 1
            continue
        ang_deg = math.degrees(math.atan2(abs(dy), max(1e-9, abs(dx))))
        if ang_deg < cfg.min_horiz_angle_deg:
            rejected_reasons["too_horizontal"] += 1
            continue
        if ang_deg > cfg.max_vert_angle_deg:
            rejected_reasons["too_vertical"] += 1
            continue

        # Bank: dx < 0 (bottom-left going up-right) => "left"; dx > 0 => "right"
        # Wait carefully: as y increases downward (yt < yb):
        # left road shoulder goes from bottom-left (xb small, yb large) up-right toward horizon (xt larger, yt smaller) -> dx = xb - xt < 0!
        # right road shoulder goes from bottom-right (xb large, yb large) up-left toward horizon (xt smaller, yt smaller) -> dx = xb - xt > 0!
        bank = "left" if dx < 0.0 else "right"
        a, b, c = _fit_line_eq(xt, yt, xb, yb)
        mx, my = 0.5 * (xt + xb), 0.5 * (yt + yb)
        ux, uy = dx / length_n, dy / length_n  # downward unit tangent
        accepted.append({
            "xt": xt, "yt": yt, "xb": xb, "yb": yb,
            "mx": mx, "my": my, "dx": dx, "dy": dy,
            "ux": ux, "uy": uy,
            "length_norm": length_n,
            "angle_deg": ang_deg,
            "bank": bank,
            "eq": (a, b, c),
            "source": src_tag,
            "px_coords": (int(round(xt * W0)), int(round(yt * H0)), int(round(xb * W0)), int(round(yb * H0))),
        })

    diag_lines = {
        "raw_line_count": len(raw_segs),
        "accepted_line_count": len(accepted),
        "rejected_line_count": len(raw_segs) - len(accepted),
        "rejected_ratio": round((len(raw_segs) - len(accepted)) / float(max(1, len(raw_segs))), 4),
        "rejected_breakdown": rejected_reasons,
        "work_wh": [pw, ph],
    }
    timings = {
        "preprocess_ms": round(t_pre, 4),
        "edge_ms": round(t_edge, 4),
        "line_detect_ms": round(t_line, 4),
    }
    return accepted, diag_lines, timings


def _ray_angle_deg(seg: Dict[str, Any], vx: float, vy: float) -> float:
    """Angle error (deg) between segment orientation and ray connecting VP -> segment midpoint."""
    rx = seg["mx"] - vx
    ry = seg["my"] - vy
    nrm = math.hypot(rx, ry)
    if nrm <= 1e-9:
        return 0.0
    # Cosine similarity magnitude (orientation undirected modulo pi, plus check forward direction ry > 0 when vy < my)
    dot = abs((rx / nrm) * seg["ux"] + (ry / nrm) * seg["uy"])
    dot = min(1.0, max(0.0, dot))
    return math.degrees(math.acos(dot))


def _weighted_ls_vp(lines: List[Dict[str, Any]]) -> Tuple[Optional[Tuple[float, float]], float]:
    """Solve min sum w_i (a_i x + b_i y + c_i)^2 with conditioning check."""
    if len(lines) < 2:
        return None, 1e9
    A = np.array([[ln["eq"][0], ln["eq"][1]] for ln in lines], dtype=np.float64)
    c = np.array([ln["eq"][2] for ln in lines], dtype=np.float64)
    w = np.array([ln["length_norm"] for ln in lines], dtype=np.float64)
    W_sqrt = np.sqrt(w)[:, None]
    Aw = A * W_sqrt
    cw = -c * np.sqrt(w)

    AtA = Aw.T @ Aw
    eigs = np.linalg.eigvalsh(AtA)
    emin, emax = float(eigs[0]), float(eigs[-1])
    if emin <= 1e-12 or emax <= 1e-12:
        return None, 1e9
    cond = float(math.sqrt(emax / emin))
    try:
        sol = np.linalg.solve(AtA, Aw.T @ cw)
        return (float(sol[0]), float(sol[1])), cond
    except np.linalg.LinAlgError:
        return None, 1e9


def estimate_vanishing_point(
    img_bgr: np.ndarray,
    cfg: Optional[HorizonEstimatorConfig] = None,
    prev_smoothed_y: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Tasks 2 & 3: Estimate image-space vanishing point & compute full geometric reliability
    diagnostics and explicit state (`VALID`, `LOW_CONFIDENCE`, `UNAVAILABLE`).
    """
    cfg = cfg or HorizonEstimatorConfig()
    H0, W0 = img_bgr.shape[:2]
    lines, line_diag, t_parts = detect_and_filter_lines(img_bgr, cfg)

    t_ransac_s = time.perf_counter()
    n_acc = len(lines)
    if n_acc < 2:
        t_r = (time.perf_counter() - t_ransac_s) * 1000.0
        return _build_vp_result(
            H0, W0, state="UNAVAILABLE", reasons=["insufficient_oriented_lines"],
            vp_norm=None, inliers=[], all_lines=lines, line_diag=line_diag,
            cond=1e9, disp=None, cross_agree=False, t_parts={**t_parts, "ransac_ms": round(t_r, 4), "qa_ms": 0.0}
        )

    # Generate valid non-parallel pairwise candidate intersections
    sin_min = math.sin(math.radians(cfg.min_pair_angle_deg))
    cands: List[Tuple[float, float, float]] = []  # (vx, vy, weight)
    max_cross_sin = 0.0
    for i in range(n_acc):
        ai, bi, ci = lines[i]["eq"]
        for j in range(i + 1, n_acc):
            aj, bj, cj = lines[j]["eq"]
            det = ai * bj - aj * bi
            max_cross_sin = max(max_cross_sin, abs(det))
            if abs(det) < sin_min:
                continue
            vx = (bi * cj - bj * ci) / det
            vy = (ci * aj - cj * ai) / det
            if (cfg.vp_x_min - 0.15) <= vx <= (cfg.vp_x_max + 0.15) and (cfg.vp_y_min - 0.15) <= vy <= (cfg.vp_y_max + 0.10):
                # Prefer cross-bank pairs cleanly
                bank_boost = 2.0 if lines[i]["bank"] != lines[j]["bank"] else 1.0
                wt = bank_boost * lines[i]["length_norm"] * lines[j]["length_norm"] * abs(det)
                cands.append((vx, vy, wt))

    if not cands:
        t_r = (time.perf_counter() - t_ransac_s) * 1000.0
        rs = ["parallel_or_degenerate_lines"] if max_cross_sin < sin_min else ["no_plausible_intersections"]
        return _build_vp_result(
            H0, W0, state="UNAVAILABLE", reasons=rs,
            vp_norm=None, inliers=[], all_lines=lines, line_diag=line_diag,
            cond=1e9, disp=None, cross_agree=False, t_parts={**t_parts, "ransac_ms": round(t_r, 4), "qa_ms": 0.0}
        )

    # Sample hypotheses (either all if small, or top/weighted deterministic subsample)
    rng = np.random.default_rng(cfg.seed)
    if len(cands) > cfg.ransac_iters:
        wts = np.array([c[2] for c in cands], dtype=np.float64)
        probs = wts / float(np.sum(wts))
        # Always include top-10 weighted + stochastic weighted sample
        top_idx = np.argsort(wts)[-min(16, cfg.ransac_iters // 4):]
        rem_n = max(1, cfg.ransac_iters - len(top_idx))
        samp_idx = rng.choice(len(cands), size=rem_n, replace=False, p=probs)
        eval_cands = [cands[k] for k in np.unique(np.concatenate([top_idx, samp_idx]))]
    else:
        eval_cands = cands

    # Vectorized MSAC hypothesis scoring
    eq_arr = np.array([ln["eq"] for ln in lines], dtype=np.float64)
    len_arr = np.array([ln["length_norm"] for ln in lines], dtype=np.float64)
    mx_arr = np.array([ln["mx"] for ln in lines], dtype=np.float64)
    my_arr = np.array([ln["my"] for ln in lines], dtype=np.float64)
    ux_arr = np.array([ln["ux"] for ln in lines], dtype=np.float64)
    uy_arr = np.array([ln["uy"] for ln in lines], dtype=np.float64)
    sin_ang_thresh = math.sin(math.radians(cfg.inlier_angle_deg))

    best_score = -1.0
    best_inlier_mask = None
    for vx, vy, _ in eval_cands:
        dists = np.abs(eq_arr[:, 0] * vx + eq_arr[:, 1] * vy + eq_arr[:, 2])
        rx = mx_arr - vx
        ry = my_arr - vy
        rn = np.maximum(1e-9, np.hypot(rx, ry))
        # sin(angle) = |r x u| / |r|
        sin_a = np.abs((rx / rn) * uy_arr - (ry / rn) * ux_arr)
        inl = (dists <= cfg.inlier_dist_norm) & (sin_a <= sin_ang_thresh) & (vy < mx_arr * 0.0 + my_arr + 0.05)
        msac_gain = np.sum(len_arr * np.where(inl, 1.0 - 0.5 * (dists / cfg.inlier_dist_norm) ** 2, 0.0))
        if msac_gain > best_score:
            best_score = msac_gain
            best_inlier_mask = inl

    inliers = [lines[i] for i in range(n_acc) if best_inlier_mask is not None and best_inlier_mask[i]]
    vp_refit, cond_num = _weighted_ls_vp(inliers)
    t_ransac = (time.perf_counter() - t_ransac_s) * 1000.0

    # Task 3: Quality Assessment
    t_qa_s = time.perf_counter()
    reasons: List[str] = []
    if vp_refit is None or len(inliers) < cfg.min_inliers_low_conf:
        reasons.append("insufficient_inlier_support")
        t_qa = (time.perf_counter() - t_qa_s) * 1000.0
        return _build_vp_result(
            H0, W0, state="UNAVAILABLE", reasons=reasons,
            vp_norm=vp_refit, inliers=inliers, all_lines=lines, line_diag=line_diag,
            cond=cond_num, disp=None, cross_agree=False, t_parts={**t_parts, "ransac_ms": round(t_ransac, 4), "qa_ms": round(t_qa, 4)}
        )

    vx_f, vy_f = vp_refit
    if not (cfg.vp_x_min <= vx_f <= cfg.vp_x_max and cfg.vp_y_min <= vy_f <= cfg.vp_y_max):
        reasons.append("implausible_vp_position")

    if cond_num > cfg.max_cond_hard:
        reasons.append("degenerate_conditioning")

    # Dispersion among pairwise intersections of INLIERS
    inlier_pts = []
    cross_pts_y = []
    left_inl = [ln for ln in inliers if ln["bank"] == "left"]
    right_inl = [ln for ln in inliers if ln["bank"] == "right"]
    for i in range(len(inliers)):
        ai, bi, ci = inliers[i]["eq"]
        for j in range(i + 1, len(inliers)):
            aj, bj, cj = inliers[j]["eq"]
            det = ai * bj - aj * bi
            if abs(det) < sin_min:
                continue
            px, py = (bi * cj - bj * ci) / det, (ci * aj - cj * ai) / det
            if abs(px - vx_f) < 0.5 and abs(py - vy_f) < 0.5:
                inlier_pts.append((px, py))
                if inliers[i]["bank"] != inliers[j]["bank"]:
                    cross_pts_y.append(py)

    disp = None
    if inlier_pts:
        pts_arr = np.asarray(inlier_pts, dtype=np.float64)
        disp = float(np.median(np.hypot(pts_arr[:, 0] - vx_f, pts_arr[:, 1] - vy_f)))

    # Check independent left vs right convergence agreement
    cross_agree = True
    if left_inl and right_inl and cross_pts_y:
        med_cross_y = float(np.median(cross_pts_y))
        if abs(med_cross_y - vy_f) > 0.08:
            cross_agree = False
            reasons.append("cross_group_disagreement")
    elif not (left_inl and right_inl):
        cross_agree = False

    # Decide final horizon state (`VALID`, `LOW_CONFIDENCE`, `UNAVAILABLE`)
    hard_fail = any(r in reasons for r in ("implausible_vp_position", "degenerate_conditioning"))
    if hard_fail:
        state = "UNAVAILABLE"
    else:
        weak_flags: List[str] = []
        if len(inliers) < cfg.min_inliers_valid:
            weak_flags.append("low_inlier_count")
        if len(left_inl) < cfg.min_left_valid or len(right_inl) < cfg.min_right_valid:
            weak_flags.append("single_sided_line_support")
        if cond_num > cfg.max_cond_valid:
            weak_flags.append("high_condition_number")
        if disp is not None and disp > cfg.max_dispersion_valid:
            weak_flags.append("high_vp_dispersion")
        if prev_smoothed_y is not None and abs(vy_f - prev_smoothed_y) > cfg.max_temporal_jump_valid:
            weak_flags.append("large_temporal_jump")
        if not cross_agree and "cross_group_disagreement" not in weak_flags:
            weak_flags.append("cross_group_unverified")

        if weak_flags:
            state = "LOW_CONFIDENCE"
            reasons.extend(weak_flags)
        else:
            state = "VALID"

    t_qa = (time.perf_counter() - t_qa_s) * 1000.0
    return _build_vp_result(
        H0, W0, state=state, reasons=reasons,
        vp_norm=(vx_f, vy_f), inliers=inliers, all_lines=lines, line_diag=line_diag,
        cond=cond_num, disp=disp, cross_agree=cross_agree,
        t_parts={**t_parts, "ransac_ms": round(t_ransac, 4), "qa_ms": round(t_qa, 4)},
    )


def _build_vp_result(
    H0: int,
    W0: int,
    state: str,
    reasons: List[str],
    vp_norm: Optional[Tuple[float, float]],
    inliers: List[Dict[str, Any]],
    all_lines: List[Dict[str, Any]],
    line_diag: Dict[str, Any],
    cond: float,
    disp: Optional[float],
    cross_agree: bool,
    t_parts: Dict[str, float],
) -> Dict[str, Any]:
    left_inl = [ln for ln in inliers if ln["bank"] == "left"]
    right_inl = [ln for ln in inliers if ln["bank"] == "right"]
    tot_ms = round(sum(t_parts.values()), 4)
    # Heuristic confidence score (explicitly NOT a calibrated probability!)
    conf_heur = 0.0
    if vp_norm is not None and state != "UNAVAILABLE":
        c_inl = min(1.0, len(inliers) / 10.0)
        c_bal = 1.0 if (left_inl and right_inl) else 0.45
        c_disp = max(0.0, 1.0 - ((disp or 0.0) / 0.08))
        c_cond = max(0.0, 1.0 - (min(cond, 60.0) / 60.0))
        conf_heur = round(float(0.35 * c_inl + 0.25 * c_bal + 0.20 * c_disp + 0.20 * c_cond), 4)

    return {
        "frame_wh": [W0, H0],
        "state": state,
        "reasons": reasons,
        "vp_norm": [round(vp_norm[0], 5), round(vp_norm[1], 5)] if vp_norm is not None else None,
        "vp_px": [round(vp_norm[0] * W0, 2), round(vp_norm[1] * H0, 2)] if vp_norm is not None else None,
        "horizon_y_ratio": round(vp_norm[1], 5) if (vp_norm is not None and state != "UNAVAILABLE") else None,
        "heuristic_confidence": conf_heur,
        "diagnostics": {
            **line_diag,
            "inlier_count": len(inliers),
            "left_inliers": len(left_inl),
            "right_inliers": len(right_inl),
            "cond_number": round(min(cond, 1e6), 3),
            "vp_dispersion_norm": round(disp, 5) if disp is not None else None,
            "cross_group_agreement": cross_agree,
        },
        "inlier_lines_px": [ln["px_coords"] for ln in inliers],
        "rejected_oriented_lines_px": [ln["px_coords"] for ln in all_lines if ln not in inliers],
        "timings_ms": {**t_parts, "total_ms": tot_ms},
    }


class DynamicHorizonTracker:
    """
    Task 4: Stateful temporal horizon-proxy filter & fallback policy manager (`A_fixed`,
    `B_dynamic_raw`, `C_dynamic_smoothed`, `D_dynamic_fallback`). Automatically resets when
    session ID, camera profile, or frame resolution changes.
    """

    def __init__(
        self,
        est_cfg: Optional[HorizonEstimatorConfig] = None,
        pol_cfg: Optional[HorizonPolicyConfig] = None,
    ):
        self.est_cfg = est_cfg or HorizonEstimatorConfig()
        self.pol_cfg = pol_cfg or HorizonPolicyConfig()
        if self.pol_cfg.policy not in FALLBACK_POLICIES:
            raise ValueError(f"Unknown policy '{self.pol_cfg.policy}'; expected {FALLBACK_POLICIES}")

        self._session_id: Optional[str] = None
        self._camera_profile: Optional[str] = None
        self._frame_wh: Optional[Tuple[int, int]] = None
        self._smoothed_y: Optional[float] = None
        self._last_valid_ts: Optional[float] = None
        self._last_valid_frame_idx: Optional[int] = None
        self._last_vp_norm: Optional[List[float]] = None
        self.reset_count: int = 0

    def reset_state(self) -> None:
        self._smoothed_y = None
        self._last_valid_ts = None
        self._last_valid_frame_idx = None
        self._last_vp_norm = None
        self.reset_count += 1

    def update(
        self,
        img_bgr: np.ndarray,
        frame_idx: int,
        ts_sec: float,
        session_id: str = "default",
        camera_profile: str = "sedan_dashcam",
        fixed_horizon_override: Optional[float] = None,
        precomputed_est: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        H0, W0 = img_bgr.shape[:2]
        fixed_h = float(fixed_horizon_override if fixed_horizon_override is not None else self.pol_cfg.fixed_horizon_ratio)

        # Strict lifecycle reset on session, camera profile, or frame resolution change
        context_reset = False
        if (
            (self._session_id is not None and self._session_id != session_id)
            or (self._camera_profile is not None and self._camera_profile != camera_profile)
            or (self._frame_wh is not None and self._frame_wh != (W0, H0))
        ):
            self.reset_state()
            context_reset = True

        self._session_id = session_id
        self._camera_profile = camera_profile
        self._frame_wh = (W0, H0)

        t0 = time.perf_counter()
        if self.pol_cfg.policy == "A_fixed":
            return {
                "policy": "A_fixed",
                "output_state": "FIXED",
                "raw_horizon_ratio": None,
                "smoothed_horizon_ratio": None,
                "active_horizon_ratio": fixed_h,
                "active_horizon_px": round(fixed_h * H0, 2),
                "measurement_age_sec": None,
                "fallback_used": True,
                "fallback_reason": "POLICY_A_FIXED_BASELINE",
                "context_reset": context_reset,
                "raw_estimator": None,
                "filter_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        est = precomputed_est if precomputed_est is not None else estimate_vanishing_point(
            img_bgr, cfg=self.est_cfg, prev_smoothed_y=self._smoothed_y
        )
        raw_state = est["state"]
        raw_y = est["horizon_y_ratio"]
        t_flt0 = time.perf_counter()

        accept_update = (raw_state == "VALID") or (
            self.pol_cfg.accept_low_conf_in_ema and raw_state == "LOW_CONFIDENCE"
        )
        if accept_update and raw_y is not None:
            if self._smoothed_y is None:
                self._smoothed_y = float(raw_y)
            else:
                a = self.pol_cfg.ema_alpha
                self._smoothed_y = a * float(raw_y) + (1.0 - a) * self._smoothed_y
            self._last_valid_ts = float(ts_sec)
            self._last_valid_frame_idx = int(frame_idx)
            self._last_vp_norm = est["vp_norm"]
            age_sec = 0.0
            eff_state = raw_state
        else:
            if self._last_valid_ts is None:
                age_sec = None
                eff_state = raw_state  # LOW_CONFIDENCE or UNAVAILABLE
            else:
                age_sec = max(0.0, float(ts_sec) - self._last_valid_ts)
                if age_sec > self.pol_cfg.max_expired_sec:
                    # Expire smoothed hold cleanly so stale prior horizon never lingers forever
                    self._smoothed_y = None
                    self._last_vp_norm = None
                    eff_state = "EXPIRED"
                elif age_sec > self.pol_cfg.max_stale_sec:
                    eff_state = "STALE"
                else:
                    eff_state = "STALE"

        # Resolve final active horizon ratio per fallback policy
        pol = self.pol_cfg.policy
        fallback_used = False
        fallback_reason = None
        if pol == "B_dynamic_raw":
            active_h = raw_y
            if active_h is None:
                fallback_reason = "RAW_UNAVAILABLE_NO_OUTPUT"
        elif pol == "C_dynamic_smoothed":
            active_h = self._smoothed_y
            if active_h is None:
                fallback_reason = "SMOOTHED_EXPIRED_OR_UNAVAILABLE"
        else:  # D_dynamic_fallback
            if eff_state in ("VALID", "STALE") and self._smoothed_y is not None:
                active_h = self._smoothed_y
                if eff_state == "STALE":
                    fallback_reason = "HOLDING_RECENT_SMOOTHED_STALE"
            else:
                active_h = fixed_h
                fallback_used = True
                fallback_reason = f"{eff_state}_FIXED_FALLBACK"

        flt_ms = (time.perf_counter() - t_flt0) * 1000.0
        return {
            "policy": pol,
            "output_state": eff_state,
            "raw_horizon_ratio": raw_y,
            "smoothed_horizon_ratio": round(self._smoothed_y, 5) if self._smoothed_y is not None else None,
            "active_horizon_ratio": round(active_h, 5) if active_h is not None else None,
            "active_horizon_px": round(active_h * H0, 2) if active_h is not None else None,
            "measurement_age_sec": round(age_sec, 4) if age_sec is not None else None,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "context_reset": context_reset,
            "raw_estimator": est,
            "filter_ms": round(flt_ms, 4),
        }

