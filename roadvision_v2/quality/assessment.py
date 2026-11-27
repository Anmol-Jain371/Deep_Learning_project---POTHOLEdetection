"""
Phase 9: Uncertainty-Aware Hazard Measurement Quality Assessment for RoadVision AI 2.0.

Produces an explainable, non-probabilistic heuristic measurement-quality index [0, 1]
and separated component diagnostics using existing pipeline outputs without modifying
existing RANSAC, cavity extraction, score polarity, or acceptance thresholds.

Scientific & Engineering Guarantees:
- Quality categories represent measurement evidence quality ONLY, never physical hazard
  severity or calibrated probability of correctness.
- Invalid and expired measurements remain INVALID regardless of high detector confidence.
- Missing diagnostics are explicitly reported as unavailable (`None`) rather than assumed 1.0.
- Correlated diagnostics (e.g., RANSAC inlier ratio, residual MAD, sample count) are
  combined inside a single Geometric Reference component rather than treated as independent.
- Temporal stability statistics are computed strictly from compatible FRESH_VALID samples;
  repeated CACHED_VALID observations are never counted as independent fresh evidence.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roadvision_v2.config import QualityConfig

QUALITY_CATEGORIES = (
    "HIGH_QUALITY",
    "MODERATE_QUALITY",
    "LOW_QUALITY",
    "INSUFFICIENT_EVIDENCE",
    "INVALID",
)

QUALITY_METHODOLOGY_ID = "heuristic_weighted_composite_v1"

# Mapping from actual pipeline rejection/warning codes to human-readable explanations
EXPLAINABLE_REJECTION_MAP: Dict[str, Dict[str, str]] = {
    "insufficient_reference_samples": {
        "title": "Insufficient reference samples",
        "explanation": (
            "Too few valid annular pavement pixels surrounded the detection to constrain "
            "a planar road reference surface."
        ),
    },
    "poor_spatial_support": {
        "title": "Poor spatial reference distribution",
        "explanation": (
            "Valid reference samples were concentrated in too few quadrants around the "
            "detection box, risking plane tilt extrapolation."
        ),
    },
    "degenerate_reference_plane": {
        "title": "Degenerate reference plane fit",
        "explanation": (
            "The fitted road plane matrix was ill-conditioned or collinear and could not "
            "provide a reliable baseline."
        ),
    },
    "low_inlier_ratio": {
        "title": "Poor reference-plane fit (low inlier ratio)",
        "explanation": (
            "Fewer annular pavement pixels than required agreed with the RANSAC road "
            "plane, indicating noisy surround or nearby obstacles."
        ),
    },
    "overlapping_neighbors_insufficient_pavement": {
        "title": "Insufficient unmasked pavement due to neighbors",
        "explanation": (
            "Neighboring pothole boxes masked too much of the surrounding annular ring "
            "to retain a reliable pavement baseline."
        ),
    },
    "unexpected_residual_polarity": {
        "title": "Unexpected residual polarity",
        "explanation": (
            "Interior inverse-depth residuals did not exhibit a consistent positive "
            "depression relative to the fitted road plane."
        ),
    },
    "unstable_cavity_residuals": {
        "title": "Unstable cavity residual distribution",
        "explanation": (
            "Interior residual spread (IQR) was high relative to the cavity signal and "
            "fewer than half of interior pixels were positive."
        ),
    },
    "unstable_multiscale_interior": {
        "title": "Unstable multi-scale interior agreement",
        "explanation": (
            "Cavity residuals varied inconsistently or flipped sign across concentric "
            "interior crop ratios."
        ),
    },
    "insufficient_or_nonfinite_interior": {
        "title": "Insufficient or non-finite cavity interior",
        "explanation": (
            "Too few finite depth pixels were present inside the selected cavity core "
            "or reference annulus."
        ),
    },
    "frame_boundary_truncation": {
        "title": "Frame boundary truncation",
        "explanation": (
            "Detection lies close to image edges with insufficient unclipped surrounding "
            "pavement quadrants."
        ),
    },
    "suspected_water_reflection": {
        "title": "Potential water or reflection contamination",
        "explanation": (
            "Specular highlights and low local texture suggest standing water or "
            "reflections that degrade monocular depth cues."
        ),
    },
    "EXPIRED": {
        "title": "Track measurement expired",
        "explanation": (
            "Cached measurement exceeded the maximum freshness age limit without a "
            "successful depth refresh."
        ),
    },
    "track_measurement_expired": {
        "title": "Track measurement expired",
        "explanation": (
            "Cached measurement exceeded the maximum freshness age limit without a "
            "successful depth refresh."
        ),
    },
    "NOT_EVALUATED": {
        "title": "Depth inference not scheduled or disabled",
        "explanation": (
            "Depth estimation was disabled for this mode or not yet scheduled for this "
            "track."
        ),
    },
    "depth_disabled": {
        "title": "Depth estimation disabled",
        "explanation": (
            "Depth model execution is disabled in the selected operating mode."
        ),
    },
    "SKIPPED_BY_BUDGET": {
        "title": "Skipped by latency budget scheduler",
        "explanation": (
            "The budget-aware depth scheduler deferred depth inference on this frame to "
            "preserve frame-rate budget."
        ),
    },
}


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def _safe_float(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def explain_diagnostic_codes(
    failure_reasons: Sequence[str],
    warnings: Sequence[str] = (),
    measurement_state: Optional[str] = None,
    fallback_reason: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return structured, human-readable explanations while preserving original codes."""
    codes: List[str] = []
    for c in list(failure_reasons or []) + list(warnings or []):
        if c and c not in codes:
            codes.append(str(c))
    if measurement_state == "EXPIRED" and "EXPIRED" not in codes:
        codes.append("EXPIRED")
    elif measurement_state == "NOT_EVALUATED":
        if fallback_reason and fallback_reason not in codes:
            codes.append(str(fallback_reason))
        elif not codes:
            codes.append("NOT_EVALUATED")

    explained: List[Dict[str, str]] = []
    for code in codes:
        if code in EXPLAINABLE_REJECTION_MAP:
            info = EXPLAINABLE_REJECTION_MAP[code]
            explained.append({"code": code, "title": info["title"], "explanation": info["explanation"]})
        elif code.startswith("boundary_truncated:"):
            edges = code.split(":", 1)[1]
            explained.append(
                {
                    "code": code,
                    "title": "Bounding box near frame boundary",
                    "explanation": f"Detection box clipped near frame edge(s): {edges}.",
                }
            )
        elif code == "neighbors_excluded_from_ring":
            explained.append(
                {
                    "code": code,
                    "title": "Neighbor detections excluded from reference ring",
                    "explanation": "Adjacent detections were masked out of the annular road baseline.",
                }
            )
        elif code.startswith("extraction_error:"):
            explained.append(
                {
                    "code": code,
                    "title": "Depth extraction runtime error",
                    "explanation": f"Cavity extraction raised {code.split(':', 1)[1]}; score marked invalid.",
                }
            )
        else:
            explained.append(
                {
                    "code": code,
                    "title": code.replace("_", " ").capitalize(),
                    "explanation": f"Pipeline diagnostic code '{code}' was recorded.",
                }
            )
    return explained


def _compute_detection_quality(
    detector_confidence: Optional[float],
    box_xyxy: Optional[Sequence[float]],
    frame_wh: Optional[Sequence[int]],
    truncated_edges: Sequence[str],
    cfg: QualityConfig,
) -> Tuple[Optional[float], List[str]]:
    conf = _safe_float(detector_confidence)
    if conf is None:
        return None, ["missing_detector_confidence"]

    conf_norm = _clamp01(conf / max(cfg.min_confidence_for_full_score, 1e-3))
    clip_factor = 1.0
    notes: List[str] = []

    edge_list = list(truncated_edges or [])
    if not edge_list and box_xyxy is not None and frame_wh is not None and len(box_xyxy) == 4 and len(frame_wh) == 2:
        w_img, h_img = float(frame_wh[0] or 0), float(frame_wh[1] or 0)
        if w_img > 0 and h_img > 0:
            x1, y1, x2, y2 = [float(v) for v in box_xyxy]
            if x1 <= 2.0:
                edge_list.append("left")
            if y1 <= 2.0:
                edge_list.append("top")
            if x2 >= w_img - 2.0:
                edge_list.append("right")
            if y2 >= h_img - 2.0:
                edge_list.append("bottom")

    if len(edge_list) >= 2:
        clip_factor = 0.70
        notes.append("multi_edge_boundary_clipping")
    elif len(edge_list) == 1:
        clip_factor = 0.85
        notes.append("single_edge_boundary_clipping")

    return round(_clamp01(conf_norm * clip_factor), 4), notes


def _compute_geometric_reference_quality(
    ref_diag: Optional[Dict[str, Any]],
    cfg: QualityConfig,
) -> Tuple[Optional[float], List[str]]:
    if not isinstance(ref_diag, dict) or not ref_diag:
        return None, ["missing_ref_diagnostics"]

    if ref_diag.get("plane_degenerate") is True:
        return 0.0, ["degenerate_reference_plane"]

    # Combine correlated geometric indicators as one grouped weighted sub-score
    sub_scores: List[Tuple[float, float]] = []
    missing: List[str] = []

    inlier_ratio = _safe_float(ref_diag.get("inlier_ratio"))
    if inlier_ratio is not None:
        denom = max(cfg.ideal_inlier_ratio - cfg.min_inlier_ratio, 1e-3)
        if inlier_ratio < cfg.min_inlier_ratio:
            inlier_score = _clamp01(0.45 * (inlier_ratio / max(cfg.min_inlier_ratio, 1e-3)))
        else:
            inlier_score = _clamp01(0.45 + 0.55 * ((inlier_ratio - cfg.min_inlier_ratio) / denom))
        sub_scores.append((0.34, inlier_score))
    else:
        missing.append("missing_inlier_ratio")

    used_samples = _safe_float(ref_diag.get("used_ref_samples"))
    if used_samples is not None:
        if used_samples < cfg.min_ref_samples:
            sample_score = _clamp01(0.40 * (used_samples / max(cfg.min_ref_samples, 1)))
        else:
            denom = max(cfg.ideal_ref_samples - cfg.min_ref_samples, 1)
            sample_score = _clamp01(0.40 + 0.60 * ((used_samples - cfg.min_ref_samples) / denom))
        sub_scores.append((0.20, sample_score))
    else:
        missing.append("missing_used_ref_samples")

    pop_quads = _safe_float(ref_diag.get("populated_quadrants"))
    if pop_quads is not None:
        sub_scores.append((0.18, _clamp01(pop_quads / 4.0)))
    else:
        missing.append("missing_populated_quadrants")

    ref_mad = _safe_float(ref_diag.get("ref_fit_mad"))
    if ref_mad is not None:
        mad_score = _clamp01(1.0 - (ref_mad / max(cfg.mad_reference_scale, 1e-4)))
        sub_scores.append((0.20, mad_score))
    else:
        missing.append("missing_ref_fit_mad")

    ret_ratio = _safe_float(ref_diag.get("annular_retention_ratio"))
    if ret_ratio is not None:
        sub_scores.append((0.08, _clamp01(ret_ratio)))

    # Require at least core plane diagnostics (inlier_ratio or used_ref_samples)
    if inlier_ratio is None and used_samples is None:
        return None, missing

    total_w = sum(w for w, _ in sub_scores)
    score = sum(w * s for w, s in sub_scores) / max(total_w, 1e-6)
    return round(_clamp01(score), 4), missing


def _compute_cavity_extraction_quality(
    cavity_diag: Optional[Dict[str, Any]],
    measurement_validity: bool,
    cfg: QualityConfig,
) -> Tuple[Optional[float], List[str]]:
    if not isinstance(cavity_diag, dict) or not cavity_diag:
        return None, ["missing_cavity_diagnostics"]

    sub_scores: List[Tuple[float, float]] = []
    missing: List[str] = []

    px = _safe_float(cavity_diag.get("cavity_sample_count") or cavity_diag.get("px"))
    if px is not None:
        if px < cfg.min_cavity_px:
            px_score = _clamp01(0.40 * (px / max(cfg.min_cavity_px, 1)))
        else:
            denom = max(cfg.ideal_cavity_px - cfg.min_cavity_px, 1)
            px_score = _clamp01(0.40 + 0.60 * ((px - cfg.min_cavity_px) / denom))
        sub_scores.append((0.35, px_score))
    else:
        missing.append("missing_cavity_sample_count")

    pos_frac = _safe_float(cavity_diag.get("positive_frac"))
    if pos_frac is not None:
        sub_scores.append((0.40, _clamp01(pos_frac)))
    else:
        missing.append("missing_positive_frac")

    iqr_val = _safe_float(cavity_diag.get("iqr_residual"))
    p75_val = _safe_float(cavity_diag.get("p75_residual") or cavity_diag.get("raw_relative_cavity_score"))
    if iqr_val is not None and p75_val is not None and abs(p75_val) > 1e-6:
        # Moderate spread relative to p75 yields higher score; extreme scatter lowers it
        ratio = iqr_val / max(abs(p75_val), 1e-5)
        stability = _clamp01(1.0 - max(0.0, ratio - 0.5) / 2.5)
        sub_scores.append((0.25, stability))

    if not sub_scores:
        return None, missing

    total_w = sum(w for w, _ in sub_scores)
    score = sum(w * s for w, s in sub_scores) / max(total_w, 1e-6)
    if not measurement_validity:
        score = min(score, 0.35)
    return round(_clamp01(score), 4), missing


def _compute_freshness_component(
    measurement_state: str,
    measurement_age_sec: Optional[float],
    cfg: QualityConfig,
) -> Optional[float]:
    if measurement_state == "FRESH_VALID":
        return 1.0
    if measurement_state == "CACHED_VALID":
        age = _safe_float(measurement_age_sec)
        if age is None:
            # Penalize conservatively when cache age is unavailable
            return round(1.0 - cfg.cached_freshness_penalty_max, 4)
        frac = min(1.0, max(0.0, age) / max(cfg.max_cache_age_sec, 1e-3))
        return round(_clamp01(1.0 - cfg.cached_freshness_penalty_max * frac), 4)
    if measurement_state in ("FRESH_INVALID", "EXPIRED"):
        return 0.0
    return None


def _compute_environmental_risk_component(
    is_water_suspected: Optional[bool],
    truncated_edges: Sequence[str],
    warnings: Sequence[str],
) -> Optional[float]:
    if is_water_suspected is None:
        return None
    if is_water_suspected:
        return 0.15
    warn_set = set(warnings or [])
    if truncated_edges or any(w.startswith("boundary_truncated:") for w in warn_set):
        return 0.70
    if "neighbors_excluded_from_ring" in warn_set:
        return 0.85
    return 1.0


def compute_track_temporal_stability(
    observations: Sequence[Dict[str, Any]],
    cfg: Optional[QualityConfig] = None,
) -> Dict[str, Any]:
    """
    Compute descriptive uncertainty and temporal stability statistics strictly from
    compatible FRESH_VALID observations on the same track.

    Never treats repeated CACHED_VALID values as independent fresh samples.
    Never mixes incompatible depth model/resolution outputs silently.
    """
    cfg = cfg or QualityConfig()
    state_counts = Counter()
    fresh_by_config: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    cached_ages: List[float] = []
    latest_age: Optional[float] = None

    for obs in observations:
        meas = obs.get("measurement") if isinstance(obs.get("measurement"), dict) else obs
        st = str(meas.get("measurement_state") or "NOT_EVALUATED")
        state_counts[st] += 1
        age_val = _safe_float(meas.get("measurement_age_sec"))
        if age_val is not None:
            latest_age = age_val
            if st == "CACHED_VALID":
                cached_ages.append(age_val)

        if st == "FRESH_VALID" and bool(meas.get("measurement_validity", False)):
            raw_score = _safe_float(meas.get("raw_relative_cavity_score"))
            if raw_score is None:
                continue
            cfg_key = (
                meas.get("model_identity"),
                meas.get("depth_input_resolution"),
                meas.get("inference_precision"),
            )
            rd = meas.get("ref_diagnostics") if isinstance(meas.get("ref_diagnostics"), dict) else {}
            fresh_by_config.setdefault(cfg_key, []).append(
                {
                    "raw_score": raw_score,
                    "ref_fit_mad": _safe_float(rd.get("ref_fit_mad")),
                }
            )

    total_obs = sum(state_counts.values())
    fresh_valid_count = int(state_counts.get("FRESH_VALID", 0))
    fresh_invalid_count = int(state_counts.get("FRESH_INVALID", 0))
    cached_valid_count = int(state_counts.get("CACHED_VALID", 0))
    expired_count = int(state_counts.get("EXPIRED", 0))
    not_evaluated_count = int(state_counts.get("NOT_EVALUATED", 0))
    fresh_coverage = round(fresh_valid_count / total_obs, 4) if total_obs > 0 else None

    incompatible_config_count = len(fresh_by_config)
    dominant_key: Optional[Tuple[Any, ...]] = None
    compat_items: List[Dict[str, Any]] = []
    if fresh_by_config:
        dominant_key = max(fresh_by_config.keys(), key=lambda k: len(fresh_by_config[k]))
        compat_items = fresh_by_config[dominant_key]

    n_compat = len(compat_items)
    if n_compat < cfg.min_fresh_samples_for_stability:
        reason = (
            f"Insufficient compatible FRESH_VALID samples ({n_compat}/{cfg.min_fresh_samples_for_stability}) "
            "to compute temporal stability statistics."
        )
        if incompatible_config_count > 1:
            reason += " Multiple depth model/resolution configurations were kept separate."
        return {
            "evidence_sufficiency": "INSUFFICIENT",
            "insufficient_evidence_reason": reason,
            "fresh_valid_count": fresh_valid_count,
            "fresh_invalid_count": fresh_invalid_count,
            "cached_valid_count": cached_valid_count,
            "expired_count": expired_count,
            "not_evaluated_count": not_evaluated_count,
            "compatible_fresh_sample_count": n_compat,
            "incompatible_depth_configs_detected": incompatible_config_count > 1,
            "fresh_measurement_coverage": fresh_coverage,
            "latest_measurement_age_sec": latest_age,
            "median_relative_score": None,
            "mad_relative_score": None,
            "iqr_relative_score": None,
            "robust_temporal_variability": None,
            "ref_fit_mad_variability": None,
            "temporal_consistency_component": None,
            "note": (
                "Descriptive empirical summary over FRESH_VALID observations only; "
                "NOT a statistical confidence interval or physical depth uncertainty."
            ),
        }

    arr = np.asarray([it["raw_score"] for it in compat_items], dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    iqr = float(q75 - q25)
    # Robust scaled dispersion around median (analogous to robust CV; dimensionless)
    robust_var = float((1.4826 * mad) / max(abs(med), 1e-4))
    temporal_component = round(_clamp01(1.0 - (robust_var / max(cfg.rel_cv_reference_scale, 1e-3))), 4)

    ref_mads = [it["ref_fit_mad"] for it in compat_items if it["ref_fit_mad"] is not None]
    ref_mad_var: Optional[float] = None
    if len(ref_mads) >= cfg.min_fresh_samples_for_stability:
        rm_arr = np.asarray(ref_mads, dtype=np.float64)
        rm_q75, rm_q25 = np.percentile(rm_arr, [75.0, 25.0])
        ref_mad_var = round(float(rm_q75 - rm_q25), 6)

    return {
        "evidence_sufficiency": "SUFFICIENT",
        "insufficient_evidence_reason": None,
        "fresh_valid_count": fresh_valid_count,
        "fresh_invalid_count": fresh_invalid_count,
        "cached_valid_count": cached_valid_count,
        "expired_count": expired_count,
        "not_evaluated_count": not_evaluated_count,
        "compatible_fresh_sample_count": n_compat,
        "incompatible_depth_configs_detected": incompatible_config_count > 1,
        "fresh_measurement_coverage": fresh_coverage,
        "latest_measurement_age_sec": latest_age,
        "median_relative_score": round(med, 6),
        "mad_relative_score": round(mad, 6),
        "iqr_relative_score": round(iqr, 6),
        "robust_temporal_variability": round(robust_var, 4),
        "ref_fit_mad_variability": ref_mad_var,
        "temporal_consistency_component": temporal_component,
        "note": (
            "Descriptive empirical summary over compatible FRESH_VALID observations only; "
            "NOT a statistical confidence interval or physical depth uncertainty."
        ),
    }


def assess_observation_quality(
    *,
    measurement_state: str,
    measurement_validity: bool,
    detector_confidence: Optional[float],
    ref_diagnostics: Optional[Dict[str, Any]] = None,
    cavity_diagnostics: Optional[Dict[str, Any]] = None,
    temporal_stability: Optional[Dict[str, Any]] = None,
    measurement_age_sec: Optional[float] = None,
    is_water_suspected: Optional[bool] = None,
    box_xyxy: Optional[Sequence[float]] = None,
    frame_wh: Optional[Sequence[int]] = None,
    truncated_edges: Sequence[str] = (),
    failure_reasons: Sequence[str] = (),
    warnings: Sequence[str] = (),
    fallback_reason: Optional[str] = None,
    source_fresh_quality: Optional[Dict[str, Any]] = None,
    cfg: Optional[QualityConfig] = None,
) -> Dict[str, Any]:
    """
    Compute explainable heuristic measurement quality assessment for a single observation.
    """
    cfg = cfg or QualityConfig()
    state = str(measurement_state or "NOT_EVALUATED")

    det_q, det_notes = _compute_detection_quality(
        detector_confidence=detector_confidence,
        box_xyxy=box_xyxy,
        frame_wh=frame_wh,
        truncated_edges=truncated_edges,
        cfg=cfg,
    )

    # For CACHED_VALID, carry original fresh geometric/cavity/environmental evidence if missing on this frame
    effective_ref_diag = ref_diagnostics
    effective_cav_diag = cavity_diagnostics
    effective_water = is_water_suspected
    source_fresh_index: Optional[float] = None
    source_fresh_timestamp: Optional[float] = None

    if state == "CACHED_VALID" and isinstance(source_fresh_quality, dict):
        src_comp = source_fresh_quality.get("components") or {}
        if not effective_ref_diag and source_fresh_quality.get("raw_ref_diagnostics"):
            effective_ref_diag = source_fresh_quality.get("raw_ref_diagnostics")
        if not effective_cav_diag and source_fresh_quality.get("raw_cavity_diagnostics"):
            effective_cav_diag = source_fresh_quality.get("raw_cavity_diagnostics")
        if effective_water is None and "is_water_suspected" in source_fresh_quality:
            effective_water = source_fresh_quality.get("is_water_suspected")
        source_fresh_index = _safe_float(source_fresh_quality.get("quality_index"))
        source_fresh_timestamp = _safe_float(source_fresh_quality.get("source_timestamp_sec"))
        inherited_geom = _safe_float(src_comp.get("geometric_reference_quality"))
        inherited_cav = _safe_float(src_comp.get("cavity_extraction_quality"))
        inherited_env = _safe_float(src_comp.get("environmental_risk"))
    else:
        inherited_geom = None
        inherited_cav = None
        inherited_env = None

    geom_q, geom_missing = _compute_geometric_reference_quality(effective_ref_diag, cfg)
    if geom_q is None and inherited_geom is not None:
        geom_q = inherited_geom
        geom_missing = []

    cav_q, cav_missing = _compute_cavity_extraction_quality(
        effective_cav_diag,
        measurement_validity=bool(measurement_validity),
        cfg=cfg,
    )
    if cav_q is None and inherited_cav is not None:
        cav_q = inherited_cav
        cav_missing = []

    temp_q = None
    if isinstance(temporal_stability, dict):
        temp_q = _safe_float(temporal_stability.get("temporal_consistency_component"))

    fresh_q = _compute_freshness_component(state, measurement_age_sec, cfg)

    env_q = _compute_environmental_risk_component(effective_water, truncated_edges, warnings)
    if env_q is None and inherited_env is not None:
        env_q = inherited_env

    components = {
        "detection_quality": det_q,
        "geometric_reference_quality": geom_q,
        "cavity_extraction_quality": cav_q,
        "temporal_consistency": temp_q,
        "measurement_freshness": fresh_q,
        "environmental_risk": env_q,
    }

    weights = {
        "detection_quality": cfg.w_detection,
        "geometric_reference_quality": cfg.w_geometric_ref,
        "cavity_extraction_quality": cfg.w_cavity_extraction,
        "temporal_consistency": cfg.w_temporal_consistency,
        "measurement_freshness": cfg.w_freshness,
        "environmental_risk": cfg.w_environmental,
    }

    missing_evidence: List[str] = []
    missing_evidence.extend(det_notes if det_q is None else [])
    if geom_q is None:
        missing_evidence.extend(geom_missing)
    if cav_q is None:
        missing_evidence.extend(cav_missing)
    if temp_q is None:
        missing_evidence.append("insufficient_fresh_temporal_history")
    if env_q is None:
        missing_evidence.append("missing_environmental_risk")

    explanations = explain_diagnostic_codes(
        failure_reasons=failure_reasons,
        warnings=warnings,
        measurement_state=state,
        fallback_reason=fallback_reason,
    )

    available_weight = sum(weights[k] for k, v in components.items() if v is not None)
    total_weight = sum(weights.values())
    evidence_coverage = round(available_weight / max(total_weight, 1e-6), 4)

    quality_index: Optional[float] = None
    category: str

    if state in ("FRESH_INVALID", "EXPIRED") or (not measurement_validity and state != "CACHED_VALID"):
        if state == "NOT_EVALUATED":
            category = "INSUFFICIENT_EVIDENCE"
        else:
            category = "INVALID"
    elif state == "NOT_EVALUATED":
        category = "INSUFFICIENT_EVIDENCE"
    elif evidence_coverage < cfg.min_component_coverage_for_index or geom_q is None:
        category = "INSUFFICIENT_EVIDENCE"
    else:
        raw_weighted = sum(weights[k] * float(v) for k, v in components.items() if v is not None)
        idx = _clamp01(raw_weighted / max(available_weight, 1e-6))

        # Cached measurements can NEVER exceed the source fresh observation's quality index
        if state == "CACHED_VALID" and source_fresh_index is not None:
            fresh_penalty_ratio = fresh_q if fresh_q is not None else (1.0 - cfg.cached_freshness_penalty_max)
            idx = min(idx, source_fresh_index * max(0.5, fresh_penalty_ratio))

        quality_index = round(_clamp01(idx), 4)
        if quality_index >= cfg.high_quality_min:
            category = "HIGH_QUALITY"
        elif quality_index >= cfg.moderate_quality_min:
            category = "MODERATE_QUALITY"
        else:
            category = "LOW_QUALITY"

    return {
        "quality_index": quality_index,
        "quality_category": category,
        "methodology": QUALITY_METHODOLOGY_ID,
        "is_experimental_heuristic": True,
        "is_calibrated_probability": False,
        "components": components,
        "component_weights": weights,
        "evidence_coverage_ratio": evidence_coverage,
        "missing_evidence": missing_evidence,
        "is_cached_inherited": state == "CACHED_VALID",
        "source_fresh_quality_index": source_fresh_index if state == "CACHED_VALID" else None,
        "source_fresh_timestamp_sec": source_fresh_timestamp if state == "CACHED_VALID" else None,
        "measurement_age_sec": _safe_float(measurement_age_sec),
        "diagnostic_explanations": explanations,
        "disclaimer": (
            "Heuristic measurement-quality index (0-1) reflecting geometric & temporal "
            "evidence only; NOT a calibrated probability or physical severity rating."
        ),
    }


def build_track_quality_summary(
    track_observations: Sequence[Dict[str, Any]],
    cfg: Optional[QualityConfig] = None,
) -> Dict[str, Any]:
    """Aggregate Phase 9 measurement quality & uncertainty indicators at track level."""
    cfg = cfg or QualityConfig()
    stability = compute_track_temporal_stability(track_observations, cfg=cfg)

    quality_indices: List[float] = []
    fresh_quality_indices: List[float] = []
    comp_accum: Dict[str, List[float]] = {
        "detection_quality": [],
        "geometric_reference_quality": [],
        "cavity_extraction_quality": [],
        "temporal_consistency": [],
        "measurement_freshness": [],
        "environmental_risk": [],
    }
    reason_counts: Counter = Counter()
    category_counts: Counter = Counter()

    for obs in track_observations:
        meas = obs.get("measurement") if isinstance(obs.get("measurement"), dict) else obs
        qa = meas.get("quality_assessment") if isinstance(meas.get("quality_assessment"), dict) else None
        st = str(meas.get("measurement_state") or "NOT_EVALUATED")

        for r in meas.get("failure_reasons") or []:
            reason_counts[str(r)] += 1
        if st == "EXPIRED":
            reason_counts["EXPIRED"] += 1

        if qa:
            cat = qa.get("quality_category")
            if cat:
                category_counts[str(cat)] += 1
            qi = _safe_float(qa.get("quality_index"))
            if qi is not None:
                quality_indices.append(qi)
                if st == "FRESH_VALID":
                    fresh_quality_indices.append(qi)
            comps = qa.get("components") or {}
            for k in comp_accum:
                cv = _safe_float(comps.get(k))
                if cv is not None:
                    comp_accum[k].append(cv)

    median_qi = round(float(np.median(quality_indices)), 4) if quality_indices else None
    median_fresh_qi = round(float(np.median(fresh_quality_indices)), 4) if fresh_quality_indices else None
    comp_summary = {
        k: (round(float(np.median(vals)), 4) if vals else None)
        for k, vals in comp_accum.items()
    }
    dominant_reasons = [
        {
            "code": code,
            "count": int(cnt),
            **(EXPLAINABLE_REJECTION_MAP.get(code, {"title": code, "explanation": f"Diagnostic '{code}'"})),
        }
        for code, cnt in reason_counts.most_common(5)
    ]

    if median_fresh_qi is not None:
        if median_fresh_qi >= cfg.high_quality_min:
            track_cat = "HIGH_QUALITY"
        elif median_fresh_qi >= cfg.moderate_quality_min:
            track_cat = "MODERATE_QUALITY"
        else:
            track_cat = "LOW_QUALITY"
    elif stability["fresh_valid_count"] == 0 and (
        stability["fresh_invalid_count"] > 0 or stability["expired_count"] > 0
    ):
        track_cat = "INVALID"
    else:
        track_cat = "INSUFFICIENT_EVIDENCE"

    return {
        "track_quality_category": track_cat,
        "median_quality_index": median_qi,
        "median_fresh_quality_index": median_fresh_qi,
        "quality_category_counts": dict(category_counts),
        "component_medians": comp_summary,
        "fresh_valid_count": stability["fresh_valid_count"],
        "fresh_invalid_count": stability["fresh_invalid_count"],
        "cached_measurement_count": stability["cached_valid_count"],
        "expired_count": stability["expired_count"],
        "not_evaluated_count": stability["not_evaluated_count"],
        "fresh_measurement_coverage": stability["fresh_measurement_coverage"],
        "evidence_sufficiency": stability["evidence_sufficiency"],
        "insufficient_evidence_reason": stability["insufficient_evidence_reason"],
        "median_relative_score": stability["median_relative_score"],
        "mad_relative_score": stability["mad_relative_score"],
        "iqr_relative_score": stability["iqr_relative_score"],
        "relative_depth_variability": stability["robust_temporal_variability"],
        "ref_fit_mad_variability": stability["ref_fit_mad_variability"],
        "dominant_rejection_reasons": dominant_reasons,
        "is_experimental_heuristic": True,
        "note": (
            "Track summary aggregates candidate observations for one session ByteTrack ID "
            "(not an independently verified physical pothole)."
        ),
    }


def summarize_session_quality(
    observations: Sequence[Dict[str, Any]],
    track_summaries: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Compute session-wide Phase 9 measurement quality breakdown."""
    cat_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    qi_vals: List[float] = []
    fresh_qi_vals: List[float] = []

    for obs in observations:
        meas = obs.get("measurement") if isinstance(obs.get("measurement"), dict) else obs
        st = str(meas.get("measurement_state") or "NOT_EVALUATED")
        qa = meas.get("quality_assessment") if isinstance(meas.get("quality_assessment"), dict) else None
        for r in meas.get("failure_reasons") or []:
            reason_counts[str(r)] += 1
        if st == "EXPIRED":
            reason_counts["EXPIRED"] += 1
        if qa:
            cat = qa.get("quality_category")
            if cat:
                cat_counts[str(cat)] += 1
            qi = _safe_float(qa.get("quality_index"))
            if qi is not None:
                qi_vals.append(qi)
                if st == "FRESH_VALID":
                    fresh_qi_vals.append(qi)

    return {
        "quality_category_counts": dict(cat_counts),
        "median_quality_index": round(float(np.median(qi_vals)), 4) if qi_vals else None,
        "median_fresh_quality_index": round(float(np.median(fresh_qi_vals)), 4) if fresh_qi_vals else None,
        "scored_observation_count": len(qi_vals),
        "total_observation_count": len(observations),
        "sufficient_evidence_track_count": sum(
            1 for ts in track_summaries if ts.get("evidence_sufficiency") == "SUFFICIENT"
        ),
        "total_track_count": len(track_summaries),
        "top_rejection_reasons": [
            {
                "code": code,
                "count": int(cnt),
                **(EXPLAINABLE_REJECTION_MAP.get(code, {"title": code, "explanation": f"Diagnostic '{code}'"})),
            }
            for code, cnt in reason_counts.most_common(8)
        ],
        "methodology": QUALITY_METHODOLOGY_ID,
        "is_experimental_heuristic": True,
        "is_calibrated_probability": False,
    }
