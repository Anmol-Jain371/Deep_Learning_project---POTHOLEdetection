"""JSON sanitization, response formatting, and scientific disclaimers for RoadVision AI 2.0 API."""

from __future__ import annotations

import math
from typing import Any, Dict

from roadvision_v2.schemas import MEASUREMENT_STATES

PIPELINE_VERSION = "2.0.0-phase8b"

SCIENTIFIC_MEASUREMENT_DISCLAIMER = (
    "RoadVision AI 2.0 depth outputs are uncalibrated relative inverse-depth observations. "
    "They are NOT metric centimetres, millimetres, physical area, or physical volume, and "
    "must NOT be mapped to physical severity classes. Track counts represent unique session "
    "track IDs, not independently confirmed unique physical potholes."
)


def sanitize_json_value(obj: Any) -> Any:
    """
    Recursively convert numpy types and replace NaN / Infinity with None (JSON null).
    Guarantees strict RFC 8259 JSON compliance.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        return sanitize_json_value(obj.item())
    if isinstance(obj, dict):
        return {str(k): sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json_value(v) for v in obj]
    return str(obj)


def empty_measurement_state_counts() -> Dict[str, int]:
    """Return zero-initialized counter dict covering every valid measurement state."""
    return {state: 0 for state in MEASUREMENT_STATES}


def compute_measurement_coverage(
    state_counts: Dict[str, int],
    total_observations: int,
) -> Dict[str, Any]:
    """
    Compute fresh valid coverage and cached availability separately with explicit denominators.
    Never combines cached reads into fresh valid coverage.
    """
    denom = max(0, int(total_observations))
    fresh_valid = int(state_counts.get("FRESH_VALID", 0))
    fresh_invalid = int(state_counts.get("FRESH_INVALID", 0))
    cached_valid = int(state_counts.get("CACHED_VALID", 0))
    expired = int(state_counts.get("EXPIRED", 0))
    not_evaluated = int(state_counts.get("NOT_EVALUATED", 0))
    evaluated_keyframe_attempts = fresh_valid + fresh_invalid

    return {
        "denominator_total_observations": denom,
        "denominator_keyframe_attempts": evaluated_keyframe_attempts,
        "counts_by_state": {
            "FRESH_VALID": fresh_valid,
            "FRESH_INVALID": fresh_invalid,
            "CACHED_VALID": cached_valid,
            "EXPIRED": expired,
            "NOT_EVALUATED": not_evaluated,
        },
        "fresh_valid_rate_over_observations": (
            round(fresh_valid / denom, 4) if denom > 0 else None
        ),
        "fresh_valid_rate_over_keyframes": (
            round(fresh_valid / evaluated_keyframe_attempts, 4)
            if evaluated_keyframe_attempts > 0
            else None
        ),
        "cached_valid_rate_over_observations": (
            round(cached_valid / denom, 4) if denom > 0 else None
        ),
    }
