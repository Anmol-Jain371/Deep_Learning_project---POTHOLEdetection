"""
Phase 13 Statistical Utilities.

Enforces correlation-aware reporting: frame-level observations inside a video/track are
never treated as independent samples when computing confidence intervals or p-values.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def summarize_array(vals: Sequence[Optional[float]]) -> Dict[str, Any]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "iqr": None,
            "p5": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(clean, dtype=np.float64)
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    p5, p95 = np.percentile(arr, [5.0, 95.0])
    return {
        "n": int(len(arr)),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(q50), 6),
        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 6),
        "iqr": round(float(q75 - q25), 6),
        "p5": round(float(p5), 6),
        "p95": round(float(p95), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def paired_difference_summary(
    a_vals: Sequence[Optional[float]],
    b_vals: Sequence[Optional[float]],
) -> Dict[str, Any]:
    """Compute paired differences (B - A) & standardized paired Cohen's d where finite pairs exist."""
    pairs: List[Tuple[float, float]] = []
    for a, b in zip(a_vals, b_vals):
        if a is None or b is None:
            continue
        fa, fb = float(a), float(b)
        if math.isfinite(fa) and math.isfinite(fb):
            pairs.append((fa, fb))
    if not pairs:
        return {"n_pairs": 0, "mean_diff": None, "median_diff": None, "std_diff": None, "cohens_d_paired": None}
    diffs = np.asarray([b - a for a, b in pairs], dtype=np.float64)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    d_eff = (mean_d / std_d) if std_d > 1e-12 else None
    return {
        "n_pairs": int(len(diffs)),
        "mean_diff": round(mean_d, 6),
        "median_diff": round(float(np.median(diffs)), 6),
        "std_diff": round(std_d, 6),
        "cohens_d_paired": round(float(d_eff), 4) if d_eff is not None else None,
    }


def group_level_bootstrap_ci(
    group_means: Sequence[float],
    *,
    seed: int = 42,
    n_boot: int = 1000,
    min_groups_for_ci: int = 5,
) -> Dict[str, Any]:
    """
    Compute bootstrap 95% CI ONLY at independent capture/session group granularity.
    Refuses narrow fake CIs when independent group count < min_groups_for_ci.
    """
    clean = [float(x) for x in group_means if x is not None and math.isfinite(float(x))]
    k = len(clean)
    if k < min_groups_for_ci:
        return {
            "group_count": k,
            "ci95_low": None,
            "ci95_high": None,
            "statistical_power_note": (
                f"Only {k} independent capture/session groups (< {min_groups_for_ci}); "
                "confidence interval omitted to avoid frame-level pseudoreplication."
            ),
        }
    rng = np.random.default_rng(seed)
    arr = np.asarray(clean, dtype=np.float64)
    means = [float(np.mean(rng.choice(arr, size=k, replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "group_count": k,
        "ci95_low": round(float(lo), 6),
        "ci95_high": round(float(hi), 6),
        "statistical_power_note": f"Group-level bootstrap over {k} independent units ({n_boot} resamples).",
    }
