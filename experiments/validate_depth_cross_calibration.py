"""
========================================================================================
RoadVision AI -- Phase 3A: Physical Depth Cross-Calibration & Integrity Evaluation
========================================================================================
Evaluates the existing monocular metric calibration equation:
    Delta_Z_cm = Delta_d_raw * (Z_road ^ 1.35) * k_disp
while fixing exponent = 1.35 and comparing:
  1. Baseline fixed coefficient (k_disp = 0.082)
  2. Previously suggested in-sample rescaled coefficient (k_disp = 0.07664, IN-SAMPLE / LEAKY)
  3. Leave-One-Out Cross-Validated proportional scaling (k_cv[-j], exponent fixed at 1.35)
  4. Leave-One-Out Affine Linear Regression baseline (y = alpha + beta * X)
  5. Leave-One-Out Constant Mean-Depth dummy baseline

CRITICAL SCIENTIFIC INTEGRITY NOTICE:
  - Inspects physical sample provenance in depth_ground_truth_manifest.csv.
  - Because unique physical pothole group IDs, field ruler logs, raw disparity deltas,
    and bounding-box rows are absent—and P01/P02 match synthetic test cases T01/T02
    while pipeline outputs at those timestamps diverge—this script explicitly sets
    `independent_physical_identity_verified = False` and frames leave-one-row-out CV
    as a statistical manifest audit rather than certified leakage-free field validation.

Preserves all production configuration files and existing reports.
========================================================================================
"""

import os
import sys
import csv
import json
import math
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_CSV = os.path.join(BASE_DIR, "depth_ground_truth_manifest.csv")
CAMERA_JSON = os.path.join(BASE_DIR, "config", "camera_params.json")
BASELINE_REPORT_JSON = os.path.join(BASE_DIR, "runs", "depth_validation_report.json")
BENCHMARK_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
os.makedirs(BENCHMARK_DIR, exist_ok=True)

OUT_JSON = os.path.join(BENCHMARK_DIR, "depth_cross_calibration_results.json")
OUT_MD = os.path.join(BENCHMARK_DIR, "depth_cross_calibration_report.md")
OUT_CSV = os.path.join(BENCHMARK_DIR, "depth_cross_calibration_predictions.csv")
PLOT_SCATTER = os.path.join(BENCHMARK_DIR, "depth_pred_vs_measured_scatter.png")
PLOT_RESIDUAL = os.path.join(BENCHMARK_DIR, "depth_residual_error_plot.png")
PLOT_FOLDS = os.path.join(BENCHMARK_DIR, "depth_kdisp_fold_variation.png")

BASELINE_K_DISP = 0.082
PREV_SUGGESTED_K_DISP = 0.07664
FIXED_EXPONENT = 1.35


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_severity_by_depth(depth_cm: float) -> str:
    if depth_cm >= 8.0:
        return "SEVERE"
    elif depth_cm >= 4.0:
        return "MODERATE"
    return "MILD"


def classify_severity_with_water(depth_cm: float, water_present: bool) -> str:
    if water_present:
        return "SEVERE"
    return classify_severity_by_depth(depth_cm)


def load_manifest_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = row["Pothole_ID"].strip()
            ts_raw = row["Timestamp_Sec_or_Frame"].strip()
            ts_sec = float(ts_raw.rstrip("s"))
            dist_m = float(row["Approx_Camera_Dist_m"].strip())
            phys_cm = float(row["Physical_Depth_cm"].strip())
            est_082_cm = float(row["Estimated_Depth_cm"].strip())
            water = row["Water_Present"].strip().upper() in ("YES", "TRUE", "1", "Y")
            notes = row["Cavity_Description_Notes"].strip()
            # Latent scale-invariant uncalibrated factor X = raw_disp * Z^1.35 = est_082 / 0.082
            x_latent = est_082_cm / BASELINE_K_DISP
            rows.append({
                "pothole_id": pid,
                "source": row["Source_Video_or_Image"].strip(),
                "timestamp_raw": ts_raw,
                "timestamp_sec": ts_sec,
                "distance_m": dist_m,
                "physical_cm": phys_cm,
                "est_082_cm": est_082_cm,
                "x_latent": x_latent,
                "water": water,
                "notes": notes,
            })
    return rows


def bootstrap_ci(values: np.ndarray, fn, n_boot: int = 10000, seed: int = 42) -> List[float]:
    if len(values) < 2:
        val = float(fn(values)) if len(values) == 1 else 0.0
        return [round(val, 4), round(val, 4)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = np.array([fn(values[i]) for i in idx], dtype=np.float64)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def compute_full_metrics(
    rows: List[Dict[str, Any]],
    preds: np.ndarray,
    method_name: str,
    provenance_tag: str
) -> Dict[str, Any]:
    y_true = np.array([r["physical_cm"] for r in rows], dtype=np.float64)
    dists = np.array([r["distance_m"] for r in rows], dtype=np.float64)
    waters = np.array([r["water"] for r in rows], dtype=bool)
    y_pred = np.asarray(preds, dtype=np.float64)

    signed_err = y_pred - y_true
    abs_err = np.abs(signed_err)
    sq_err = signed_err ** 2

    nonzero_mask = y_true > 0.0
    ape_pct = (abs_err[nonzero_mask] / y_true[nonzero_mask]) * 100.0

    # Severity concordance (both pure depth-threshold and with water override matching validate_physical_depth.py)
    true_cats = [classify_severity_by_depth(t) for t in y_true]
    pred_cats_water = [classify_severity_with_water(p, bool(w)) for p, w in zip(y_pred, waters)]
    pred_cats_depth_only = [classify_severity_by_depth(p) for p in y_pred]

    conc_water = sum(t == p for t, p in zip(true_cats, pred_cats_water)) / len(rows) * 100.0
    conc_depth_only = sum(t == p for t, p in zip(true_cats, pred_cats_depth_only)) / len(rows) * 100.0

    cm_water = {c: {c2: 0 for c2 in ("MILD", "MODERATE", "SEVERE")} for c in ("MILD", "MODERATE", "SEVERE")}
    for t, p in zip(true_cats, pred_cats_water):
        cm_water[t][p] += 1

    # Distance bands
    band_defs = [("2-5m", 2.0, 5.0), ("5-8m", 5.0, 8.0), ("8-11m", 8.0, 11.0), ("11-15m", 11.0, 15.0001)]
    bands_out = {}
    for bname, lo, hi in band_defs:
        mask = (dists >= lo) & (dists < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            b_conc = sum(true_cats[i] == pred_cats_water[i] for i in range(len(rows)) if mask[i]) / cnt * 100.0
            bands_out[bname] = {
                "samples": cnt,
                "mae_cm": round(float(np.mean(abs_err[mask])), 4),
                "rmse_cm": round(float(np.sqrt(np.mean(sq_err[mask]))), 4),
                "mbe_cm": round(float(np.mean(signed_err[mask])), 4),
                "concordance_pct": round(float(b_conc), 1),
            }
        else:
            bands_out[bname] = {"samples": 0, "mae_cm": None, "rmse_cm": None, "mbe_cm": None, "concordance_pct": None}

    # Dry vs Wet
    dry_mask = ~waters
    wet_mask = waters

    def subset_stats(mask):
        cnt = int(np.sum(mask))
        if cnt == 0:
            return {"count": 0}
        return {
            "count": cnt,
            "mae_cm": round(float(np.mean(abs_err[mask])), 4),
            "rmse_cm": round(float(np.sqrt(np.mean(sq_err[mask]))), 4),
            "mbe_cm": round(float(np.mean(signed_err[mask])), 4),
            "medae_cm": round(float(np.median(abs_err[mask])), 4),
            "maxae_cm": round(float(np.max(abs_err[mask])), 4),
        }

    # Pair indices for paired bootstrap CIs
    idx_all = np.arange(len(rows))
    mae_ci = bootstrap_ci(abs_err, np.mean)
    mbe_ci = bootstrap_ci(signed_err, np.mean)
    rmse_ci = bootstrap_ci(sq_err, lambda x: math.sqrt(float(np.mean(x))))

    return {
        "method": method_name,
        "provenance": provenance_tag,
        "sample_count": len(rows),
        "mae_cm": round(float(np.mean(abs_err)), 4),
        "mae_ci95_bootstrap": mae_ci,
        "rmse_cm": round(float(np.sqrt(np.mean(sq_err))), 4),
        "rmse_ci95_bootstrap": rmse_ci,
        "mape_pct": round(float(np.mean(ape_pct)), 2),
        "mape_pct_1dec": round(float(np.mean(ape_pct)), 1),
        "mbe_cm": round(float(np.mean(signed_err)), 4),
        "mbe_ci95_bootstrap": mbe_ci,
        "medae_cm": round(float(np.median(abs_err)), 4),
        "maxae_cm": round(float(np.max(abs_err)), 4),
        "severity_concordance_with_water_pct": round(float(conc_water), 1),
        "severity_concordance_depth_only_pct": round(float(conc_depth_only), 1),
        "confusion_matrix_with_water": cm_water,
        "distance_bands": bands_out,
        "dry_vs_water": {
            "dry": subset_stats(dry_mask),
            "water": subset_stats(wet_mask),
        },
        "predictions": [round(float(v), 4) for v in y_pred],
        "signed_errors": [round(float(v), 4) for v in signed_err],
        "abs_errors": [round(float(v), 4) for v in abs_err],
    }


def run_loo_experiments(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    y_true = np.array([r["physical_cm"] for r in rows], dtype=np.float64)
    x_lat = np.array([r["x_latent"] for r in rows], dtype=np.float64)
    y_pred_082 = np.array([r["est_082_cm"] for r in rows], dtype=np.float64)

    # 1. Fixed baseline k = 0.082 (reproduces existing report)
    preds_082 = y_pred_082.copy()

    # 2. Previously suggested k = 0.07664 (IN-SAMPLE / LEAKY: computed from all 15 rows)
    preds_07664 = x_lat * PREV_SUGGESTED_K_DISP

    # 3. Leave-One-Out CV proportional scaling (fixed exponent = 1.35)
    preds_loo_k = np.zeros(n, dtype=np.float64)
    fold_k_values = np.zeros(n, dtype=np.float64)

    # 4. Leave-One-Out CV Affine Linear baseline (y = alpha + beta * X)
    preds_loo_affine = np.zeros(n, dtype=np.float64)
    fold_affine_params = []

    # 5. Leave-One-Out Constant Mean-Depth dummy baseline
    preds_loo_mean = np.zeros(n, dtype=np.float64)

    fold_records = []
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False

        x_tr, y_tr = x_lat[mask], y_true[mask]

        # Proportional zero-intercept OLS: min sum (y - k * x)^2 => k = sum(x*y) / sum(x^2)
        k_fold = float(np.sum(x_tr * y_tr) / np.sum(x_tr ** 2))
        fold_k_values[j] = k_fold
        preds_loo_k[j] = k_fold * x_lat[j]

        # Affine OLS: y = alpha + beta * x
        A = np.column_stack([x_tr, np.ones_like(x_tr)])
        beta, alpha = np.linalg.lstsq(A, y_tr, rcond=None)[0]
        preds_loo_affine[j] = float(alpha + beta * x_lat[j])
        fold_affine_params.append({"alpha": round(float(alpha), 4), "beta": round(float(beta), 6)})

        # Constant mean baseline
        preds_loo_mean[j] = float(np.mean(y_tr))

        fold_records.append({
            "fold_idx": j + 1,
            "excluded_id": rows[j]["pothole_id"],
            "timestamp_sec": rows[j]["timestamp_sec"],
            "distance_m": rows[j]["distance_m"],
            "water": rows[j]["water"],
            "measured_cm": rows[j]["physical_cm"],
            "pred_fixed_082_cm": round(float(preds_082[j]), 4),
            "pred_insample_07664_cm": round(float(preds_07664[j]), 4),
            "fitted_k_cv": round(k_fold, 6),
            "pred_loo_cv_cm": round(float(preds_loo_k[j]), 4),
            "err_loo_cv_cm": round(float(preds_loo_k[j] - y_true[j]), 4),
            "abs_err_loo_cv_cm": round(float(abs(preds_loo_k[j] - y_true[j])), 4),
            "pred_loo_affine_cm": round(float(preds_loo_affine[j]), 4),
            "err_loo_affine_cm": round(float(preds_loo_affine[j] - y_true[j]), 4),
            "pred_loo_mean_cm": round(float(preds_loo_mean[j]), 4),
            "err_loo_mean_cm": round(float(preds_loo_mean[j] - y_true[j]), 4),
        })

    return {
        "preds_082": preds_082,
        "preds_07664": preds_07664,
        "preds_loo_k": preds_loo_k,
        "preds_loo_affine": preds_loo_affine,
        "preds_loo_mean": preds_loo_mean,
        "fold_k_values": fold_k_values,
        "fold_affine_params": fold_affine_params,
        "fold_records": fold_records,
    }


def make_plots(rows: List[Dict[str, Any]], loo_data: Dict[str, Any]):
    y_true = np.array([r["physical_cm"] for r in rows])
    dists = np.array([r["distance_m"] for r in rows])
    pids = [r["pothole_id"] for r in rows]
    waters = np.array([r["water"] for r in rows])

    p082 = loo_data["preds_082"]
    p076 = loo_data["preds_07664"]
    p_cv = loo_data["preds_loo_k"]
    p_aff = loo_data["preds_loo_affine"]
    k_folds = loo_data["fold_k_values"]

    # Plot 1: Predicted vs Measured Scatter Plot
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=160)
    lim_max = max(float(np.max(y_true)), float(np.max(p082))) + 1.2
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1.2, label="Ideal 1:1 Identity")
    ax.fill_between([0, lim_max], [0 - 1.0, lim_max - 1.0], [1.0, lim_max + 1.0], color="#e5e7eb", alpha=0.45, label="±1.0 cm Band")
    ax.scatter(y_true, p082, marker="o", s=55, color="#dc2626", alpha=0.85, label="Fixed Baseline (k=0.082)")
    ax.scatter(y_true, p076, marker="^", s=55, color="#f59e0b", alpha=0.80, label="In-Sample Rescaled (k=0.07664, leaky)")
    ax.scatter(y_true, p_cv, marker="s", s=55, color="#2563eb", alpha=0.85, label="LOO-CV Fitted k (exponent=1.35)")
    ax.scatter(y_true, p_aff, marker="D", s=42, color="#10b981", alpha=0.75, label="LOO Affine Baseline")
    # Severity boundary guides
    for thr in (4.0, 8.0):
        ax.axvline(thr, color="#94a3b8", linestyle=":", linewidth=0.9)
        ax.axhline(thr, color="#94a3b8", linestyle=":", linewidth=0.9)
    ax.set_xlim(1.5, lim_max)
    ax.set_ylim(1.5, lim_max)
    ax.set_xlabel("Manifest Recorded Depth (cm)", fontsize=11)
    ax.set_ylabel("Calibrated Predicted Depth (cm)", fontsize=11)
    ax.set_title("Phase 3A: Predicted vs. Manifest Depth (n=15; Unverified Provenance Caveat)", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER)
    plt.close(fig)

    # Plot 2: Residual Error Plot (vs Distance and vs Measured Depth)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=160)
    res_082 = p082 - y_true
    res_076 = p076 - y_true
    res_cv = p_cv - y_true
    res_aff = p_aff - y_true

    for ax, x_vals, xlabel in [
        (axes[0], dists, "Approximate Camera Distance (m)"),
        (axes[1], y_true, "Manifest Recorded Depth (cm)"),
    ]:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        ax.axhspan(-1.0, 1.0, color="#e5e7eb", alpha=0.45)
        ax.scatter(x_vals, res_082, marker="o", s=50, color="#dc2626", label="k=0.082 (Fixed)")
        ax.scatter(x_vals, res_076, marker="^", s=50, color="#f59e0b", label="k=0.07664 (In-Sample)")
        ax.scatter(x_vals, res_cv, marker="s", s=50, color="#2563eb", label="LOO-CV k")
        ax.scatter(x_vals, res_aff, marker="D", s=40, color="#10b981", label="LOO Affine")
        # Mark wet potholes
        if np.any(waters):
            ax.scatter(x_vals[waters], res_cv[waters], facecolors="none", edgecolors="#0284c7", s=140, linewidths=1.8, label="Water-Flagged (n=2)")
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel("Residual Error (Pred - True, cm)", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Phase 3A Residual Error Diagnostics Across Calibration Strategies", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_RESIDUAL)
    plt.close(fig)

    # Plot 3: LOO Calibration Coefficient k_disp Variation Across Folds
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=160)
    x_idx = np.arange(len(pids))
    ax.plot(x_idx, k_folds, marker="o", linewidth=1.8, color="#2563eb", label="Fold-fitted k_cv[-j] (14 calibration rows)")
    ax.axhline(BASELINE_K_DISP, color="#dc2626", linestyle="--", linewidth=1.4, label=f"Fixed Production k_disp = {BASELINE_K_DISP}")
    ax.axhline(PREV_SUGGESTED_K_DISP, color="#f59e0b", linestyle="-.", linewidth=1.4, label=f"In-Sample Suggested k_disp = {PREV_SUGGESTED_K_DISP} (leaky)")
    ax.axhline(float(np.mean(k_folds)), color="#1d4ed8", linestyle=":", linewidth=1.2, label=f"LOO Mean k_cv = {np.mean(k_folds):.5f}")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(pids, rotation=45)
    ax.set_xlabel("Excluded Sample Fold", fontsize=10.5)
    ax.set_ylabel("Fitted Calibration Coefficient k_disp", fontsize=10.5)
    ax.set_title("Leave-One-Out Calibration Coefficient (k_disp) Stability Across Folds (exp = 1.35)", fontsize=11.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8.5, loc="best")
    plt.tight_layout()
    plt.savefig(PLOT_FOLDS)
    plt.close(fig)


def write_markdown_report(summary: Dict[str, Any], fold_records: List[Dict[str, Any]]):
    methods = summary["methods"]
    k_stats = summary["loo_coefficient_variation"]

    lines = [
        "# Phase 3A: Physical Depth Cross-Calibration & Integrity Report",
        "",
        f"- **Manifest Audited:** `depth_ground_truth_manifest.csv` (`SHA256: {summary['manifest_sha256']}`)",
        f"- **Baseline Report Reference:** `runs/depth_validation_report.json` (`SHA256: {summary['baseline_report_sha256']}`)",
        f"- **Fixed Calibration Equation Evaluated:** $\\Delta Z_{{cm}} = \\Delta d_{{raw}} \\cdot Z_{{road}}^{{1.35}} \\cdot k_{{disp}}$ (`depth_exponent = 1.35` held fixed)",
        "",
        "> [!CAUTION]",
        "> **Critical Scientific Caveat (Sample Independence & Provenance):** As documented in [`reports/depth_ground_truth_audit.md`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/depth_ground_truth_audit.md), `depth_ground_truth_manifest.csv` lacks unique physical pothole group IDs, field ruler photos, raw disparity deltas, and bounding-box coordinates (`y2`). Furthermore, `P01`/`P02` match synthetic test rows `T01`/`T02` from `validate_physical_depth.py`, and re-running `RoadVisionPipeline` at `0.8s–16.0s` produces divergent predictions (including `0` detections at `13.2s`). Therefore, leave-one-row-out CV (`independent_physical_identity_verified = False`) measures statistical behavior on the existing manifest **and must not be claimed as certified leakage-free field validation** until independent field measurements are collected.",
        "",
        "## 1. Baseline Reproduction & Method Comparison (`n = 15`)",
        "",
        "| Method | Provenance | MAE (cm) [95% Boot CI] | RMSE (cm) [95% Boot CI] | MBE (cm) | MedAE (cm) | MaxAE (cm) | MAPE (%) | Concordance (Depth / Water-Flag) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key in [
        "fixed_0.082",
        "insample_0.07664_leaky",
        "loo_cv_proportional",
        "loo_cv_affine_baseline",
        "loo_cv_mean_baseline",
    ]:
        m = methods[key]
        lines.append(
            f"| **{m['method']}** | {m['provenance']} | "
            f"{m['mae_cm']:.2f} [{m['mae_ci95_bootstrap'][0]:.2f}, {m['mae_ci95_bootstrap'][1]:.2f}] | "
            f"{m['rmse_cm']:.2f} [{m['rmse_ci95_bootstrap'][0]:.2f}, {m['rmse_ci95_bootstrap'][1]:.2f}] | "
            f"{m['mbe_cm']:+.2f} | {m['medae_cm']:.2f} | {m['maxae_cm']:.2f} | {m['mape_pct']:.1f}% | "
            f"{m['severity_concordance_depth_only_pct']:.1f}% / {m['severity_concordance_with_water_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "## 2. Calibration Coefficient Variation Across Folds (`k_cv[-j]`, Exponent = 1.35)",
        "",
        f"- **Production Fixed $k_{{disp}}$:** `{BASELINE_K_DISP}`",
        f"- **In-Sample Rescaled $k_{{disp}}$ (Leaky, all 15 rows):** `{PREV_SUGGESTED_K_DISP}`",
        f"- **LOO-CV $k_{{cv}}$ Mean ± SD:** `{k_stats['mean']:.6f} ± {k_stats['std']:.6f}` (CV: `{k_stats['coeff_of_variation_pct']:.2f}%`)",
        f"- **LOO-CV $k_{{cv}}$ Range [Min, Max]:** `[{k_stats['min']:.6f}, {k_stats['max']:.6f}]`",
        "",
        "## 3. Distance-Band & Surface Condition Breakdown (`MAE / RMSE / n`)",
        "",
        "| Method | 2–5m (n=4) | 5–8m (n=4) | 8–11m (n=4) | 11–15m (n=3) | Dry (n=13) | Water (n=2)* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for key in ["fixed_0.082", "insample_0.07664_leaky", "loo_cv_proportional", "loo_cv_affine_baseline"]:
        m = methods[key]
        b = m["distance_bands"]
        dw = m["dry_vs_water"]
        lines.append(
            f"| {m['method']} | "
            f"{b['2-5m']['mae_cm']:.2f}/{b['2-5m']['rmse_cm']:.2f} | "
            f"{b['5-8m']['mae_cm']:.2f}/{b['5-8m']['rmse_cm']:.2f} | "
            f"{b['8-11m']['mae_cm']:.2f}/{b['8-11m']['rmse_cm']:.2f} | "
            f"{b['11-15m']['mae_cm']:.2f}/{b['11-15m']['rmse_cm']:.2f} | "
            f"{dw['dry']['mae_cm']:.2f}/{dw['dry']['rmse_cm']:.2f} | "
            f"{dw['water']['mae_cm']:.2f}/{dw['water']['rmse_cm']:.2f} |"
        )

    lines.extend([
        "",
        "*Note: `Water (n=2)` has only 2 samples (`P08`, `P15`), which is insufficient for statistical generalization or confidence intervals.*",
        "",
        "## 4. Per-Fold Predictions (`k_cv[-j]` vs. Fixed `0.082`)",
        "",
        "| ID | Dist (m) | Water | Meas (cm) | Pred `0.082` | Pred `0.07664` | Fold $k_{cv}$ | Pred `LOO-CV` | Err `LOO-CV` | Pred `LOO-Affine` |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in fold_records:
        lines.append(
            f"| {r['excluded_id']} | {r['distance_m']:.1f} | {'YES' if r['water'] else 'NO'} | {r['measured_cm']:.1f} | "
            f"{r['pred_fixed_082_cm']:.2f} | {r['pred_insample_07664_cm']:.2f} | {r['fitted_k_cv']:.5f} | "
            f"{r['pred_loo_cv_cm']:.2f} | {r['err_loo_cv_cm']:+.2f} | {r['pred_loo_affine_cm']:.2f} |"
        )

    lines.extend([
        "",
        "## 5. Generated Visual Diagnostics",
        "",
        f"- Scatter Plot: [`depth_pred_vs_measured_scatter.png`](file:///{PLOT_SCATTER.replace(os.sep, '/')})",
        f"- Residual Error Plot: [`depth_residual_error_plot.png`](file:///{PLOT_RESIDUAL.replace(os.sep, '/')})",
        f"- LOO Coefficient Variation Plot: [`depth_kdisp_fold_variation.png`](file:///{PLOT_FOLDS.replace(os.sep, '/')})",
    ])

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = load_manifest_rows(MANIFEST_CSV)
    loo_data = run_loo_experiments(rows)

    methods = {
        "fixed_0.082": compute_full_metrics(
            rows, loo_data["preds_082"],
            "Fixed k_disp = 0.082 (Baseline)",
            "Pre-existing engine constant predating manifest population"
        ),
        "insample_0.07664_leaky": compute_full_metrics(
            rows, loo_data["preds_07664"],
            "In-Sample Rescaled k_disp = 0.07664",
            "IN-SAMPLE / LEAKY: derived across all 15 rows in validate_physical_depth.py"
        ),
        "loo_cv_proportional": compute_full_metrics(
            rows, loo_data["preds_loo_k"],
            "LOO-CV Proportional k_disp (exp=1.35)",
            "Leave-one-row-out zero-intercept OLS (independent physical identity unverified)"
        ),
        "loo_cv_affine_baseline": compute_full_metrics(
            rows, loo_data["preds_loo_affine"],
            "LOO-CV Affine Linear Baseline (a + b*X)",
            "Leave-one-row-out affine OLS baseline"
        ),
        "loo_cv_mean_baseline": compute_full_metrics(
            rows, loo_data["preds_loo_mean"],
            "LOO-CV Mean-Depth Dummy Baseline",
            "Distance/disparity-agnostic sample mean baseline"
        ),
    }

    k_vals = loo_data["fold_k_values"]
    k_summary = {
        "mean": round(float(np.mean(k_vals)), 6),
        "std": round(float(np.std(k_vals, ddof=1)), 6),
        "min": round(float(np.min(k_vals)), 6),
        "max": round(float(np.max(k_vals)), 6),
        "median": round(float(np.median(k_vals)), 6),
        "coeff_of_variation_pct": round(float(np.std(k_vals, ddof=1) / np.mean(k_vals) * 100.0), 3),
    }

    # Verify exact reproduction of prior report
    prev_report = json.load(open(BASELINE_REPORT_JSON, "r", encoding="utf-8"))
    repro_check = {
        "prior_mae_cm": prev_report["mae_cm"],
        "reproduced_mae_cm": round(methods["fixed_0.082"]["mae_cm"], 2),
        "prior_rmse_cm": prev_report["rmse_cm"],
        "reproduced_rmse_cm": round(methods["fixed_0.082"]["rmse_cm"], 2),
        "prior_mape_pct": prev_report["mape_pct"],
        "reproduced_mape_pct": methods["fixed_0.082"]["mape_pct_1dec"],
        "prior_concordance_pct": prev_report["severity_concordance_pct"],
        "reproduced_concordance_pct": methods["fixed_0.082"]["severity_concordance_with_water_pct"],
    }

    summary = {
        "experiment": "Phase 3A Independent Physical Depth Calibration Validation",
        "manifest_path": MANIFEST_CSV,
        "manifest_sha256": sha256_file(MANIFEST_CSV),
        "baseline_report_sha256": sha256_file(BASELINE_REPORT_JSON),
        "camera_params_sha256": sha256_file(CAMERA_JSON),
        "fixed_exponent": FIXED_EXPONENT,
        "sample_independence_audit": {
            "total_manifest_rows": len(rows),
            "independent_physical_identity_verified": False,
            "verified_independent_pothole_groups": None,
            "reason": (
                "No physical pothole group IDs, field ruler logs, raw disparity deltas, "
                "or bounding-box pixel coordinates are recorded; all 15 timestamps fall "
                "monotonically within 0.8s-16.0s of a single continuous 40.2s video; P01/P02 "
                "match synthetic test rows T01/T02; and live pipeline predictions at those "
                "timestamps do not match Estimated_Depth_cm."
            ),
        },
        "baseline_reproduction_check": repro_check,
        "loo_coefficient_variation": k_summary,
        "methods": methods,
        "fold_records": loo_data["fold_records"],
        "plots": {
            "scatter": PLOT_SCATTER,
            "residuals": PLOT_RESIDUAL,
            "fold_variation": PLOT_FOLDS,
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(loo_data["fold_records"][0].keys()))
        writer.writeheader()
        writer.writerows(loo_data["fold_records"])

    make_plots(rows, loo_data)
    write_markdown_report(summary, loo_data["fold_records"])

    print("=== PHASE 3A CROSS-CALIBRATION SUMMARY ===")
    print("Baseline reproduction:", json.dumps(repro_check))
    print("LOO k_cv variation   :", json.dumps(k_summary))
    for k, m in methods.items():
        print(
            f"  {k:<25} | MAE={m['mae_cm']:.4f} CI{m['mae_ci95_bootstrap']} | "
            f"RMSE={m['rmse_cm']:.4f} | MBE={m['mbe_cm']:+.4f} | "
            f"MedAE={m['medae_cm']:.4f} | MaxAE={m['maxae_cm']:.4f} | "
            f"MAPE={m['mape_pct']:.2f}% | Conc={m['severity_concordance_with_water_pct']:.1f}%"
        )
    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
