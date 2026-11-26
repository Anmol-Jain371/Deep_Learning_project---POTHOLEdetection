# Phase 3A: Physical Depth Cross-Calibration & Integrity Report

- **Manifest Audited:** `depth_ground_truth_manifest.csv` (`SHA256: 7f4761a3fbcbb84cc1427f0e95aa6a90d288ace5cb8bbd013d7826e1dc493b48`)
- **Baseline Report Reference:** `runs/depth_validation_report.json` (`SHA256: 348b453468e19690644e44626aa7301a83c14f54a7bbaec72062b18605554c98`)
- **Fixed Calibration Equation Evaluated:** $\Delta Z_{cm} = \Delta d_{raw} \cdot Z_{road}^{1.35} \cdot k_{disp}$ (`depth_exponent = 1.35` held fixed)

> [!CAUTION]
> **Critical Scientific Caveat (Sample Independence & Provenance):** As documented in [`reports/depth_ground_truth_audit.md`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/depth_ground_truth_audit.md), `depth_ground_truth_manifest.csv` lacks unique physical pothole group IDs, field ruler photos, raw disparity deltas, and bounding-box coordinates (`y2`). Furthermore, `P01`/`P02` match synthetic test rows `T01`/`T02` from `validate_physical_depth.py`, and re-running `RoadVisionPipeline` at `0.8s–16.0s` produces divergent predictions (including `0` detections at `13.2s`). Therefore, leave-one-row-out CV (`independent_physical_identity_verified = False`) measures statistical behavior on the existing manifest **and must not be claimed as certified leakage-free field validation** until independent field measurements are collected.

## 1. Baseline Reproduction & Method Comparison (`n = 15`)

| Method | Provenance | MAE (cm) [95% Boot CI] | RMSE (cm) [95% Boot CI] | MBE (cm) | MedAE (cm) | MaxAE (cm) | MAPE (%) | Concordance (Depth / Water-Flag) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Fixed k_disp = 0.082 (Baseline)** | Pre-existing engine constant predating manifest population | 0.56 [0.47, 0.65] | 0.59 [0.49, 0.69] | +0.43 | 0.50 | 0.90 | 9.6% | 100.0% / 100.0% |
| **In-Sample Rescaled k_disp = 0.07664** | IN-SAMPLE / LEAKY: derived across all 15 rows in validate_physical_depth.py | 0.23 [0.12, 0.37] | 0.34 [0.14, 0.51] | -0.02 | 0.16 | 0.94 | 4.4% | 100.0% / 100.0% |
| **LOO-CV Proportional k_disp (exp=1.35)** | Leave-one-row-out zero-intercept OLS (independent physical identity unverified) | 0.24 [0.13, 0.39] | 0.36 [0.15, 0.53] | -0.01 | 0.17 | 0.97 | 4.6% | 100.0% / 100.0% |
| **LOO-CV Affine Linear Baseline (a + b*X)** | Leave-one-row-out affine OLS baseline | 0.27 [0.16, 0.42] | 0.38 [0.19, 0.54] | +0.01 | 0.22 | 0.99 | 5.6% | 100.0% / 100.0% |
| **LOO-CV Mean-Depth Dummy Baseline** | Distance/disparity-agnostic sample mean baseline | 2.34 [1.57, 3.08] | 2.78 [2.03, 3.40] | +0.00 | 2.29 | 4.97 | 47.2% | 40.0% / 53.3% |

## 2. Calibration Coefficient Variation Across Folds (`k_cv[-j]`, Exponent = 1.35)

- **Production Fixed $k_{disp}$:** `0.082`
- **In-Sample Rescaled $k_{disp}$ (Leaky, all 15 rows):** `0.07664`
- **LOO-CV $k_{cv}$ Mean ± SD:** `0.076640 ± 0.000214` (CV: `0.28%`)
- **LOO-CV $k_{cv}$ Range [Min, Max]:** `[0.076123, 0.076950]`

## 3. Distance-Band & Surface Condition Breakdown (`MAE / RMSE / n`)

| Method | 2–5m (n=4) | 5–8m (n=4) | 8–11m (n=4) | 11–15m (n=3) | Dry (n=13) | Water (n=2)* |
|---|---:|---:|---:|---:|---:|---:|
| Fixed k_disp = 0.082 (Baseline) | 0.40/0.40 | 0.57/0.59 | 0.57/0.59 | 0.73/0.77 | 0.52/0.54 | 0.85/0.85 |
| In-Sample Rescaled k_disp = 0.07664 | 0.32/0.40 | 0.07/0.09 | 0.32/0.48 | 0.21/0.22 | 0.24/0.36 | 0.17/0.22 |
| LOO-CV Proportional k_disp (exp=1.35) | 0.33/0.41 | 0.07/0.09 | 0.33/0.50 | 0.23/0.24 | 0.25/0.37 | 0.19/0.24 |
| LOO-CV Affine Linear Baseline (a + b*X) | 0.40/0.45 | 0.09/0.12 | 0.36/0.51 | 0.24/0.26 | 0.29/0.39 | 0.18/0.23 |

*Note: `Water (n=2)` has only 2 samples (`P08`, `P15`), which is insufficient for statistical generalization or confidence intervals.*

## 4. Per-Fold Predictions (`k_cv[-j]` vs. Fixed `0.082`)

| ID | Dist (m) | Water | Meas (cm) | Pred `0.082` | Pred `0.07664` | Fold $k_{cv}$ | Pred `LOO-CV` | Err `LOO-CV` | Pred `LOO-Affine` |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| P01 | 3.2 | NO | 3.0 | 3.40 | 3.18 | 0.07670 | 3.18 | +0.18 | 3.29 |
| P02 | 4.1 | NO | 5.5 | 5.10 | 4.77 | 0.07624 | 4.74 | -0.76 | 4.74 |
| P03 | 4.8 | NO | 2.8 | 3.20 | 2.99 | 0.07670 | 2.99 | +0.19 | 3.12 |
| P04 | 3.6 | NO | 8.5 | 8.90 | 8.32 | 0.07645 | 8.30 | -0.20 | 8.28 |
| P05 | 5.4 | NO | 6.2 | 6.80 | 6.36 | 0.07675 | 6.36 | +0.16 | 6.38 |
| P06 | 6.1 | NO | 4.5 | 4.90 | 4.58 | 0.07668 | 4.58 | +0.08 | 4.64 |
| P07 | 6.9 | NO | 7.2 | 7.70 | 7.20 | 0.07663 | 7.20 | -0.00 | 7.20 |
| P08 | 7.8 | YES | 11.0 | 11.80 | 11.03 | 0.07668 | 11.03 | +0.03 | 10.96 |
| P09 | 8.6 | NO | 3.5 | 3.90 | 3.65 | 0.07669 | 3.65 | +0.15 | 3.74 |
| P10 | 9.2 | NO | 5.8 | 5.20 | 4.86 | 0.07612 | 4.83 | -0.97 | 4.81 |
| P11 | 10.1 | NO | 9.0 | 9.70 | 9.07 | 0.07671 | 9.07 | +0.07 | 9.05 |
| P12 | 10.7 | NO | 6.5 | 7.10 | 6.64 | 0.07674 | 6.64 | +0.14 | 6.66 |
| P13 | 11.8 | NO | 10.5 | 11.40 | 10.65 | 0.07685 | 10.68 | +0.18 | 10.64 |
| P14 | 12.6 | NO | 3.2 | 3.60 | 3.36 | 0.07670 | 3.37 | +0.17 | 3.47 |
| P15 | 13.5 | YES | 8.2 | 9.10 | 8.51 | 0.07695 | 8.54 | +0.34 | 8.52 |

## 5. Generated Visual Diagnostics

- Scatter Plot: [`depth_pred_vs_measured_scatter.png`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/depth_pred_vs_measured_scatter.png)
- Residual Error Plot: [`depth_residual_error_plot.png`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/depth_residual_error_plot.png)
- LOO Coefficient Variation Plot: [`depth_kdisp_fold_variation.png`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/depth_kdisp_fold_variation.png)
