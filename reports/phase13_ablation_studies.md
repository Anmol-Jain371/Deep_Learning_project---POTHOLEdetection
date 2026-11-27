# Phase 13 — Ablation Studies & Component Synthesis

**Run ID:** `phase13_full_20261127`  
**Artifacts:** `result_geometry.json`, `result_depth_cavity.json`, `result_scheduler_temporal.json`, `tables/*.csv`, `figures/fig2..fig4`

---

## 1. Reference-Plane Estimation Ablation (`N = 45` synthetic cases, `seed = 42`)

| Method | Valid Count | Gate Concordance (%) | Invalid Rejection (%) | Mean Plane RMSE | Outlier Mean RMSE | Mean Inlier Ratio | Mean Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_legacy_annular` | 45/45 | 77.78 | 0.00 | N/A | N/A | N/A | 0.638 |
| `B_ols_plane` | 35/45 | 100.00 | 100.00 | 0.000630 | 0.001065 | 0.9839 | 2.670 |
| `C_ransac_baseline` | 35/45 | 100.00 | 100.00 | **0.000285** | **0.000029** | 0.9965 | 9.014 |
| `D_optimized_msac_ransac` | 35/45 | 100.00 | 100.00 | 0.000396 | 0.000131 | 0.9961 | **4.514** |

- **Interpretation:** Legacy annular extraction lacks plane-validity / raised-surface gates (`0%` invalid rejection). Under injected surrounding outliers (`10–40%`), both RANSAC/MSAC variants cut plane-recovery error by `8.1x–36.7x` relative to OLS while `D_optimized_msac_ransac` cuts latency **2.00x** vs baseline RANSAC (`4.514 ms` vs `9.014 ms`). Synthetic plane RMSE validates planar fit stability ONLY—never physical pothole depth.

## 2. Depth Model & Resolution Ablation (Sequential on RTX 2050, `17` frames, `42` candidate boxes)

| Arm | Input Size | Precision | Mean Inf (ms) | P95 Inf (ms) | Peak Alloc VRAM (MB) | Fresh Valid Coverage (%) | Mean Inlier Ratio | Spearman vs DAv2-518 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `midas_256_fp32` | 256 | `fp32` | 16.451 | 20.903 | 127.43 | 28.57 (12/42) | 0.6357 | 0.9794 |
| `midas_256_fp16` | 256 | `fp16_autocast` | 21.410 | 27.770 | 163.49 | 28.57 (12/42) | 0.6430 | 0.9795 |
| `dav2_266_fp16` | 266 | `fp16_autocast` | 25.801 | 30.503 | 243.42 | 50.00 (21/42) | 0.7919 | 0.9987 |
| `dav2_336_fp16` | 336 | `fp16_autocast` | 48.175 | 48.681 | 274.69 | 59.52 (25/42) | 0.7993 | 0.9992 |
| `dav2_336_fp32` | 336 | `fp32` | 60.256 | 60.438 | 218.41 | 61.90 (26/42) | 0.8111 | 0.9992 |
| `dav2_392_fp16` | 392 | `fp16_autocast` | 68.817 | 69.134 | 313.59 | 66.67 (28/42) | 0.8146 | 0.9995 |
| `dav2_518_fp16` | 518 | `fp16_autocast` | 147.008 | 147.389 | 409.69 | 66.67 (28/42) | 0.8398 | 1.0000 |

- `DAv2 336` retains `99.92%` rank concordance against the `518` consistency reference at `3.05x` lower inference latency (`48.18 ms` vs `147.01 ms`) and more than doubles valid box coverage over `MiDaS 256` (`59.5%–61.9%` vs `28.6%`). Agreement with `DAv2 518` is an internal relative-consistency check ONLY, NOT metric depth ground truth.

## 3. Cavity Extraction & Pothole440 Grouped Segmentation
- **Over identical `DAv2 336 fp16` depth maps (`42` boxes):** `full_box` (`61.9%` valid, `7.490 ms`), `rect_trim_20` (`61.9%` valid, `5.671 ms`), `elliptical_core` (`59.52%` valid, `6.686 ms`), with identical annular `mean_inlier_ratio = 0.7993`.
- **`dataset_pothole440_seg_grouped/test` (`N = 15` unaugmented close-up originals):** `yolov8n-seg` reaches `mean_mask_iou = 0.5977`, `median_mask_iou = 0.6906`, `mean_dice = 0.6763`. Close-up overhead imagery is domain-shifted from dashcam road perspectives.

## 4. Adaptive Depth Scheduling (Frozen `150`-frame Stream, `464` observations)

| Strategy | Depth Calls / 150f | Call Reduction (%) | Fresh Coverage (%) | Effective Avail. (%) | Expired Obs | Mean Age (s) | Sim. Mean Frame (ms) | Sim. FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_always_on` | 150 | 0.00 | 66.59 | 66.59 | 0 | 0.0000 | 86.94 | 11.50 |
| `B_candidate_gated` | 144 | 4.00 | 66.59 | 66.59 | 0 | 0.0000 | 84.30 | 11.86 |
| `C_fixed_interval` (K=3) | 48 | 68.00 | 22.20 | 68.32 | 39 | 0.0515 | 38.09 | 26.25 |
| `D_track_aware` | 95 | 36.67 | 45.91 | 69.61 | 0 | 0.0220 | 60.96 | 16.40 |
| `E_budget_aware` | 80 | 46.67 | 36.21 | 71.34 | 0 | 0.0387 | 53.67 | 18.63 |

- Notice `C_fixed_interval` reduces calls most (`68%`) at the cost of `39` expired observations and `22.2%` fresh coverage, whereas `D_track_aware` and `E_budget_aware` avoid expired gaps (`0` expired) while cutting depth invocations by `36.7%` and `46.7%`.

## 5. Temporal Filtering Ablation (`25`-step synthetic sequence with gap & expiry)
- `raw`: `mean_diff = 0.03500`, `hides_expired = False`
- `legacy_ema_zero_clamp`: `mean_diff = 0.02563`, **`hides_expired = True` (`0/2` expired steps masked null)**
- `validity_aware_ema`: `mean_diff = 0.00862`, `hides_expired = False` (`2/2` masked null)
- `rolling_median_validity_ema`: `mean_diff = 0.00574`, `hides_expired = False` (`2/2` masked null; slowest step tracking at `f12 = 0.13294` vs target `0.22`)

## 6. Phase 10 Relative Geometry Serialization Fidelity
- Synthetic validation (`sloped + 10×12 NaN block`): `raw_mean_residual = 0.010664` vs `downsampled_mean_residual = 0.011059` (`abs diff = 0.000396`), `reference_mask_true_cells = 1,141`, `cavity_mask_true_cells = 304`, `null_cells_preserved = 9`, `cross_section_gap_preserved = true`, `snapshot_json_kb = 91.03 KB`, `generation_overhead_ms = 98.38 ms`, **`extra_depth_inference_calls = 0`**.
