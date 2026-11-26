# Phase 6A — Core-Aware Cavity Extraction Report

Module: [experiments/core_aware_cavity_extraction.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/core_aware_cavity_extraction.py) | Unit tests: [tests/test_core_aware_cavity_extraction.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_core_aware_cavity_extraction.py) (`4/4` passing) | Benchmark data: [phase6a_depth_comparison.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_depth_comparison.json).

> [!NOTE]
> Synthetic verification measures mathematical residual recovery on 2D surfaces only. Geometric cores are **not segmentation masks** and cannot claim physical-depth accuracy.

---

## 1. Strategies Compared (Reference Plane Held Constant)

While holding the fitted `fit_ransac_optimized` (`stratified_quadrant`, `512`) reference plane strictly identical across arms:
- **`full_box`:** Entire detector box (`100%` interior area).
- **`rect_trim_10`, `rect_trim_20`, `rect_trim_30`:** Symmetric rectangular margin trims (`~64%`, `~36%`, `~16%` box area) rejecting intact pavement corners cleanly (`min_interior_px = 16`).
- **`elliptical_core`:** Inscribed aspect-matched ellipse (`ellipse_shrink = 0.90`, `~63.5%` box area) trimming all four corners symmetrically.
- **`multi_scale_agreement`:** Evaluates nested consistency across `full_box`, `rect_trim_10`, `rect_trim_20`, and `rect_trim_30`; rejects `unstable_multiscale_interior` on sign flips or normalized scale spread `> 0.85`.
- **`spatially_coherent_residual` (`selection_biased = True`):** Largest 8-connected spatial component above `0.25 * noise_floor` (`>= 10%` box area & `>= 16 px`). Strictly separated because thresholded component selection induces upward selection bias if contradictory negative residuals are ignored.

All strategies preserve full-box diagnostics (`full_p50`, `full_p75`, `full_pos_frac`, `full_neg_frac`) and enforce scale-normalized quality gates.

---

## 2. Task 4 Controlled Synthetic Suite (`12 Scenarios`)

| Strategy | Validity Acc (`/12`) | Positive Scenarios (`/10`) | Flat/Raised False Accepts (`/2`) | Mean / Max P75 Err vs Selected True | Mean Box Coverage Ratio | Selection Biased? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `full_box` | `10/12` | `8/10` | `0/2` | `0.0322` / `0.3147` | `1.0000` | No |
| `rect_trim_10` | `10/12` | `8/10` | `0/2` | `0.0299` / `0.2889` | `0.6425` | No |
| **`rect_trim_20`** | **`12/12`** | **`10/10`** | **`0/2`** | `0.0306` / `0.2945` | `0.3600` | No |
| `rect_trim_30` | `11/12` | `9/10` | `0/2` | `0.0304` / `0.2948` | `0.1588` | No |
| `elliptical_core` | `10/12` | `8/10` | `0/2` | `0.0308` / `0.2969` | `0.6348` | No |
| `multi_scale_agreement` | `3/12` | `1/10` | `0/2` | `0.0306` / `0.2945` | `0.3600` | No |
| `spatially_coherent_residual` | `12/12` | `10/10` | `0/2` | `0.0313` / `0.2999` | `0.3100` | **YES** |

**Key Mathematical Trade-offs Observed:**
1. **Perimeter Edge Contamination Recovery (`rect_trim_20`):** On `contaminated_bbox_edges`, `full_box`, `rect_trim_10`, and `elliptical_core` (`0.90` axes reach near the contaminated outer rim) reject cleanly on residual instability (`8/10` pos valid), whereas **`rect_trim_20` clears the outer border (`12/12`)** cleanly without starving the off-center depression (which `rect_trim_30` clipped too tightly, dropping to `11/12`!).
2. **Why `multi_scale_agreement` Over-Rejected Clean Bowls (`3/12`):** For a realistic parabolic crater inside a rectangular bounding box, trimming corners increases P75 from `0.953` (`full_box`) -> `1.974` (`trim_10`) -> `2.687` (`trim_20`) -> `3.141` (`trim_30`), creating natural radial variance `rel_spread > 0.85`. Thus magnitude-spread gating across radial trims conflates healthy depth-profile taper with instability (`unstable_multiscale_interior`).

---

## 3. Dashcam Dev (98 Regions) & Exploratory Video Behavior

On 36 dev frames (`MiDaS_small` reference fixed):
- **`full_box`:** `50/98` (`51.02%`), median `scale_norm_p75 = 2.2984`, `1.22 ms`.
- **`rect_trim_10`:** `53/98` (`54.08%`), median `scale_norm_p75 = 1.9955`, `1.05 ms`.
- **`elliptical_core`:** `53/98` (`54.08%`), median `scale_norm_p75 = 2.0247`, `1.37 ms`.
- **`rect_trim_20` / `30`:** `51/98` (`52.04%`) / `49/98` (`50.00%`) (`unexpected_residual_polarity` increases from `34` to `38` when trimming too aggressively on decentered real potholes).
- **`spatially_coherent_residual` (`selection_biased=True`):** `63/98` (`64.29%`) — explicitly reflects cherry-picking positive blobs when boxes contain real contradictory negative depth regions; rejected as a primary estimator.

On the **360-frame video benchmark (`575` regions)**:
- For **`MiDaS_small`**, `elliptical_core` slightly *lowered* coverage (`37.04%` vs `39.30%`), proving urban `rawSurvey` MiDaS rejections originate inside the pit itself rather than bbox edges.
- Conversely, for **`Depth Anything V2 Small`**, `elliptical_core` cleanly **boosted valid coverage from `57.91%` (`333/575`) to `61.74%` (`355/575`)** (`45.54%` vs `37.50%` on Rangoon, `72.49%` vs `68.28%` on Rural), demonstrating that geometric corner trimming helps **only when the underlying depth backbone actually resolves interior cavity geometry cleanly**.
