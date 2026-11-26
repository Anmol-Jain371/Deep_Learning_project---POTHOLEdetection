# Synthetic Mathematical Validation of Relative Depth Strategies (Phase 5A)

Executed via [test_relative_depth_synthetic.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/test_relative_depth_synthetic.py) and logged in [synthetic_relative_depth_validation.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/synthetic_relative_depth_validation.json). All synthetic depths emulate MiDaS inverse-depth polarity (higher = closer, cavity = lower than surrounding road plane). Unit tests verifying core invariants pass `5/5` in [test_robust_relative_depth.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_robust_relative_depth.py).

> [!NOTE]
> These metrics measure **mathematical reference-plane recovery and relative-residual accuracy on noise/outlier-controlled 2D synthetic inverse-depth surfaces** — they do **not** represent real-world metric accuracy in centimeters.

---

## 1. Aggregate Comparison Across All 12 Synthetic Scenarios

| Method | Validity Classification Accuracy | Mean Norm Ref-Plane RMSE | Max Norm Ref-Plane RMSE | Mean Norm P75 Abs Err | Max Norm P75 Abs Err | Mean Norm Med Abs Err | Mean Extraction Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `median_baseline` | `10/12` | `1.4926` | `3.2475` | `1.2222` | `3.6440` | `1.3569` | `1.7604` |
| `existing_annular` | `12/12` | `0.5935` | `3.2643` | `0.6554` | `3.9328` | `0.6456` | `2.7568` |
| `ols_plane` | `12/12` | `0.5512` | `3.6860` | `0.6186` | `4.3566` | `0.6345` | `2.3298` |
| **`ransac_plane`** | **`12/12`** | **`0.0003`** | **`0.0028`** | **`0.0003`** | **`0.0030`** | **`0.0003`** | `8.7109` |
| `huber_irls_plane` | `12/12` | `0.4202` | `4.1864` | `0.5023` | `5.0071` | `0.5022` | `6.7634` |

*(Normalized metrics divide errors by the affine scale factor `5.0` on `affine_scale_shift_5x` so all 12 scenarios weigh on a common synthetic unit scale.)*

---

## 2. Scenario-by-Scenario Breakdown (Ref-Plane RMSE / P75 Abs Error)

| Scenario | Expected Valid | True P75 | `median_baseline` | `existing_annular` | `ols_plane` | `ransac_plane` | `huber_irls_plane` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. `flat_no_depression` | False | `0.0000` | Invalid / `0.0` | Invalid / `0.0` | Invalid / `0.0` | Invalid / `0.0` | Invalid / `0.0` |
| 2. `flat_with_cavity` | True | `1.4287` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` |
| 3. `inclined_plane_with_cavity` | True | `1.4287` | **FAIL (Invalid)** `1.9730` / `1.3087` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` |
| 4. `irregular_cavity_shape` | True | `1.2426` | `1.2436` / `0.7124` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` |
| 5. `gaussian_noise` | True | `1.4197` | `1.2436` / `0.6461` | `0.0018` / `0.0016` | `0.0024` / `0.0027` | `0.0028` / `0.0030` | `0.0028` / `0.0031` |
| 6. `reference_outliers_20pct` | True | `1.4287` | `1.6602` / `1.7440` | `2.4394` / `2.4352` | `1.8236` / `1.8267` | **`0.0` / `0.0`** | `0.0126` / `0.0127` |
| 7. `missing_pixels_sparse_8pct` | True | `1.4287` | `1.0328` / `0.4313` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` |
| 8. `missing_pixels_crater_fatal` | False | `0.0000` | Invalid (`invalid_or_nonfinite_depth`) | Invalid | Invalid | Invalid | Invalid |
| 9. `neighboring_cavity` | True | `1.4287` | `1.5351` / `1.5440` | `0.2298` / `0.1844` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` |
| 10. `partial_boundary_clipping` | True | `1.9340` | **FAIL (Invalid)** `1.7470` / `1.5474` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` |
| 11. `contaminated_curb_block` | True | `1.4287` | `3.2475` / `3.6440` | `3.2643` / `3.9328` | `3.6860` / `4.3566` | **`0.0` / `0.0`** | `4.1864` / `5.0071` |
| 12. `affine_scale_shift_5x` | True | `7.1434` | `6.2178` / `3.2201` (raw) | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` | `0.0` / `0.0` |

*(Wait-free zero error denotes exact numerical recovery to 4+ decimal places; `existing_annular` fails to mask neighbor depression in Scenario 9 (`0.2298` RMSE).)*

---

## 3. Key Mathematical Takeaways

1. **Perspective Slope Breakdown (`median_baseline`):** Without planar detrending, sloped road backgrounds create symmetric positive and negative residuals across the box (`unexpected_residual_polarity`), misclassifying genuine craters (`10/12` validity accuracy) and distorting `P75` by `45%–92%`.
2. **Neighbor Leakage (`existing_annular`):** Production's unmasked outer ring overlaps neighboring potholes, pulling the fitted plane down (`0.2298` ref RMSE, `-12.9%` P75 underestimate), whereas neighbor-masked adaptive rings recover `0.0` RMSE cleanly.
3. **High-Leverage Outlier vs Random Outlier Breakdown (`huber_irls_plane` vs `ransac_plane`):**
   - On scattered `20%` ring outliers (`reference_outliers_20pct`), Huber IRLS achieves `0.0126` ref RMSE (`0.89%` P75 error) while `ols_plane` degrades to `1.8236` (`127.9%` P75 error).
   - However, on a contiguous raised **curb/edge block** (`contaminated_curb_block`), Huber IRLS has a $0\%$ leverage breakdown point and gets pulled toward the high-leverage step (`4.1864` ref RMSE, `350.5%` P75 error), whereas `ransac_plane` anchored by neighbor finite-difference `local_sigma` rejects the block completely (`0.0` ref RMSE, `0.0` P75 error, `0.6750` inlier ratio).
4. **Affine Scale & Shift Invariance:** Multiplying the scene by `5x` plus an offset scales valid planar `P75` residuals linearly (`1.4287 -> 7.1434`, exact `0.0` error) and leaves `scale_normalized_p75` strictly invariant.
