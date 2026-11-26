# Robust Relative Depth Extraction & Adaptive Road Reference Methodology (Phase 5A)

> [!IMPORTANT]
> **Relative Units Only — Zero Production Changes:** [experiments/robust_relative_depth.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/robust_relative_depth.py) is an isolated experimental research module. Production files (`roadvision_engine.py`, `server.py`, `config/camera_params.json`, frozen datasets, checkpoints, and `k_disp = 0.082`, `depth_exponent = 1.35`) remain untouched. All outputs represent **relative inverse-depth units** from `MiDaS_small` (NOT metric centimeters, physical area, or volume).

---

## 1. Architecture & Color-Space Fix

- **`MiDaSDepthRunner`** loads cached TorchHub `intel-isl/MiDaS` (`MiDaS_small`) and `small_transform` on CUDA (`torch.amp.autocast("cuda")`) or CPU.
- **BGR-to-RGB Fix:** OpenCV frames (`cv2.VideoCapture` / `cv2.imread`) are explicitly converted via `cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)` before `self.transform`, correcting the production BGR ordering mismatch audited in [existing_depth_pipeline_audit.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/existing_depth_pipeline_audit.md).
- Predictions are bilinearly upsampled back to `(H, W)` as `float32`.

---

## 2. Adaptive Neighbor-Excluded Annular Road Reference (`build_adaptive_annular_region`)

Configured via `RobustDepthConfig`:
- **Scale & Aspect-Aware Margin:** Instead of the legacy `max(15, min(25, ...))` fixed pixel cap, outer margins scale proportionally with bounding-box dimensions (`bw, bh`):
  `mx = clamp(round(0.45 * bw), 16, 120)` and `my = clamp(round(0.45 * bh), 16, 120)`.
- **Cavity Rim Buffer (`inner_pad_ratio = 0.05`):** Expands the inner exclusion box slightly around `(x1, y1, x2, y2)` so sloped pothole lip edges do not contaminate reference road pixels.
- **Horizon Guard:** Constrains `ry1 >= round(H * 0.15)` when `y1` sits below the `0.15` horizon cutoff line (`horizon_cutoff_ratio = 0.15`), avoiding sky contamination without forcing an invalid floor when detections sit near `horizon_y`.
- **Neighbor Box Exclusion (`exclude_neighbors = True`, `neighbor_pad_px = 4`):** Masks out all other detected boxes inside the outer patch (`±4 px` buffer) so adjacent potholes do not bias the fitted road surface.
- **Finite & Geometric Accounting:** Masks non-finite (`NaN`/`Inf`) pixels, records `annular_retention_ratio = valid_ref_samples / ideal_ring_px`, computes 4-sector spatial support around box center (`top`, `bottom`, `left`, `right` axial sectors), and detects boundary clipping within `boundary_trunc_px = 2` of left/right/bottom/horizon edges.

---

## 3. Five Reference-Surface Strategies (`fit_reference_surface`)

For all planar strategies on normalized patch coordinates $y_n, x_n \in [-1, 1]$ (`_normalize_coords`, preventing ill-conditioned pixel matrices), the reference surface is modeled in 2D inverse-depth image space:

$$Z_{\text{ref}}(y_n, x_n) = a\,y_n + b\,x_n + c$$

1. **`median_baseline`:** Constant reference $Z_{\text{ref}} = \text{median}(Z_{\text{ring}})$ over the adaptive neighbor-excluded ring.
2. **`existing_annular`:** Reproduces production `AnnularDonutDepthExtractor` geometry (`margin` capped `[15, 25] px`, unnormalized pixel coordinates `(y, x)`, no neighbor exclusion, no inner pad) for controlled ablation alongside its `legacy_raw_p85` output.
3. **`ols_plane`:** Closed-form least squares `np.linalg.lstsq([yn, xn, 1], Z_ring)` with explicit numerical rank (`rank < 3`) and singular-value condition-number guard (`cond > 1e4`).
4. **`ransac_plane` (Deterministic Local-Sigma MSAC + OLS Refinement):**
   - Subsamples deterministically up to `ransac_sample_limit = 1500` pixels (`seed = 42`).
   - Estimates intrinsic pixel-to-pixel surface noise using neighbor finite differences on the valid ring (`_estimate_local_surface_sigma`):
     $$\hat{\sigma}_{\text{local}} = \max(10^{-4},\; 1.048 \cdot \text{median}(|\Delta Z_{\text{adj}}|))$$
     This makes the initial consensus scale immune to step discontinuities (e.g., curbs or grass shoulders) that inflate global MAD/IQR.
   - Threshold $\tau = \max(10^{-3},\; 3.0\,\hat{\sigma}_{\text{local}},\; 0.005 \cdot \text{IQR}(Z_{\text{sub}}))$.
   - Evaluates `ransac_iters = 120` non-collinear 3-point planes (`det2 >= 1e-4`), maximizing inlier count and breaking ties by truncated L1 MSAC loss $\sum \min(|r|, \tau)$.
   - Refines via OLS on full-ring inliers within `max(tau, 2.5 * 1.4826 * mad_inl)` when `inliers >= 6` and well-conditioned.
5. **`huber_irls_plane` (L1 Warm-Start + IRLS):**
   - Evaluates 32 deterministic 3-point planes (`seed + 1`) and picks the lowest L1 residual candidate (or OLS if lower) as warm-start so isolated step outliers do not immediately corrupt initial residuals.
   - Runs up to `huber_max_iters = 20` Iteratively Reweighted Least Squares updates with Huber constant `huber_k = 1.345` and scale $\sigma = \max(\hat{\sigma}_{\text{local}}, 10^{-4}, 1.4826 \cdot \text{MAD}(r))$, solving $\text{diag}(\sqrt{w}) A c = \sqrt{w} z$ until $\|\Delta c\|_\infty < 10^{-5}$.

---

## 4. Cavity Residual Polarity & Robust Statistics (`compute_cavity_statistics`)

In MiDaS relative inverse depth, **higher values mean closer to the camera** and **lower values mean farther away / deeper**. Therefore, the signed cavity residual inside the candidate box is defined as:

$$r(y, x) = Z_{\text{ref}}(y, x) - Z(y, x)$$

- $r > 0$: pixel is deeper (farther) than the surrounding reference road surface (**expected pothole cavity**).
- $r < 0$: pixel appears closer/higher than the fitted plane (bump, raised object, specular sky/water mirror, or plane misfit).

Computed summaries:
- `median_residual`, `iqr_residual` (`P75 - P25`), `mad_residual`
- `p75_residual`, `p85_residual`, `p90_residual`
- `positive_depression_frac` ($\text{frac}(r > 0)$)
- `above_noise_depression_frac` ($\text{frac}(r > \max(10^{-4}, 1.4826 \cdot \text{ref\_mad}))$)
- `scale_normalized_p75` ($P_{75}(r) / \max(10^{-4}, 1.4826 \cdot \text{ref\_mad})$)

---

## 5. Eleven-Point Quality Assessment & Strict Invalidity (`assess_measurement_quality`)

Instead of clamping invalid or noisy results to `0.0 cm`, failed checks record explicit `failure_reasons` and return `is_valid = False` with `None` for public `relative_cavity_stats` residual metrics (raw diagnostic stats remain separated for plots/debugging only):

| # | Check | Condition Triggering Rejection (`failure_reasons`) or Warning | Default Threshold |
| :--- | :--- | :--- | :--- |
| 1 | `invalid_or_nonfinite_depth` | Low finite crater coverage, tiny crater box, or excessive NaN/Inf in ring | `finite_crater_ratio < 0.90`, `valid_crater_px < 16`, or `nonfinite_ring_px > max(10, 0.10 * ideal_ring_px)` |
| 2 | `insufficient_reference_samples` | Too few valid surrounding pixels | `< 40` (`< 20` for `existing_annular`) |
| 3 | `poor_spatial_support` | One-sided ring support (exempt on `existing_annular`) | `populated_quadrants < 2` (`>=5` px/quad) or `max_quadrant_frac > 0.80` |
| 4 | `degenerate_reference_plane` | Collinear/ill-conditioned plane fit (`median_baseline` exempt) | `rank < 3` or `cond > 1e4` |
| 5 | `low_inlier_ratio` | Reference surface strongly violated (`ransac_plane`, `huber_irls_plane`) | `inlier_ratio < 0.45` |
| 6 | `overlapping_neighbors_insufficient_pavement` | Neighbor exclusion removed excessive ring (`neighbors_excluded_from_ring` warning otherwise) | `annular_retention_ratio < 0.25` |
| 7 | `high_reference_fit_residual` | Optional hard MAD cap | `ref_mad > 1e6` |
| 8 | `unexpected_residual_polarity` | Flat pavement, upside-down sign, or cavity signal below noise floor (`noise_eps = max(1e-5, 0.5*1.4826*ref_mad)`) | `p75 <= noise_eps`, `pos_strong_frac < 0.22`, or negative tail dominates weak `p75 <= 2*noise_eps` |
| 9 | `unstable_cavity_residuals` | High residual scatter without consistent positive majority | `iqr > max(3.0 * p75, 1.0)` and `positive_depression_frac < 0.50` |
| 10 | `frame_boundary_truncation` | Warning if touching frame/horizon border (`<=2 px`); hard reject if `>=2` edges clipped and `<2` quadrants | `boundary_trunc_px = 2` |
| 11 | `suspected_water_reflection` | `SpecularAquaplaneDetector.inspect_patch` flags puddle glare | Rejects when `reject_water_patches = True` |

**Heuristic Reliability Score (Descriptive Only):** When `is_valid=True`, computes $0.25 \cdot (\min(1, Q/4) + \text{inlier\_ratio} + \min(1, \text{pos\_frac}/0.80) + \min(1, \text{snr}_{P75}/4.0)) \in [0, 1]$, and `0.0` when invalid. **Disclaimer:** This is a deterministic heuristic summary — **NOT a calibrated probability of physical depth accuracy.**

## 6. Known Mathematical Limitations

1. **2D Image Plane Approximation:** Fits $Z_{\text{ref}}(y_n, x_n)$ linear in normalized *pixel* coordinates rather than true 3D metric geometry; non-linear inverse depth perspective warp over very tall bounding boxes (`my` up to `120 px`) or curved crowns deviates from exact planarity.
2. **Unknown Monocular Scale & Shift:** MiDaS predictions carry arbitrary affine scale $s > 0$ and shift $t$; without camera extrinsics or depth sensors, relative residuals $s \cdot \Delta Z_{\text{true}}$ cannot be converted to centimeters.
3. **Rectangular Bounding-Box Contamination:** Corners of rectangular YOLO detections contain intact pavement (`Flat corner` effect), pulling `median_residual` downward relative to `P75`/`P85` unless polygon masks isolate the crater.
4. **Water Specularity:** Puddle reflections often invert monocular depth gradients, requiring explicit rejection rather than numeric estimation.
