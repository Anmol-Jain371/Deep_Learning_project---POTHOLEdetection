# RoadVision AI — Phase 7B: Synthetic Projective Horizon Verification

> [!NOTE]
> All synthetic tests use controlled projective renderings with exact planted vanishing-point coordinates `(vx, vy)` generated in [benchmark_dynamic_horizon.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_dynamic_horizon.py) and unit-tested in [test_dynamic_horizon_estimator.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_dynamic_horizon_estimator.py). Passing synthetic projective tests confirms mathematical geometric correctness under clean perspective assumptions, **NOT** real-world camera calibration or metric depth validity.

---

## 1. Aggregate Variant Comparison (21 Positive Cases + 4 Negative Controls)

| Variant | Positive `VALID` Rate | All-Case Mean / P95 `|err_y|` (norm) | `VALID`-Only Mean / P95 `|err_y|` (norm) | `360p` vs `720p` Norm Diff | Negative Control False `VALID` |
|---|---:|---:|---:|---:|---:|
| **`lsd_320`** | `19 / 21` (**90.48%**) | `0.010` / `0.006` | **`0.003` / `0.006`** | `0.00534` | **`0 / 4`** |
| `hough_320` | `18 / 21` (85.71%) | `0.010` / `0.006` | `0.003` / `0.006` | `0.00167` | **`0 / 4`** |
| `hybrid_320` | `20 / 21` (95.24%) | `0.010` / `0.005` | `0.003` / `0.005` | `0.00305` | **`0 / 4`** |
| `lsd_fullres` | `21 / 21` (**100.0%**) | **`0.001` / `0.002`** | **`0.001` / `0.002`** | **`0.00112`** | **`0 / 4`** |

---

## 2. Per-Scenario Error Breakdown (`lsd_320`)

Observed directly in [phase7b_horizon_results.json](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_horizon_results.json) and visualized in [synthetic_horizon_validation.png](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_viz/synthetic_horizon_validation.png):

| Scenario Group | Test Cases | `lsd_320` State(s) & Reasons | Mean `|err_y|` (norm) | Physical / Projective Takeaway |
|---|---|---|---:|---|
| **`multi_pitch`** | `vy ∈ {0.25, 0.33, 0.40, 0.48, 0.56}` | `5/5 VALID` | `0.00360` | Sub-`0.6%` normalized vertical VP accuracy (`~0.9–1.9 px` at `360p`). |
| **`angled_road`** | `vx ∈ {0.25, 0.38, 0.50, 0.62, 0.75}` (`vy=0.40`) | `5/5 VALID` | `0.00172` | Immune to road curvature/yaw shifts left or right. |
| **`resolution_scaling`** | `640×360` vs `1280×720` (`vp=(0.54, 0.36)`) | `2/2 VALID` | `0.00267` | Normalized horizon differs by only `0.00534` across `2×` resolution change. |
| **`missing_markings`** | Outer curbs only (`no lane dashes`) | `LOW_CONFIDENCE` (`low_inlier_count`) | `0.00473` | Estimator locates VP accurately from curbs alone, AND conservatively flags reduced inlier count. |
| **`partial_occlusion`** | Center truck blocking 44% road width | `VALID` | `0.00549` | Peripheral curb segments preserve stable VP consensus. |
| **`spurious_clutter`** | Vertical poles + horizontal cracks | `VALID` | `0.00622` | Angle gate (`12°–80°`) strips clutter cleanly before RANSAC. |
| **`shadows`** | Harsh dark shadow band across mid-road | `VALID` | `0.00289` | Robust to illumination transitions along continuous road lines. |
| **`heavy_noise`** | Gaussian noise `sigma = 12, 22` | `2/2 VALID` | `0.00299` | MSAC + weighted refit resists severe pixel noise. |
| **`camera_roll`** | `roll_deg = -6°, +6°` | `2/2 VALID` | `0.00295` | **Important Caveat:** VP coordinate `(vx, vy)` remains accurate near center, **but** a flat horizontal line `y = vy` is rotated relative to the true `6°`-tilted horizon toward the image corners! |
| **`incompatible_dual_vp`** | Cross-street VP at `(0.15, 0.55)` | `LOW_CONFIDENCE` (`single_sided_line_support`, `cross_group_unverified`) | `0.14585` | When a competing cross-street dominates, single-sided gating refuses `VALID`, preventing corrupt `VALID` updates to the tracker! |

### Negative Controls (`lsd_320` & All Variants)
All 4 degenerate/empty scenes returned **`0 / 4` false `VALID`s** (`100% UNAVAILABLE`):
- `blank_asphalt` → `UNAVAILABLE` (`insufficient_oriented_lines`)
- `parallel_lines` → `UNAVAILABLE` (`parallel_or_degenerate_lines`)
- `vertical_clutter_only` → `UNAVAILABLE` (`insufficient_oriented_lines`)
- `horizontal_clutter_only` → `UNAVAILABLE` (`insufficient_oriented_lines`)

---

## 3. Temporal Step-Change & Dropout Simulation (55 Frames @ 25 FPS)

Simulated step pitch jump (`vy = 0.34` on frames `0..17` → `12` frames complete blank occlusion `18..29` → `vy = 0.48` on frames `30..54`):

| Policy | Availability Rate | Visible GT Error Mean / P95 (norm) | Frame-to-Frame Jitter Mean (norm) | Behavior Summary |
|---|---:|---:|---:|---|
| `A_fixed` (`0.40`) | `100.0%` | `0.072` / `0.080` | `0.000` | Large static bias whenever camera pitch differs from `0.40`. |
| `B_dynamic_raw` | `78.18%` | `0.004` / `0.005` | `0.000` | Drops completely during occlusion (`12` missing outputs). |
| `C_dynamic_smoothed` | `96.36%` | `0.004` / `0.005` | `0.000` | Holds smooth value across `STALE`, drops only on final `2 EXPIRED` frames. |
| **`D_dynamic_fallback`** | **`100.0%`** | **`0.004` / `0.005`** | `0.003` | Zero dropouts; falls back safely to `0.40` after `max_expired_sec=0.40s` and re-acquires `0.48` immediately after occlusion clears. |
