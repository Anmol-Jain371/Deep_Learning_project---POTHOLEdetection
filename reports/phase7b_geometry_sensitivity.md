# RoadVision AI — Phase 7B: Camera Geometry Sensitivity & Approach-Drift Impact

> [!IMPORTANT]
> **Critical Finding:** Substituting a dynamic vanishing-point horizon proxy (`D_dynamic_fallback`) into RoadVision AI's existing `MetricDepthCalibrator` **slightly worsens** within-track `depth_cm` variance (`0.396 → 0.404` mean CV, P99 `0.930 → 1.361`) and **fails to resolve** monocular approach-distance drift (`|Spearman rho| = 0.473` across `19` approaching tracks). Dynamic horizon estimation alone cannot deliver reliable physical metric sizing.

---

## 1. Sensitivity of `Z_road_m` and `depth_cm` to `horizon_ratio` (Task 7)

Using the exact `MetricDepthCalibrator` formulas for `sedan_dashcam` (`1280×720`, `H_cam=1.25 m`, `VFOV=60°`, `depth_exponent=1.35`, `k_disp=0.082`) in [geometry_sensitivity_analysis.png](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_viz/geometry_sensitivity_analysis.png):

| Box Bottom `y2 / H` | Metric at `h=0.30` | `h=0.35` | `h=0.40` (Default) | `h=0.44` (`+4%` shift) | `h=0.48` (`+8%` shift) | Multiplier Range (`0.30 → 0.48`) |
|---|---|---:|---:|---:|---:|---:|
| **`0.46`** (far road) | `6.77 m` (`0.27×`) | `9.72 m` (`0.43×`) | `18.04 m` (`1.00×`) | `51.28 m` (**`4.10×`**) | `51.96 m` (`4.17×` clamped) | **`15.7×` depth swing** |
| **`0.55`** (mid-far) | `4.33 m` (`0.50×`) | `5.38 m` (`0.67×`) | `7.22 m` (`1.00×`) | `9.74 m` (**`1.50×`**) | `15.28 m` (**`2.75×`**) | **`5.5×` depth swing** |
| **`0.70`** (mid-near) | `2.71 m` (`0.68×`) | `3.08 m` (`0.81×`) | `3.61 m` (`1.00×`) | `4.15 m` (`1.21×`) | `4.90 m` (`1.51×`) | `2.2×` depth swing |
| **`0.85`** (near foreground) | `1.97 m` (`0.76×`) | `2.16 m` (`0.86×`) | `2.41 m` (`1.00×`) | `2.63 m` (`1.13×`) | `2.92 m` (`1.30×`) | `1.7×` depth swing |

Because `depth_cm ∝ (1 / dy)^1.35`, any small horizon estimation noise or camera bounce on far detections (`y2_ratio <= 0.55`) explodes non-linearly: just a **`+0.04` horizon shift (`0.40 → 0.44`) increases computed `depth_cm` by `4.10×`** at `y2=0.46` and **`1.50×`** at `y2=0.55`.

---

## 2. Resolution Asymmetry from Hardcoded `max(15.0, y2 - horizon_y)`

In `roadvision_engine.py`, `dy` is clamped to **`15.0` raw pixels** regardless of video height (`360p` vs `720p`):

| Normalized `y2 / H` | `Z_road` @ `360p` (`dy_360` min 15px) | `Z_road` @ `720p` (`dy_720` min 15px) | `depth_cm(720p) / depth_cm(360p)` |
|---:|---:|---:|---:|
| **`0.42`** | `25.98 m` (clamped at 15px vs 7.2px) | `51.96 m` (clamped at 15px vs 14.4px) | **`2.549×`** |
| **`0.43`** | `25.98 m` (clamped at 15px vs 10.8px) | `36.08 m` (unclamped `21.6 px`) | **`1.558×`** |
| `0.45` | `21.65 m` | `21.65 m` | `1.000×` |
| `0.65` | `4.33 m` | `4.33 m` | `1.000×` |

Using a pixel constant rather than a resolution-normalized minimum ratio (`e.g. 0.03 * H`) produces up to a **`2.55×` depth output discontinuity** purely from changing stream resolution!

---

## 3. Camera Profile Assumption Spread (`720p`, `y2 / H = 0.65`)

| Profile | `H_cam` / `VFOV` / `h` | `Z_road` @ `y2=0.65` | Relative `depth_cm` Multiplier (`raw_disp=1`) |
|---|---|---:|---:|
| `light_mobility_twowheeler` | `0.95 m` / `55°` / `0.38` | `3.369 m` | `0.423` |
| `sedan_dashcam` | `1.25 m` / `60°` / `0.40` | `4.330 m` | `0.593` |
| `suv_crossover` | `1.55 m` / `65°` / `0.42` | `5.276 m` | `0.774` |
| `commercial_truck_bus` | `2.20 m` / `70°` / `0.46` | `8.256 m` | **`1.417`** (**`3.35×`** higher than two-wheeler) |

Selecting an inaccurate preset alters downstream depth output by more than **`3×`** even with identical pixel detections and disparities.

---

## 4. Empirical Matched-Track Approach-Drift Test (`DAv2 336`, `elliptical_core`)

To test whether substituting dynamic vanishing-point horizon (`D_dynamic_fallback`) reduces within-track instability or approach-distance drift under identical `k_disp=0.082, depth_exponent=1.35`, all `33` multi-frame valid tracks (`>= 5` valid measurements across `ruralRoad_202f`, `rawSurvey_160f_10fps`, `rangoonRoad_f200_399`, including `19` approaching tracks) were evaluated:

| Quantity Tested (`n=33` tracks, `19` approaching) | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|
| Within-track CV — Raw `DAv2 336` `scale_norm_p75` | **`0.345`** | `0.334` | `0.761` | `1.362` |
| Within-track CV — `depth_cm` with Fixed `h=0.40` | `0.396` | `0.394` | `0.826` | `0.930` |
| Within-track CV — `depth_cm` with Dynamic Horizon (`D`) | **`0.404`** | `0.392` | `0.875` | **`1.361`** |
| Approaching `|Spearman ρ(t, depth_cm)|` — Fixed `h=0.40` | `0.601` | `0.600` | `0.957` | `0.975` |
| Approaching `|Spearman ρ(t, depth_cm)|` — Dynamic (`D`) | `0.473` | `0.427` | `0.926` | `0.957` |
| Approaching Signed `Spearman ρ` (Fixed vs Dynamic) | `-0.596` vs `-0.468` | `-0.600` vs `-0.427` | — | — |

### Physical Interpretation
1. Multiplying raw DAv2 relative cavity disparity (`CV 0.345`) by the uncalibrated geometric factor `(H_cam * fy / dy)^1.35` **increases** within-track inconsistency (`0.345 → 0.396`).
2. Replacing fixed `0.40` with dynamic horizon `D` slightly **worsens** overall within-track CV (`0.396 → 0.404`, tail P99 `0.930 → 1.361`) because small jitter in estimated `horizon_y` propagates nonlinearly into `1 / dy^1.35`.
3. Although mean approaching correlation softened slightly (`-0.596 → -0.468`), strong monotonic drift persists as potholes move closer (`|ρ| = 0.473`), proving that **dynamic horizon estimation does NOT make RoadVision AI's monocular depth predictions distance-invariant or metric.**
