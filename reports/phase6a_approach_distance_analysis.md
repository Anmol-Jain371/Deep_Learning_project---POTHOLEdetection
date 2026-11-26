# Phase 6A — Approach-Distance & Perspective Normalization Investigation

Data: [phase6a_depth_comparison.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_depth_comparison.json) (`Task 8`), [phase6a_per_track.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_per_track.csv), and [phase6a_perf_and_approach_drift.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_viz/phase6a_perf_and_approach_drift.png).

> [!CAUTION]
> **Not Physical Distance Correction:** Neither statistical dispersion normalization nor bounding-box geometric weighting converts uncalibrated relative inverse depth into physical metric distance or physical pothole depth (cm). No undocumented exponents or historical manifest constants were used.

---

## 1. Candidate Normalizations & Affine Invariance Verification

Tested across matched ByteTrack trajectories (`>= 5` valid observations per track: `15` tracks / `213` obs for `midas_elliptical`; `16` tracks / `355` obs for `dav2_elliptical`) tracking bounding-box pixel area, image-relative bottom coordinate `y2 / H`, track age, and camera profile (`sedan_dashcam`; **true intrinsics/extrinsics unavailable**):

1. `raw_p75`: unnormalized P75 relative cavity residual.
2. `scale_norm_p75`: `p75 / max(1e-5, 1.4826 * ref_mad)`.
3. `frame_iqr_norm_p75`: `p75 / IQR(depth_frame)`.
4. `area_weighted_iqr_p75`: `(p75 / IQR(depth_frame)) / sqrt(box_area / (H * W))`.
5. `y2_weighted_iqr_p75`: `(p75 / IQR(depth_frame)) / max(0.05, y2 / H - 0.15)`.

**Direct Affine Transformation Check (`5.0 * depth + 100.0`):**
- `raw_p75` ratio = **`5.000003`** (scales linearly with global model scale; not scale-invariant).
- `frame_iqr_norm_p75` abs diff = **`0.000000`**, `area_weighted_iqr_p75` diff = **`0.000002`**, `y2_weighted_iqr_p75` diff = **`0.000001`** (strictly affine-invariant!).
- `scale_norm_p75` caveat: affine-invariant whenever sensor/surface noise produces `1.4826 * ref_mad >> 1e-5` (Phase 5B real MiDaS check `0.00000`), but breaks down on mathematically **zero-noise** synthetic planes where `ref_mad == 0` hits the fixed `1e-5` floor.

---

## 2. Within-Track Drift vs Pooled Cross-Pothole Correlations

Distinguishing **within-track approach evolution** (mean per-track Spearman $\rho$ across approaching frames) from **pooled correlation across distinct potholes** (which conflates camera distance with genuinely larger vs smaller craters):

| Normalization Candidate | Model (`Elliptical`) | Within-Track `ρ` vs Box Area | Within-Track `ρ` vs `y2/H` | Within-Track Mean CV | Pooled `ρ` vs Area | Pooled `ρ` vs `y2/H` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `raw_p75` | `MiDaS` | `+0.2227` | `-0.0124` | `0.5632` | `+0.3957` | `+0.1186` |
| | `DAv2` | `+0.1150` | **`+0.2992`** | `0.3187` | `+0.3616` | `+0.1338` |
| `scale_norm_p75` | `MiDaS` | `+0.0852` | `-0.0138` | **`0.5054`** | `+0.1804` | `+0.0214` |
| | `DAv2` | `-0.1297` | `-0.0684` | **`0.2933`** | `+0.2483` | `+0.3622` |
| `frame_iqr_norm_p75` | `MiDaS` | `+0.2326` | `-0.0099` | `0.5471` | `+0.3955` | `+0.1175` |
| | `DAv2` | `+0.1195` | `+0.3052` | `0.3203` | `+0.4307` | `+0.1420` |
| `area_weighted_iqr_p75` | `MiDaS` | **`+0.0389`** | `-0.1877` | `0.5150` | `+0.1336` | `-0.0967` |
| | `DAv2` | `-0.1768` | **`+0.0318`** | `0.3053` | `-0.0878` | `-0.0794` |
| `y2_weighted_iqr_p75` | `MiDaS` | `+0.0973` | `-0.1624` | `0.5411` | `+0.2342` | `-0.1482` |
| | `DAv2` | **`-0.0926`** | `+0.0680` | `0.2957` | `+0.2311` | `-0.1816` |

---

## 3. Empirical Conclusions

1. **Global Frame IQR Does NOT Remove Approach Drift:** Dividing by `IQR(depth_frame)` behaves nearly identically to `raw_p75` (`MiDaS ρ_area = +0.2326` vs `+0.2227`; `DAv2 ρ_y2 = +0.3052` vs `+0.2992`), because the whole-scene inverse-depth range stays approximately constant as a single pothole rolls closer across the foreground.
2. **Depth Anything V2 Small Tracks Are Substantially More Stable Baseline-to-Baseline:** Across every normalization, DAv2 within-track temporal CV sits at **`0.293–0.320`** versus **`0.505–0.563`** for MiDaS (`~43%` lower dispersion!).
3. **Geometric Area / `y2` Weighting Weakens Monotonic Growth Without Over-Claiming Metric Scale:** Weighting `frame_iqr_norm_p75` by `1 / sqrt(area_frac)` or `1 / (y2/H - 0.15)` pulls within-track approach correlation near zero (`+0.039` MiDaS / `-0.093` DAv2) while preserving exact affine invariance (`diff <= 2e-6`). However, slight negative sign flips (`-0.177` DAv2 area) demonstrate that simplified $1/\sqrt{A}$ proxies slightly over-correct aspect/box jitter, confirming true physical severity normalization requires known camera extrinsics or calibrated depth targets.
