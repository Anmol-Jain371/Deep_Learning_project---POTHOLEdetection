# RoadVision AI — Phase 7B: Existing Camera Geometry & Calibration Feasibility Audit

> [!IMPORTANT]
> **Scientific Epistemology Notice:** This audit inspects the exact code paths in [roadvision_engine.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_engine.py) and [camera_params.json](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/config/camera_params.json). RoadVision AI currently possesses **zero** independently verified physical camera intrinsics, extrinsics, or ground-truth pothole metric depth measurements. All distance and centimeter numbers produced by `MetricDepthCalibrator` are uncalibrated model outputs, not verified physical measurements. No production files were modified.

---

## 1. Audited Files & Implemented Equations

Inspecting [roadvision_engine.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_engine.py) (`load_camera_config`, `MetricDepthCalibrator`, `RoadVisionPipeline`) and [camera_params.json](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/config/camera_params.json) confirms the exact production geometry model:

### 1.1 Static Profile Configuration
`config/camera_params.json` stores four static presets plus shared empirical constants:

| Profile Key | Assumed `camera_height_m` (\(H_{\text{cam}}\)) | Assumed `vfov_deg` | Static `horizon_ratio` | Assumed `k_disp` |
|---|---:|---:|---:|---:|
| `sedan_dashcam` | `1.25` m | `60.0°` | `0.40` | `0.082` |
| `suv_crossover` | `1.55` m | `65.0°` | `0.42` | `0.082` |
| `commercial_truck_bus` | `2.20` m | `70.0°` | `0.46` | `0.082` |
| `light_mobility_twowheeler` | `0.95` m | `55.0°` | `0.38` | `0.082` |

Shared `calibration_constants` in [camera_params.json](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/config/camera_params.json):
- `horizon_cutoff_ratio = 0.15`
- `depth_exponent = 1.35`
- `width_scale = 1.0`
- `length_exponent = 1.15`
- `max_depth_clip_cm = 35.0`

### 1.2 Exact Distance & Sizing Equations (`MetricDepthCalibrator`)
1. **Horizon Pixel Coordinate (Static & Truncated):**
   $$\text{horizon\_y} = \lfloor H \cdot \text{horizon\_ratio} \rfloor$$
   In `RoadVisionPipeline`, detections with `y2 <= HORIZON_Y` are filtered out as above-horizon false positives.
2. **Assumed Vertical Focal Length:**
   $$f_y = \frac{H / 2}{\tan(\text{VFOV}_{\text{deg}} / 2)}$$
3. **Horizon-Relative Pixel Offset & Clamp:**
   $$dy = \max\bigl(15.0,\; \text{box\_bottom\_y} - \text{horizon\_y}\bigr)$$
4. **Approximate Road-Surface Camera Distance:**
   $$Z_{\text{road}} = \frac{H_{\text{cam}} \cdot f_y}{dy}$$
5. **Heuristic Metric Depth, Width, & Length Mapping:**
   $$\Delta Z_{\text{cm}} = \text{clip}\bigl(\text{raw\_disparity\_delta} \cdot Z_{\text{road}}^{1.35} \cdot 0.082,\; 0.0,\; 35.0\bigr)$$
   $$W_{\text{cm}} = \text{clip}\!\left(\frac{pw_{\text{px}} \cdot Z_{\text{road}}}{f_y} \cdot 100.0,\; 5.0,\; 250.0\right), \quad L_{\text{cm}} = \text{clip}\!\left(\frac{ph_{\text{px}} \cdot Z_{\text{road}}^{1.15}}{f_y} \cdot 100.0,\; 5.0,\; 250.0\right)$$

---

## 2. Structural & Mathematical Audit Findings

| Audit Dimension | Observed Implementation Reality | Mathematical / Physical Implication |
|---|---|---|
| **Dynamic vs Static Horizon** | Fixed constant (`int(frame_height * horizon_ratio)` per preset). | Any pitch change (braking, speed bumps, mount tilt, handheld phone angle) shifts the true horizon without updating `horizon_y`, severely distorting `dy`. |
| **Pitch Consistency** | `horizon_ratio = 0.40` (vs image center `0.50`) at `60° VFOV` implies ~`9.9°` upward/downward optical axis offset relative to horizon, yet `Z_road = (H_cam * fy) / dy` omits any pitch angle \(\theta\) correction (\(\cos\theta\) / ray rotation). | Only strictly first-order valid for very small pitch on a locally flat plane with exact pinhole `VFOV` and zero lens distortion. |
| **Camera Roll & Road Slope** | Horizon is treated as strictly horizontal (`constant y` across full frame width) on zero-grade planar road. | Camera roll breaks horizontal equality; uphill/downhill grades shift true road horizon vertically relative to the horizon at infinity. |
| **Focal Length & Intrinsics** | Derived solely from guessed preset `vfov_deg`; **zero** true intrinsics (`fx, fy, cx, cy, k1..k3`). | Zoomed crops, smartphone cameras, dashcam wide-angle fisheye/barrel distortion, or different aspect ratios silently violate `fy`. |
| **Numerical Instability Near Horizon** | Clamped via hardcoded pixel floor `max(15.0, dy)`. | **Inverse singularity & resolution asymmetry:** near `dy -> 0`, \(1/dy \to \infty\); additionally, `15.0 px` equals `4.17%` of frame height at `360p` (`Z_max = 25.98 m` for sedan) vs `2.08%` at `720p` (`Z_max = 51.96 m` for sedan) — a **`2.55×` `depth_cm` discontinuity** across resolutions at identical normalized coordinates (see [phase7b_geometry_sensitivity.md](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/phase7b_geometry_sensitivity.md)). |
| **Depth Scaling Constants** | Fixed global constants (`k_disp = 0.082`, `depth_exponent = 1.35`). | Calibrated neither against depth-model output scale invariance nor against any laser/structured-light/manual physical depth dataset. |

---

## 3. Task 8 — Independent Verification Feasibility with Existing Project Data

An explicit check across every asset currently present in the workspace confirms **why camera intrinsics, extrinsics, and metric road geometry cannot be independently verified from existing data**:

| Required Quantity | Available Project Data Status | Can It Be Independently Verified Today? |
|---|---|---|
| **True Intrinsics (`fx, fy, cx, cy`, distortion)** | Mixed downloaded stock clips, web/YouTube footage (`ruralRoad`, `Rangoon Road`, `raw_survey` multi-clip montage), 70 unlabeled phone images (`dataset_phone_potholes`), no EXIF focal metadata or checkerboard sequences. | **No.** Stock/re-encoded footage lacks camera make/sensor metadata, and arbitrary multi-clip cuts (`raw_survey`) mix unknown lenses inside one file. |
| **Camera Height (\(H_{\text{cam}}\)), Pitch (\(\theta\)), & Roll (\(\phi\))** | Only four nominal user-selected profile guesses (`0.95–2.20 m`); no IMU, vehicle CAN/suspension telemetry, or mount calibration measurements. | **No.** Without known lane-width / chessboard / 3D control points and known `K`, a single-image 1-point vanishing point determines only an image pixel \((v_x, v_y)\), NOT unique \((H_{\text{cam}}, \theta, f_y)\). |
| **Road Plane Orientation / Longitudinal Slope** | Monocular RGB video without stereo, LiDAR, RTK-GPS/IMU, or lane-width markings of known physical gauge. | **No.** Visual inspection of `rawSurvey_160f_10fps` and `ruralRoad_202f` confirms both uncalibrated downward pitches and montage cuts, with no physical scale bars. |
| **Physical Pothole Metric Depth (cm)** | `Pothole440` remains excluded/blocked for depth validation; no physically measured cavity depth ground truth exists in the repo. | **No.** Relative disparity deltas (`raw_disp`) multiplied by unverified `Z^1.35 * 0.082` cannot be calibrated to physical centimeters yet. |

**Conclusion:** Replacing static `horizon_ratio` with a classical vanishing-point tracker only supplies an *image-space horizon proxy* `(vx, vy)` when road perspective cues are strong — it **cannot** supply verified physical extrinsics or metric calibration by itself.
