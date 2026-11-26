# Phase 5A — Existing RoadVision Depth Pipeline Audit

**Audited Code & Configs:**
- [roadvision_engine.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_engine.py) (`MetricDepthCalibrator`, `AnnularDonutDepthExtractor`, `SpecularAquaplaneDetector`, `TemporalEMASmoother`, `RoadVisionPipeline`)
- [config/camera_params.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/config/camera_params.json)
- Prior audit baselines: [depth_ground_truth_audit.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/depth_ground_truth_audit.md), [depth_cross_calibration_report.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/depth_cross_calibration_report.md), [pothole440_depth_encoding_report.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/pothole440_depth_encoding_report.md)

---

## 1. End-to-End Pipeline Implementation Inventory

| Component | Exact Implementation Observed in `roadvision_engine.py` | Key Lines |
|---|---|---|
| **MiDaS Variant** | `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")` + `transforms.small_transform` | `L521–L523` |
| **Inference & Gating** | Conditional gating (`has_hazards`: any raw box has `y2 > H * horizon_cutoff_ratio` (`0.15`)); CUDA `autocast`, bilinear interpolation to `(H, W)` | `L641–L670` |
| **Input Color Channel Bug** | Passes OpenCV `BGR` `frame` directly into `self.depth_transform(frame)` without `BGR -> RGB` conversion | `L597, L656` |
| **Relative Depth Polarity** | Inverse relative depth (`MiDaS_small`): higher values = closer (`1/Z + shift`), lower values = deeper/farther; depression = `expected_road - crater_patch > 0` | `L246–L247` |
| **Annular Region** | Expands `[x1, y1, x2, y2]` by `mx = max(15, min(25, int(0.25*bw)))`, `my = max(15, min(25, int(0.25*bh)))`; excludes inner bbox | `L219–L231` |
| **Reference Plane Fitting** | Unregularized OLS (`np.linalg.lstsq`) fitting `z(y, x) = a*y + b*x + c` over unnormalized local `[ys, xs]` whenever `len(ys) >= 20` | `L234–L240` |
| **Fallback Baseline** | If `< 20` donut pixels, scalar `np.median(surrounding_road) - np.percentile(crater_patch, 25)` | `L255–L260` |
| **Cavity Statistic** | `max(0.0, np.percentile(expected_road - crater_patch, 85))` over entire rectangular `crater_patch` | `L249–L251` |
| **Metric Calibration** | `Z_road = (H_cam * fy) / max(15, y2 - horizon_y)`; `depth_cm = raw_disp * (Z_road ** 1.35) * 0.082`, clipped `[0.0, 35.0] cm` (`horizon_ratio=0.40`, `VFOV=60°`, `H_cam=1.25 m` for `sedan_dashcam`) | `L128–L174` |
| **Water / Glare Check** | `SpecularAquaplaneDetector`: `V > 185 & S < 45` (`ratio > 0.06`) AND grayscale Laplacian variance `< 210.0` on raw BGR crop | `L276–L292` |
| **Temporal Smoother** | `TemporalEMASmoother` (`history_len=10`): `alpha * calibrated_cm + (1 - alpha) * median(history)` (`alpha=0.45` if `y2 >= best_y2` else `0.20`); overrides severity to `SEVERE [WATER]` if `water_votes / obs > 0.30` | `L378–L479` |

---

## 2. Mathematical & Implementation Weaknesses Identified

1. **BGR-to-RGB Color Channel Inversion Before MiDaS (`L656`):**
   - `cv2.VideoCapture.read()` returns BGR images, whereas `intel-isl/MiDaS` `small_transform` expects RGB ordering for ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Passing BGR silently permutes Red and Blue features.
2. **Unweighted OLS Reference Fitting Without Outlier Rejection (`L240`):**
   - All pixels in the annular ring contribute equally to `np.linalg.lstsq`. A roadside curb, tree shadow edge, bumper, adjacent pothole, or cropped pavement crack inside the ring corrupts the plane slope `(a, b)` and intercept `c` arbitrarily.
3. **Adjacent Potholes Not Masked Out of Annular Ring (`L227–L231`):**
   - `AnnularDonutDepthExtractor` masks only the target bounding box `[x1:x2, y1:y2]`. Neighboring potholes inside `[rx1:rx2, ry1:ry2]` depress the fitted reference surface downward, shrinking measured cavity residuals.
4. **Capped `25 px` Outer Margin & Unnormalized Pixel Coordinates (`L219–L238`):**
   - For large foreground potholes (e.g., `300×150 px`), `margin <= 25 px` samples only a narrow immediate rim often still inside the beveled crater apron, whereas tiny `25×15 px` detections receive `15 px` margins (`60%–100%` of box size) that bleed into off-road backgrounds. Raw unnormalized `ys, xs` also degrade matrix conditioning relative to normalized `[-1, 1]` coordinates.
5. **Silent Zero Clamping Hides Invalid Measurements (`L251, L262, L163`):**
   - Negative disparities (`expected_road < crater_patch`), plane fitting failures, tiny boxes (`< 5 px`), and empty regions silently return `0.0` instead of an explicit invalid flag. `TemporalEMASmoother` then pushes those fabricated `0.0 cm` readings into the rolling median (`L429`), pulling valid track estimates toward zero.
6. **Rectangular Box Dilution of Cavity Statistics (`L249–L250`):**
   - Taking the 85th percentile across the full bounding box assumes ~15%+ of rectangular box pixels occupy deep crater floor; irregular diagonal cavities inside oversized boxes dilute this quantile, while extreme corner noise spikes can dominate.
7. **Unverified & Physically Misaligned Metric Calibration Assumptions (`L167–L174`):**
   - MiDaS outputs uncalibrated relative inverse depth $d = s \cdot (1/Z) + t$ whose scale $s$ and shift $t$ vary **per frame** depending on image depth composition. Applying fixed static `k_disp = 0.082` and `depth_exponent = 1.35` across arbitrary frames assumes fixed scale $s$, which is mathematically violated and empirically unverified (Phase 3A showed `depth_ground_truth_manifest.csv` `P01–P15` cannot be reproduced from the video and overlaps synthetic test cases `T01–T02`).
8. **Water-Filled Pothole Depth Contamination (`L707–L731`):**
   - Although `SpecularAquaplaneDetector` flags reflective water puddles, the pipeline **still extracts MiDaS disparity on mirror-like sky reflections** (`is_water = True`) and feeds the corrupted depth and volume into `TemporalEMASmoother` and CSV outputs.
9. **Horizon Constant Ambiguity (`L129, L579–L580`):**
   - `camera_params.json` defines both `horizon_cutoff_ratio = 0.15` (used to gate detections) and `horizon_ratio = 0.40` (used inside `MetricDepthCalibrator`), creating situations where detections between `0.15H` and `0.40H` pass gating (`L649`) and hit `dy = max(15.0, y2 - horizon_y)` floor clamping (`Z_road` saturating at `~26 m`).
