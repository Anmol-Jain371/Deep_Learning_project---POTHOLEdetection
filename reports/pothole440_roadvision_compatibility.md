# Phase 3B: RoadVision AI Compatibility Assessment for Pothole440

> **Evaluated Against:** [`roadvision_engine.py`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_engine.py) (`MetricDepthCalibrator`, `AnnularDonutDepthExtractor`, `TemporalEMASmoother`), [`config/camera_params.json`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/config/camera_params.json), and [`dataset_inventory.json`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_inspection/dataset_inventory.json)

---

## 1. Compatibility Matrix Across RoadVision AI Objectives

| # | Objective | Classification | Detailed Scientific Justification |
|---|---|---|---|
| 1 | **Instance Segmentation** | **Partially Supported** *(after grouped re-splitting)* | `434/440` RGB-mask pairs have clean `1200×800` aligned binary masks (`cv2.findContours` extracts valid YOLO polygons). However, `34/40` test samples share base pothole IDs with `train` and `6` masks are empty (`aug` edge-cases). Training/evaluating instance segmentation is viable **only after re-splitting by the `98` base pothole IDs** (ignoring the leaked official split). |
| 2 | **Relative Depth Estimation** | **Partially Supported** *(masked interior only)* | Within the valid Go!SCAN patch (`~20%` median frame area), `395/395` non-empty `train+val` maps show monotonic cavity depth relief (`inside > outside`). However, **`99.5%` of frames contain constant padding (`1–147`, median `79.8%` area)** with sharp artificial step cliffs, so full-frame relative depth metrics fail unless padded pixels are masked out first and splits are re-grouped by base ID. |
| 3 | **Absolute Metric Cavity Depth Validation** | **Requires Additional Metadata** | Depth maps are 8-bit `uint8` PNGs (`0` PNG metadata, no `mm/level` factor, variable baseline offsets `1–147`). Furthermore, Pothole440 images are **top-down/close-up static frames**, incompatible with `MetricDepthCalibrator`'s forward-dashcam pinhole ground-plane formula ($Z(y) = H_{\text{cam}} f_y / (y_2 - y_{\text{horizon}})$) since there is no road horizon (`y_horizon`) or `H_cam`. |
| 4 | **Annular Road-Plane Fitting (`AnnularDonutDepthExtractor`)** | **Partially Supported / Incompatible Unmodified** | While real pavement inside the Go!SCAN patch exhibits slight planar slope amenable to `z = ay + bx + c`, expanding an annular margin around the mask overlaps the **constant-fill patch discontinuity** (`sample_06`, `sample_12`; `186` masks already touch constant mode fill). Running `AnnularDonutDepthExtractor` unmodified fits a plane across an artificial cliff rather than pavement. |
| 5 | **Adaptive Depth Calibration (`k_disp` / `depth_exponent`)** | **Unsupported** | RoadVision's `MetricDepthCalibrator` calibrates `Delta_Z = Delta_d_raw * Z_road^1.35 * k_disp` across dashcam perspective ranges ($Z_{\text{road}} \in [2, 15]\text{ m}$). Pothole440 lacks dashcam perspective distances, camera intrinsics/heights, and metric ground-truth depths—making it unusable for tuning `k_disp = 0.082` or `depth_exponent = 1.35`. |
| 6 | **MiDaS vs. Depth Anything V2 Comparison** | **Partially Supported** *(masked ordinal/affine ranking)* | Can benchmark relative cavity relief ordering (Spearman $\rho$, scale-and-shift invariant Pearson $r$, or affine-aligned error **strictly inside valid non-padded Go!SCAN regions**) on grouped base captures, remembering domain differences (top-down macro view vs. dashcam corridor view) and sign inversion (MiDaS inverse depth decreases in holes; Pothole440 depth increases). |
| 7 | **Surface Area Estimation ($cm^2$)** | **Requires Additional Metadata** | Pixel area (`fg_coverage_pct`) is known accurately, but converting pixels² to physical $cm^2$ requires camera-to-ground distance $Z$, focal length / GSD ($\text{mm/px}$), and camera tilt—none of which exist in Pothole440. |
| 8 | **Volume Estimation ($cm^3$ / Litres)** | **Requires Additional Metadata** | Volumetric integration requires both metric pixel scale ($\text{mm/px}$ in X/Y) and metric cavity depth ($\text{mm/level}$ in Z), plus physical volume ground truth—all absent from Pothole440. |
| 9 | **3D Reconstruction** | **Unsupported / Requires Additional Metadata** | Only uncalibrated 2D-projected `uint8` cropped patches are provided without intrinsics, point clouds (`.ply`), or metric Z-scaling; uncalibrated visual heightfields (`plot_surface`) can be plotted only in arbitrary pixel units, not true 3D geometry. |

---

## 2. Specific Structural Mismatches with `roadvision_engine.py`

1. **Temporal Modules (`TemporalEMASmoother` & `ByteTrack`):** Pothole440 consists of static augmented images rather than continuous video streams, so `TemporalEMASmoother` and `road_tracker.yaml` cannot be evaluated on it.
2. **Horizon Filtering (`horizon_cutoff_ratio = 0.15`, `horizon_ratio = 0.40`):** In `MetricDepthCalibrator` and `process_video()`, top image rows represent sky/horizon, whereas Pothole440 top rows are close-up asphalt texture (`Z` nearly uniform across top and bottom).
3. **Domain Distribution Shift:** Mixing `1200×800` top-down pavement crops directly into RoadVision's dashcam object detector (`dataset_unified`) risks biasing bounding-box aspect ratios and scales away from forward-looking vehicle corridors.
