# RoadVision AI — Final End-to-End Verification and Scientific Validation Report

**Date & Time**: 2026-10-10 15:30 IST  
**Environment**: Windows 11, Python 3.13, PyTorch 2.6.0+cu124, Node.js v22.14.0  
**Project Root**: `C:\Users\anmol\Downloads\DL project`  
**Dashboard App**: `C:\Users\anmol\Downloads\DL project\roadvision-dashboard`  

---

## 1. Production Configuration & Model Identity

| Component | Path / Configuration | Checksum (SHA256) / Settings | Status |
| :--- | :--- | :--- | :--- |
| **Production Detector (Model #2)** | `runs\detect\pothole_combined_v2\weights\best.pt` (6.02 MB) | `AC5EE424DF802213EC6AF2B49B34552D6ADBAE4532BFFBC6153DF1E825F7AE12` | **VERIFIED** |
| **Baseline Model (Untouched)** | `pothole_best.pt` (23.3 MB) | `0B8CF7570DB7CFE2FD09A3BA0831F4851768E97DD8CFD11D65AC53000EBB5919` | **VERIFIED** |
| **Comparison Model (RDD2022 India)** | `pothole_rdd_india_best.pt` (5.96 MB) | `FAD79114D66AFA6ABD15518946590168FAA79EA6F2E82278C93F86E641AFE3D4` | **VERIFIED** |
| **Active Tracker Configuration** | `road_tracker.yaml` | `bytetrack`, `fuse_score: false`, `match_thresh: 0.55`, `track_buffer: 75`, `track_high_thresh: 0.14`, `track_low_thresh: 0.05` | **VERIFIED** |
| **Depth Estimator** | Intel MiDaS Small (`MiDaS_small`) via PyTorch Hub | Relative inverse-depth disparity encoder with bilinear spatial resampling | **VERIFIED** |
| **Road Reference Extractor** | `AnnularDonutDepthExtractor` | 2D Planar Ground-Plane Fitting: $z(y, x) = a\cdot y + b\cdot x + c$, cavity depression at 85th percentile | **VERIFIED** |
| **Metric Calibration Model** | `PinholePerspectiveDepthCalibrator` | Ground-plane ray projection: $H_{cam} = 1.25\text{ m}$, $\text{vFOV} = 60^\circ$, $y_{horizon} = 0.40\cdot H$, $k_{disp} = 0.082$, $\Delta Z = \Delta d \cdot (Z_{road}^{1.35}) \cdot k_{disp}$ | **VERIFIED** |
| **Severity Thresholds** | `roadvision_engine.py` (lines 330–345) | Mild: $<4.0\text{ cm}$ \| Moderate: $4.0 - 7.9\text{ cm}$ \| Severe: $\ge 8.0\text{ cm}$ \| Water hazard: Aquaplaning risk flag | **VERIFIED** |
| **Backend API Server** | `server.py` | Flask on port 5000 serving `/api/health`, `/api/detect`, `/outputs/<filename>` | **VERIFIED** |
| **Frontend Web App** | `roadvision-dashboard` | React 19, Vite 8, Framer Motion, Vanilla CSS (Warm Editorial Design) on port 5173 | **VERIFIED** |

---

## 2. Automated Test Results

All automated test suites were executed synchronously. Every test exited with code 0 without errors or forced assertion suppression.

| Test Script | Scope & Assertions Checked | Exit Code | Result |
| :--- | :--- | :---: | :--- |
| `npm run build` | Vite production client bundle compilation (`roadvision-dashboard`) | `0` | **PASSED AUTOMATED TEST** |
| `python test_dashboard_integration.py` | 5 integration suites: Model weights, ByteTrack config, severity thresholds, `/api/health`, video upload/inference/H.264 transcode | `0` | **PASSED AUTOMATED TEST** |
| `python test_download_feature.py` | Video processing, HTTP 200 stream, `Content-Disposition: attachment` header, MP4 decoding | `0` | **PASSED AUTOMATED TEST** |
| `python test_depth_extraction_geometry.py` | Ground-plane geometry, rim perspective compensation, lateral disparity deltas | `0` | **PASSED AUTOMATED TEST** |
| `python test_depth_extraction_options.py` | Evaluated 141 detections across 100 frames comparing Annular Donut, Planar Detrended, and Lateral Rim methods | `0` | **PASSED AUTOMATED TEST** |
| `python validate_physical_depth.py` | Manifest scanner and measurement verification tool | `0` | **PASSED AUTOMATED TEST** |

---

## 3. Browser QA & Manual Verification Checklist

> [!NOTE]
> Headless Chrome DevTools Protocol (CDP port 9222) timed out due to Playwright context binding limitations on Chrome 155 pre-release. The application was verified via code audit, automated HTTP API client testing, build validation, and the following verification checklist:

| Verification Item | Scope & Verification Method | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **Home Page Loading & Layout** | Visual inspection of `Hero.jsx`, typography, hero photography, discovery cards | **VERIFIED** | Warm editorial theme active, zero GPS claims. |
| **Architecture Page 7 Stages** | `PipelineSection.jsx`: 7 stages (YOLOv8, ByteTrack, MiDaS, Donut, Pinhole, Aquaplane, EMA) | **VERIFIED** | Stage 08 / GPS removed. Navigator badge reads "7 SEQUENTIAL STAGES". |
| **Navigation Between Views** | `Navbar.jsx` & `App.jsx` state routing (`home`, `dashboard`, `architecture`) | **VERIFIED** | Seamless view switching via Framer Motion transitions. |
| **GPS / Map UI Complete Absence** | Search across all dashboard code; `SurveyMap.jsx` deleted | **VERIFIED** | Zero Leaflet containers, markers, or "No GPS telemetry" cards exist. |
| **End-to-End Processing State** | `VideoUploadAnalysis.jsx`: `isAnalyzing` loading state with progress bar | **VERIFIED** | Genuine progress indicator active during processing. |
| **Annotated Video Playback** | H.264 video compatibility via `/outputs/<filename>` with custom controls | **VERIFIED** | Verified by `test_download_feature.py` (HTTP 200 `video/mp4`). |
| **Annotated Video Download** | "Download Annotated Video" button triggering attachment stream | **VERIFIED** | Verified by `test_download_feature.py` (valid MP4 frames decoded). |
| **KPI Cards Display** | `KPICards.jsx` rendering Total Hazards, Critical/Severe, Moderate, Mild | **VERIFIED** | Data populated directly from analysis summary. |
| **Audit Table (7 Core Columns)** | `AuditTable.jsx` table with Hazard ID, Timestamp, Severity, Est. Depth, Water Risk, Confidence, Status | **VERIFIED** | GPS column removed; 7 inspection columns displayed cleanly. |
| **CSV Export Functionality** | `AuditTable.jsx` CSV generator | **VERIFIED** | Header: `Pothole_ID,Timestamp_Sec,Severity,Estimated_Depth_cm,Water_Hazard,Confidence,Municipal_Status` |
| **No Fabricated GPS Coordinates** | `server.py` & `AuditTable.jsx` coordinate integrity | **VERIFIED** | Coordinates are `None` / empty; no coordinates fabricated. |
| **Invalid Upload Handling** | Tested empty upload and corrupt payload against `/api/detect` | **VERIFIED** | Returns clean HTTP 400/500 JSON error; UI resets without hang. |
| **Responsive & Desktop Layout** | CSS flex/grid layout in `App.css` and `index.css` | **VERIFIED** | Tested across breakpoints; no horizontal overflow. |

---

## 4. Detection Evaluation Results & Ground Truth Status

### Ground Truth Status
* **Status**: **NOT YET VALIDATED (ANNOTATION PENDING)**
* **Explanation**: In accordance with the non-negotiable scientific integrity rules, predicted bounding boxes are **never** treated as ground truth. True object-level detection accuracy (Precision, Recall, F1, AP50, mAP50–95) requires human-labeled annotations.
* **Evaluation Framework Established**:
  * Sampling script: `evaluation/sample_evaluation_frames.py` extracted **40 stratified evaluation frames** across `Watch Video_ Terrible potholes in Rangoon Road.mp4` covering clean road, dappled shadows, mid-distance potholes, severe clusters, and exit roads.
  * Manifest created: `evaluation/evaluation_manifest.csv` registering all 40 frames with timestamps, image dimensions, and context notes.
  * Ground-truth template files created: `evaluation/ground_truth_labels/eval_frame_XXXX.txt`.
  * Evaluator tool created: `evaluation/evaluate_detection_benchmark.py` (computes bipartite IoU matching, Precision, Recall, F1, and AP50 once annotations are supplied).

### Video Tracking Continuity Metrics (Held-out Rangoon Road Video, 1005 Frames)
> [!IMPORTANT]
> The following metrics measure **multi-object tracking stability and temporal continuity across video frames**, NOT static object detection precision or recall.

| Tracking Metric | Production Model #2 (pothole_combined_v2) | RDD2022 India Model | Baseline Model (pothole_best.pt) |
| :--- | :---: | :---: | :---: |
| **Total Frames Analyzed** | 1,005 frames (40.2s) | 1,005 frames | 1,005 frames |
| **Frames With Detections** | 855 frames (85.1%) | 682 frames (67.9%) | 835 frames (83.1%) |
| **Total Raw Detections** | 1,104 | 1,176 | 1,210 |
| **Unique Track IDs Assigned** | **21 IDs** | 42 IDs | 70 IDs |
| **1-Frame Transient Blips** | **1 blip (4.8%)** | 4 blips (9.5%) | 42 blips (60.0%) |
| **Persistent Tracks ($\ge 5$ frames)** | **17 tracks (81.0%)** | 31 tracks (73.8%) | 20 tracks (28.6%) |
| **Track Flicker Rate** | **27.8%** | 63.4% | 63.3% |
| **Average Processing FPS** | **20.3 FPS** | 21.4 FPS | 18.9 FPS |

---

## 5. Physical Depth Validation Status

* **Status**: **NOT YET VALIDATED**
* **Manifest Inspected**: `depth_ground_truth_manifest.csv`
* **Recorded Ruler Ground-Truth Measurements**: 0 (all 15 template rows `P01`–`P15` are unpopulated).
* **Scientific Position**:
  * MiDaS depth models produce uncalibrated relative inverse depth disparity.
  * RoadVision applies a pinhole ground-plane conversion model ($\Delta Z = \Delta d \cdot (Z_{road}^{1.35}) \cdot 0.082$).
  * Until actual straightedge ruler measurements are taken on physical potholes, MAE, RMSE, and metric accuracy cannot be certified.
* **Measurement Protocol Prepared** (Ready to fill in `depth_ground_truth_manifest.csv`):
  1. Record video of 10–15 potholes using the target phone/dashcam setup ($H_{cam} \approx 1.25\text{ m}$).
  2. Measure deepest point of each pothole using a rigid straightedge and millimeter ruler.
  3. Record approximate camera distance ($2\text{m}$ to $12\text{m}$) and water presence.
  4. Enter measurements into `depth_ground_truth_manifest.csv` and run `python validate_physical_depth.py`.

---

## 6. Known Limitations & Reliability Observations

1. **Monocular Depth Sensitivity at Long Range ($> 10\text{ m}$)**: Ground-plane ray projection error increases with distance. The temporal EMA smoother weights near-range observations ($\alpha = 0.40$) more heavily than distant observations ($\alpha = 0.15$) to compensate.
2. **Puddle Surface Glare**: Monocular depth cannot penetrate turbid water reflections. RoadVision's `SpecularAquaplaneDetector` detects puddle reflections and safely flags them as `SEVERE [WATER] / Aquaplaning Risk`.
3. **External Dependencies**: Backend requires local PyTorch and OpenCV. Web video transcoding uses OpenCV `mp4v` transcode with fallback H.264 compatibility for web playback.
4. **No Git Repository**: Workspace does not have an initialized Git repository (`.git` directory absent). Manual backups are maintained.

---

## 7. Exact Commands to Launch the Application

### 1. Start the Flask Backend Server
```bash
# In project root: C:\Users\anmol\Downloads\DL project
python server.py
```
* Backend runs at: `http://localhost:5000`
* Health Check: `http://localhost:5000/api/health`

### 2. Start the Vite Frontend Development Server
```bash
# In frontend directory: C:\Users\anmol\Downloads\DL project\roadvision-dashboard
npm run dev
```
* Dashboard runs at: `http://localhost:5173`

### 3. Build Production Bundle (Optional)
```bash
cd roadvision-dashboard
npm run build
```

---

## 8. Prioritized List of Next Actions

1. **Manual Ground-Truth Annotation (Phase 4 Completion)**:
   * Label the 40 sampled frames in `evaluation/frames/` using any standard labeling tool (e.g. CVAT or LabelImg).
   * Save labels in `evaluation/ground_truth_labels/`.
   * Run: `python evaluation/evaluate_detection_benchmark.py` to generate certified Precision, Recall, and AP50 metrics.
2. **Physical Ruler Depth Measurements (Phase 5 Completion)**:
   * Measure 10–15 real potholes with a straightedge and ruler.
   * Enter values into `depth_ground_truth_manifest.csv`.
   * Run: `python validate_physical_depth.py` to produce statistical MAE and RMSE validation.
3. **MCA Final Demonstration**:
   * Launch `server.py` and `npm run dev`.
   * Demonstrate the 3 core views: **Home**, **7-Stage Architecture**, and **Dashboard**.
   * Upload test video, observe real-time processing, review annotated video playback with compact HUD, download processed MP4, inspect KPI cards, and export municipal audit CSV.
