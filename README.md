# RoadVision AI 🚗
### Real-Time 3D Pothole Perception, Severity Estimation & Municipal Road Audit System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![MiDaS](https://img.shields.io/badge/Intel-MiDaS_3D-0071C5?logo=intel&logoColor=white)](https://github.com/isl-org/MiDaS)
[![React](https://img.shields.io/badge/Frontend-React_19_+_Vite-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Flask](https://img.shields.io/badge/Backend-Flask_REST_API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

---

## 📌 Why This Project?

Most pothole detection projects just slap a 2D bounding box detector (like standard YOLO) on single dashcam photos. In the real world, that doesn't actually help a city road crew:

1. **A 2D box has no depth**: A harmless 2 cm surface scrape and an axle-snapping 14 cm crater look identical in a flat bounding box. Cities need to know which ones to fix *first*.
2. **Duplicate counting**: When a car drives past a pothole for 3 seconds, the camera sees it across 75+ video frames. Without tracking, the system logs 75 separate potholes instead of 1.
3. **Puddles blind depth models**: Rain-filled potholes act like mirrors reflecting the sky, which tricks depth networks into seeing flat road or infinite distance.
4. **Road perspective tilt**: Dashcam cameras look down at the road, so road disparity naturally rises from horizon to hood. Simple baseline subtractions falsely report massive depths.

**RoadVision AI** is built to solve these real-world edge cases. It takes standard dashcam or smartphone video ($H_{\text{cam}} \approx 1.25\text{ m}$), detects potholes, tracks each unique hazard with ByteTrack, estimates depth in metric centimeters using monocular 3D + pinhole optics, flags water hazards, and outputs an automated municipal repair schedule (CSV).

---

## 🏗️ How the Pipeline Works

```
Video Stream (Dashcam / Phone)
    │
    ▼
[Stage 1] YOLOv8s Spatial Localization (2D Bounding Boxes)
    │
    ▼
[Stage 2] ByteTrack Multi-Object Tracking (Kalman Filter, Unique IDs)
    │
    ├─ No Hazard? ──► [Bypass Depth] ──► Runs at 70-80+ FPS
    │
    ▼ Hazard Found?
[Stage 3] Intel MiDaS Small (Relative Inverse Disparity Map)
    │
    ▼
[Stage 4] Annular Donut Mask + 2D Planar Detrending (Isolate Clean Road Baseline)
    │
    ▼
[Stage 5] Pinhole Ground-Plane Calibration (Disparity -> Metric cm)
    │
    ▼
[Stage 6] Specular Aquaplane Detector (HSV Glare + Laplacian Texture Check)
    │
    ▼
[Stage 7] Temporal EMA Smoothing (Proximity-Weighted Alpha Filter)
    │
    ▼
Annotated HUD Video + Municipal Contractor Audit (CSV)
```

### The 7 Core Steps:
1. **YOLOv8s Detector**: Fine-tuned on combined Indian road damage datasets (RDD2022 India Class D40 + asphalt craters). Outputs spatial coordinates $[x_1, y_1, x_2, y_2, \text{conf}]$.
2. **ByteTrack Tracker**: Uses a Kalman Filter and two-stage Hungarian association. Configured with `fuse_score: false` so that temporary low-confidence drops over bumps don't break tracking trajectories. Prevents duplicate recounts.
3. **Intel MiDaS 3D Monocular Depth**: Predicts dense inverse relative disparity. We use **conditional execution gating**: MiDaS is bypassed on clean road frames (~70% of driving), keeping the pipeline fast (**45–50+ FPS**).
4. **Annular Donut Mask & Planar Detrending**: Slices a 25px ring around the crater to sample clean road, fits a 2D local plane ($z = ay + bx + c$) via least-squares, and subtracts the crater floor at the 85th percentile ($\text{P}_{85}$). This cancels out the perspective tilt of the road.
5. **Pinhole Optical Metric Calibration**: Projects camera mounting height ($1.25\text{ m}$), vertical FOV ($60^\circ$), and horizon row ($0.40H$) to calculate longitudinal distance $Z(y)$ along the road. Scales disparity into physical depth:
   $$\Delta Z_{\text{cm}} = \Delta d_{\text{raw}} \cdot (Z_{\text{road}}(y))^{1.35} \cdot 0.082$$
6. **Specular Aquaplane Detector**: Checks for sky reflection glare in HSV space ($V > 185, S < 45$) and low Laplacian texture variance ($\sigma^2 < 210$). Flags puddles as `SEVERE [WATER]` to warn of hydroplaning risk.
7. **Temporal EMA Smoother**: Smooths readings across video frames using an Exponential Moving Average. Weights observations closer to the vehicle ($\alpha = 0.40$) more heavily than distant ones ($\alpha = 0.15$) to damp out camera vibration.

---

## 🚦 Municipal Severity Classification

RoadVision categorizes every detected defect and assigns automated work orders:

| Severity Tier | Depth Range | Automated Work Order | Risk Level |
| :--- | :---: | :--- | :--- |
| 🔴 **Severe** | $\ge 8.0\text{ cm}$ | `"Urgent Work Order Dispatched"` | Rim damage, tire puncture, axle fracture |
| 🟡 **Moderate** | $4.0 - 7.9\text{ cm}$ | `"Scheduled Inspection"` | Progressive pavement decay, suspension wear |
| 🟢 **Mild** | $< 4.0\text{ cm}$ | `"Logged / Routine Maintenance"` | Cosmetic wear / early surface cracks |
| 🌊 **Water Hazard** | Puddle Flag | `"Aquaplaning Risk / Drain Clearing"` | High-speed hydroplaning, hidden crater floor |

---

## 📊 Model Training & Benchmarks

### Training Setup (YOLOv8s)
- **Dataset**: IEEE BigData RDD2022 India (Class D40) + Curated asphalt craters (~4,113 total images, 85% train / 15% val).
- **Training Strategy**: Transfer learning from COCO-pretrained weights.
- **Optimizer**: AdamW ($lr_0 = 0.002$, cosine decay to $0.01$).
- **Epochs**: **25 epochs** with early stopping (`patience=10`).
- **Resolution**: Multi-scale up to $1280 \times 1280$ (retains small, distant surface cracks).
- **Augmentations**: Mosaic (1.0), MixUp (0.15), HSV jitter (H: 0.015, S: 0.7, V: 0.4), Random Perspective (0.0005). Mosaic disabled for final 10 epochs (`close_mosaic: 10`).

### Validation Results (on held-out val split)
- **mAP@50**: **67.2% (0.6717)** *(Competitive with international RDD challenge baselines)*
- **Precision**: **72.9%**
- **Recall**: **60.6%**
- **mAP@50-95**: **33.9%**

### Video Tracking Continuity (Unseen 1,005-frame Rangoon Road test drive)
- **Detection Frame Coverage**: **85.1%** (855 frames)
- **Tracking Continuity**: **81.0% persistent tracks** ($\ge 5$ frames)
- **Transient Blips**: Cut from 42 blips down to **1 blip** (97.6% drop) by tuning ByteTrack to `fuse_score: false`.
- **Flicker Rate**: Reduced by **35.5%**.
- **Duplicate Recounts**: **0** (assigned clean persistent track IDs `#P-001` to `#P-006`).

---

## 💻 Tech Stack

- **Computer Vision & DL**: PyTorch 2.0+, Ultralytics YOLOv8s, Intel MiDaS Small, ByteTrack, OpenCV, NumPy.
- **Backend API**: Flask, Flask-CORS (Python 3.10+).
- **Frontend Dashboard**: React 19, Vite 8, Framer Motion, Lucide Icons, Vanilla CSS (Warm Editorial UI).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ with pip
- Node.js 18+ and npm
- (Optional) NVIDIA GPU with CUDA for faster inference. CPU inference is fully supported.

### 2. Clone the Repository
```bash
git clone https://github.com/Anmol-Jain371/Deep_Learning_project---POTHOLEdetection.git
cd Deep_Learning_project---POTHOLEdetection
```

### 3. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the Flask API server
python server.py
```
Backend runs at `http://localhost:5000`.  
Verify health check: `http://localhost:5000/api/health`.

### 4. Frontend Setup
In a new terminal:
```bash
cd roadvision-dashboard
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 📁 Repository Structure

```
├── roadvision_engine.py          # Core 7-stage perception pipeline
├── server.py                     # Flask REST backend API (/api/health, /api/detect)
├── road_tracker.yaml             # Production ByteTrack tracker configuration
├── requirements.txt              # Python package dependencies
├── runs/
│   └── detect/
│       └── pothole_combined_v2/  # Production Model #2 weights (best.pt) & training logs
├── roadvision-dashboard/         # Full-stack React + Vite web dashboard
│   ├── src/
│   │   ├── components/           # UI components (Hero, PipelineSection, VideoUpload, KPICards, AuditTable)
│   │   ├── App.jsx               # Main application shell & view routing
│   │   └── App.css               # Styling
│   └── package.json
├── depth_ground_truth_manifest.csv # Ground-truth straightedge ruler measurement protocol
├── validate_physical_depth.py    # Statistical evaluation tool for physical depth
├── test_dashboard_integration.py # Automated end-to-end integration test suite
└── README.md
```

---

## 🛡️ Research & Geospatial Integrity

- **GPS Coordinates**: The pipeline does **not** simulate or fabricate GPS coordinates. When videos lack embedded NMEA or SRT subtitle tracks, coordinates strictly output as `None`. The dashboard clearly notifies users that GPS requires metadata input.
- **Physical Depth Ground Truth**: Monocular depth from a single camera is an uncalibrated optical projection. We do not claim certified physical accuracy without ground truth. The repository includes [depth_ground_truth_manifest.csv](depth_ground_truth_manifest.csv) and [validate_physical_depth.py](validate_physical_depth.py) for direct ruler-measurement calibration.

---

## 👨‍💻 Author

**Anmol Jain**  
GitHub: [@Anmol-Jain371](https://github.com/Anmol-Jain371)
