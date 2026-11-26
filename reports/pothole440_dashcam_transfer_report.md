# Phase 4A.3 — Dashcam Domain-Transfer & Video Demonstration Report

**Study Type:** Exploratory Domain-Transfer Analysis (Close-Range Top-Down Pothole440 vs Oblique Dashcam & Phone Road Streams)  
**Models Compared:**
- **Phase 2 Controlled Detector:** `YOLOv8n` ([best.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt), `SHA256: ddd6a5be27ac21f3ef17ffe17fbac5ea901cf2a7f7706956c5e551a669490343`, trained on `dataset_unified`)
- **Phase 4A.2 Segmentation Model:** `YOLOv8n-seg` ([best.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1/weights/best.pt), `SHA256: 73f525322681468322dd129589279b016eeeab1197879a8876e3071e8e0d6d6d`, trained on Pothole440 grouped splits only)

**Artifacts & Scripts:**
- Domain-transfer evaluation script: [evaluate_dashcam_transfer.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/evaluate_dashcam_transfer.py)
- Domain-transfer summary & 16 side-by-side PNGs: [dashcam_transfer_summary.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_dashcam_transfer/dashcam_transfer_summary.json) in [pothole440_dashcam_transfer/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_dashcam_transfer)
- Research-only video demo script: [demo_pothole_segmentation_video.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/demo_pothole_segmentation_video.py)
- Annotated demo video & telemetry: [demo_rangoon_300f_yolov8n_seg.mp4](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_dashcam_transfer/demo_rangoon_300f_yolov8n_seg.mp4), [demo_video_summary.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_dashcam_transfer/demo_video_summary.json)

---

## 1. Strict Scientific Rules & Annotation-Quality Caveats

1. **No Fabricated Segmentation Metrics:** None of the RoadVision dashcam or phone datasets have pixel segmentation masks; **zero segmentation IoU or Dice values are calculated** on dashcam frames.
2. **Empty Annotation Exclusion:** `evaluation/ground_truth_labels/` contains `40` empty `.txt` files (`PENDING_MANUAL_LABEL` in [evaluation_manifest.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/evaluation/evaluation_manifest.csv)) and is strictly excluded — never treated as verified negative ground truth.
3. **Exploratory Detection Benchmarks Only:** Box-level metrics (`imgsz=640`, `iou_nms=0.70`) are reported only on datasets with reviewed bounding-box labels, with explicit caveats:
   - **`rangoon_35` ([dataset_v2_candidates/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_v2_candidates), `35` frames / `76` boxes):** Previously inspected exploratory set of non-consecutive frames from a single video (`Watch Video_ Terrible potholes in Rangoon Road.mp4`); does not represent general dashcam performance.
   - **`phone_val_14` ([dataset_phone_potholes/images/val](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_phone_potholes/images/val), `14` frames / `52` boxes):** Derived from `dataset_unified` provenance (seen distribution for `YOLOv8n` Phase 2, zero-shot unseen domain for `YOLOv8n-seg`).

---

## 2. Exploratory Bounding-Box Detection Comparison (`IoU = 0.50`)

| Dataset | Model | Box AP@50 (`conf≥0.001`) | `conf=0.25` (P / R / F1) | `conf=0.50` (P / R / F1) | TP / FP / FN (`conf=0.25`) |
|---|---|---:|---:|---:|---:|
| **`rangoon_35`** (`76` GT boxes) | **Phase 2 `YOLOv8n` Det** | **`0.2265`** | `0.5128 / 0.2632 / 0.3478` | `0.5500 / 0.1447 / 0.2292` | `20 / 19 / 56` |
| **`rangoon_35`** (`76` GT boxes) | **Phase 4A.2 `YOLOv8n-seg`** | `0.0852` | `0.2727 / 0.0395 / 0.0690` | `0.2222 / 0.0263 / 0.0471` | `3 / 8 / 73` |
| **`phone_val_14`** (`52` GT boxes) | **Phase 2 `YOLOv8n` Det** | **`0.9609`** | `0.7576 / 0.9615 / 0.8475` | `0.9500 / 0.7308 / 0.8261` | `50 / 16 / 2` |
| **`phone_val_14`** (`52` GT boxes) | **Phase 4A.2 `YOLOv8n-seg`** | `0.0888` | `1.0000 / 0.0385 / 0.0741` | `1.0000 / 0.0385 / 0.0741` | `2 / 0 / 50` |

Key quantitative finding: Across oblique phone/dashcam scenes, `YOLOv8n-seg` trained exclusively on top-down `1200×800` Pothole440 cavities exhibits severe recall collapse (`3.95%` recall on `rangoon_35`, `3.85%` recall on `phone_val_14`).

---

## 3. Reproducible Frame-by-Frame Visual Comparison (`n=16`)

Comparable settings: `imgsz=640`, `conf=0.25`, `iou_nms=0.70`, `retina_masks=True` (counts surviving `conf=0.50` also recorded).

| Sample ID | Source | Frame / Timestamp | Res | Det (`0.25`/`0.50`, Max Conf) | Seg (`0.25`/`0.50`, Max Conf) | Primary Observed Phenomenon |
|---|---|---|---|---|---|---|
| `dt_01` | `Watch Video_ Terrible potholes in Rangoon Road.mp4` | `f=0` (`0.00s`) | `640×360` | `1 / 1` (`0.673`) | `1 / 1` (`0.914`) | Seg nails near-field muddy crater; Det fires FP on left concrete patch |
| `dt_02` | `Watch Video_ Terrible potholes in Rangoon Road.mp4` | `f=40` (`1.60s`) | `640×360` | `2 / 1` (`0.636`) | `0 / 0` | **Det wins / Seg fails:** shadowed/low-contrast water puddles missed by Seg |
| `dt_03` | `Watch Video_ Terrible potholes in Rangoon Road.mp4` | `f=211` (`8.44s`) | `640×360` | `1 / 1` (`0.761`) | `1 / 1` (`0.618`) | **Catastrophic mask boundary explosion:** Seg box/mask expands over bottom ~50% intact road |
| `dt_04` | `Watch Video_ Terrible potholes in Rangoon Road.mp4` | `f=255` (`10.20s`) | `640×360` | `3 / 2` (`0.631`) | `1 / 1` (`0.658`) | Multiple water-filled potholes: Seg captures front cavity only, misses distant puddle |
| `dt_05` | `Watch Video_ Terrible potholes in Rangoon Road.mp4` | `f=715` (`28.60s`) | `640×360` | `0 / 0` | `1 / 1` (`0.920`) | Near-field rough dry crater segmented cleanly by Seg (`0.92`), missed by Det |
| `dt_06` | `Watch Video_ Terrible potholes in Rangoon Road.mp4` | `f=803` (`32.12s`) | `640×360` | `2 / 1` (`0.834`) | `0 / 0` | **Det wins / Seg fails:** deep shadowed & water-filled oblique craters missed by Seg |
| `dt_07` | `dataset_phone_potholes/images/val/frame_0005.jpg` | `f=5` (still) | `1280×720` | `8 / 5` (`0.892`) | `0 / 0` | **Multiple potholes (8 GT):** Det localizes all 8; Seg detects 0 |
| `dt_08` | `dataset_phone_potholes/images/val/frame_0028.jpg` | `f=28` (still) | `1280×720` | `2 / 2` (`0.954`) | `1 / 1` (`0.689`) | Near bottom-left foreground puddle segmented; mid-distance puddle missed by Seg |
| `dt_09` | `dataset_phone_potholes/images/val/frame_0035.jpg` | `f=35` (still) | `1280×720` | `3 / 3` (`0.821`) | `1 / 1` (`0.537`) | Foreground oblique cavity detected; 2 mid-range cavities missed by Seg |
| `dt_10` | `roadvision-dashboard/public/videos/raw_survey.mp4` | `f=40` (`4.00s`) | `1280×720` | `0 / 0` | `1 / 1` (`0.855`) | Partial mask covering only right half of low-angle foreground crater |
| `dt_11` | `roadvision-dashboard/public/videos/raw_survey.mp4` | `f=80` (`8.00s`) | `1280×720` | `2 / 0` (`0.469`) | `1 / 1` (`0.920`) | Both respond to foreground road feature |
| `dt_12` | `roadvision-dashboard/public/videos/raw_survey.mp4` | `f=120` (`12.00s`) | `1280×720` | `3 / 2` (`0.803`) | `0 / 0` | **Multiple oblique water-filled potholes:** Det catches foreground/mid; Seg misses all |
| `dt_13` | `uploads/upload_1794847200_ruralRoad_potHoles.mp4` | `f=60` (`2.40s`) | `640×360` | `2 / 2` (`0.703`) | `2 / 1` (`0.770`) | **Whole-road collapse FP:** Seg paints ~80% of frame (dirt road + grass) as foreground |
| `dt_14` | `uploads/upload_1794847200_ruralRoad_potHoles.mp4` | `f=75` (`3.00s`) | `640×360` | `3 / 2` (`0.856`) | `2 / 2` (`0.777`) | **Whole-road collapse FP:** massive frame-spanning false-positive mask over textured road |
| `dt_15` | `uploads/upload_1794847200_ruralRoad_potHoles.mp4` | `f=105` (`4.20s`) | `640×360` | `5 / 2` (`0.862`) | `0 / 0` | **Small/distant failure:** Det detects 5 multi-scale potholes; Seg detects 0 |
| `dt_16` | `uploads/upload_1794847200_ruralRoad_potHoles.mp4` | `f=150` (`6.00s`) | `640×360` | `5 / 4` (`0.848`) | `1 / 0` (`0.254`) | **Whole-road collapse FP** at `conf=0.25`; Det detects 5 cavities cleanly |

### 3.1 Summary of Domain-Transfer Failure Modes
- **Small / Distant & Oblique Foreshortened Potholes:** `YOLOv8n-seg` misses nearly all mid-field and distant potholes in oblique camera views (`dt_04`, `dt_07`, `dt_12`, `dt_15`) because Pothole440 contains only close-range near-nadir captures occupying `2%–24%` of the frame.
- **Whole-Frame Foreground Collapse / Mask Boundary Explosion:** Because Pothole440 never exposes the segmentation model to sky, trees, roadside grass, vehicles, or long perspective horizons, encountering textured rural roads (`dt_13`, `dt_14`, `dt_16`) or dense foreground cracking (`dt_03`) triggers oversized bounding boxes whose masks flood intact road and shoulder background.
- **Shadowed & Water-Filled Potholes:** On near-field water cavities (`dt_01`, `dt_04`, `dt_08`) `YOLOv8n-seg` sometimes traces crisp puddle boundaries, yet misses shadowed or obliquely lit pools (`dt_02`, `dt_06`, `dt_12`) that `Phase 2 YOLOv8n` reliably detects.
- **Cases Where Detector Succeeds and Segmentation Fails:** Occurred pervasively across `rangoon_35`, `phone_val_14`, `raw_survey.mp4 f=120`, and `ruralRoad_potHoles.mp4 f=105`. Conversely, on extreme close-up near-nadir Rangoon craters (`dt_01`, `dt_05`), `YOLOv8n-seg` occasionally fired where `Phase 2 YOLOv8n` missed.

---

## 4. Research-Only Video Processing Demonstration (Task 5)

Executed [demo_pothole_segmentation_video.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/demo_pothole_segmentation_video.py) on `Watch Video_ Terrible potholes in Rangoon Road.mp4` (`--conf 0.50`, `--max-frames 300`):
- **Output Video:** [demo_rangoon_300f_yolov8n_seg.mp4](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_dashcam_transfer/demo_rangoon_300f_yolov8n_seg.mp4) (`640×360` preserved, `25.0 FPS` preserved, `300` frames / `12.0 s` processed)
- **Latency (`NVIDIA GeForce RTX 2050`):** `14.77 ms` mean per frame (`13.66 ms` median, `22.96 ms` p95, `67.71 FPS` standalone segmentation + overlay draw; excludes ByteTrack, MiDaS, and planar calibration).
- **Raw Frame-Level Detection Telemetry:** `175 / 300` frames contained `>= 1` raw detection (`199` summed frame-level detections, `0.663` mean/frame, max `4` in a frame). **Caveat:** Because no object tracker is attached in this research script, raw detection sums do **not** represent unique physical potholes.

---

## 5. Final Phase 4A.3 Conclusion & Recommendation

**Selected Recommendation: C — Retain the existing detector and use segmentation only as an experimental auxiliary capability.**

*Scientific Justification:*
- On in-domain close-range pavement imagery, segmentation provides a verified `+28.93%` relative IoU gain and `-87.45%` reduction in false-positive pavement area over rectangular bounding boxes ([Box-vs-Mask Report](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/pothole440_box_vs_mask_comparison.md)), validating its future potential for cavity area estimation.
- However, this exploratory evaluation shows that Pothole440-trained `YOLOv8n-seg` **does not generalize to forward-facing dashcam streams** (`0.0852` AP@50 on Rangoon, `0.0888` AP@50 on phone footage, and whole-frame foreground collapse on `ruralRoad_potHoles.mp4` and `Rangoon f=211`), whereas the Phase 2 dashcam detector achieves `0.9609` AP@50 on `phone_val_14` and avoids road-surface collapse.
- Therefore, RoadVision **must retain the existing dashcam detector** (`roadvision_engine.py` / Phase 2 pipeline) and keep segmentation strictly as an **experimental auxiliary capability** unless and until a multi-domain dashcam segmentation dataset is curated and validated (**Recommendation B** remains the required future engineering prerequisite before any production replacement).
