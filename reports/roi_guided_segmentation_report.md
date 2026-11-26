# Phase 4A.4 — Detector-Guided ROI Segmentation & Performance Report

**Status:** Complete (Experimental Evaluation — Production Files Untouched)  
**Module:** [experiments/roi_guided_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/roi_guided_segmentation.py)  
**Benchmark Script:** [experiments/benchmark_roi_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_roi_segmentation.py)  
**Unit Tests:** [tests/test_roi_guided_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_roi_guided_segmentation.py) (`9/9` passed)  
**Comparison Data:** [roi_segmentation_comparison.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_segmentation_comparison.json), [roi_segmentation_per_frame.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_segmentation_per_frame.csv)  
**Visualizations:** [4-Panel Comparison PNGs (`36`)](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_seg_comparison_viz), [Rural Unseen Demo Video](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_demo_rural_road_unseen.mp4), [Rangoon 300-Frame Demo Video](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_demo_rangoon_300f.mp4)

---

## 1. Architecture & Coordinate Preservation Rules

The experimental cascade pairs the Phase 2 controlled dashcam detector with Phase 4A.2 `YOLOv8n-seg` inside detector-guided crops:
1. **Proposal Generation:** Full-frame inference via `Phase 2 YOLOv8n` ([best.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt), `imgsz=640`, `conf=0.25`, `iou=0.70`).
2. **ROI Extraction & Clipping:** Expand each box by `max(min_pad_px=24, crop_pad_ratio=0.40 * box_dim)` on each side and clip strictly to `[0, W) × [0, H)`.
3. **Aspect-Ratio-Preserving Letterboxing:** Scale each variable-aspect crop cleanly into `640×640` padding (`value=114`) without stretching pothole geometry, enabling both sequential (`batch_roi=False`) and true multi-crop batched (`batch_roi=True`, `max_batch_size=8`) GPU inference.
4. **Inverse Mapping & Association:** Unletterbox predicted instance masks back to `(crop_h, crop_w)` via bilinear interpolation (`>= 0.50`), embed into original `(H, W)` frame coordinates, associate with the detector box by mask-box IoU, and evaluate geometric quality rules.
5. **Detection-Preserving Fallback:** Any rejected or empty ROI mask reverts to the filled detector bounding box (`bbox_fallback`), preserving `100%` of detector proposals.

Unit tests in [test_roi_guided_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_roi_guided_segmentation.py) verified boundary clipping, wide-aspect (`4:1`) letterbox/unletterbox roundtrip mask IoU (`> 0.97`), box coordinate inverse projection (`< 0.5 px` error), quality rejection rules, and mocked detection preservation.

---

## 2. Development Dataset Provenance (`n = 36` Diverse Frames)

Zero thresholds or padding parameters were tuned on the completed Pothole440 grouped test split. All frames in `roi_segmentation_comparison.json` were verified by pixel MD5 against `dataset_unified/images/train`:

| Session / Source | Frames | Detector Exposure (`Phase 2 YOLOv8n`) | Seg Exposure (`Phase 4A.2`) | Covered Conditions |
|---|---:|---|---|---|
| `uploads/upload_1794847200_ruralRoad_potHoles.mp4` | `12` | **Unseen** (`0` train matches) | **Unseen** | Small/distant potholes, dense multi-pothole damage, sunlit water puddles, shoulder grass |
| `Watch Video_ Terrible potholes in Rangoon Road.mp4` | `12` | **Unseen temporal frames** (`0` overlap with `dataset_v2_candidates` 35 frames or train) | **Unseen** | Water/mud cavities, shadowed craters, road cracks, pale patches, curbs |
| `roadvision-dashboard/public/videos/raw_survey.mp4` | `6` | **Unseen raw frames** (`0` train MD5 matches) | **Unseen** | Low-angle flooded craters, shadowed oblique puddles, urban traffic |
| `dataset_phone_potholes/images/val` | `6` | **Exposed provenance** (`dataset_unified` phone pool) | **Unseen** | Oblique flooded multi-pothole street scenes (`2–8` GT boxes/frame) |

---

## 3. Three-Way Inference Strategy Comparison (`n = 36` Frames)

| Metric | Strategy A: Det Only | Strategy B: Full-Frame `YOLOv8n-seg` | Strategy C: ROI-Guided Seg + QA Fallback |
|---|---:|---:|---:|
| **Total Retained Detections** | `98` | `22` across `16/36` frames | **`98`** (**`100%`** detection preservation) |
| **Segmentation Attempts** | `0` | `36` full frames | `98` ROI crops |
| **Accepted Pixel Masks** | — | `22` (unfiltered) | **`11`** (`11.22%` of proposals) |
| **Rejected / Empty / Fallback** | — | — | `87` rejected (`88.78%`), incl. `68` empty (`69.39%`) |
| **Excessive Frame Coverage (`> 35%` area)** | `0` frames | **`5 / 36` frames** (up to `81.6%`!) | **`0 / 36` frames** |
| **Mean Active Image Coverage** | Bounding boxes only | `30.432%` | **`5.756%`** (accepted masks) |

### Verified Bounding-Box Detection Benchmarks (Box-Only, `IoU = 0.50`)
*Note: Empty `.txt` files in `evaluation/ground_truth_labels/` (`40` `PENDING_MANUAL_LABEL`) were excluded; no segmentation IoU/Dice is computed without pixel masks.*
- **`rangoon_35` (`dataset_v2_candidates`, `76` boxes):** Strategy A & Strategy C identically achieve `AP@50 = 0.2265` (`conf=0.25` P/R/F1 `0.5128 / 0.2632 / 0.3478`), whereas Strategy B drops to `AP@50 = 0.0852` (`0.2727 / 0.0395 / 0.0690`).
- **`phone_val_14` (`dataset_phone_potholes/val`, `52` boxes):** Strategy A & Strategy C achieve `AP@50 = 0.9609` (`0.7576 / 0.9615 / 0.8475`), whereas Strategy B drops to `AP@50 = 0.0888` (`1.0000 / 0.0385 / 0.0741`).

---

## 4. Visual Plausibility vs. True Accuracy & Qualitative Error Analysis

Because dashcam frames lack pixel segmentation ground truth, **mask acceptance rate (`11.22%`) measures geometric heuristic consistency rather than true pixel IoU**. Qualitative visual review of [roi_seg_comparison_viz/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_seg_comparison_viz) revealed four critical behaviors:
1. **Whole-Road Collapse Eliminated ([dev_rural_f0060.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_seg_comparison_viz/dev_rural_f0060.png), [dev_rangoon_f0230.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_seg_comparison_viz/dev_rangoon_f0230.png)):** Strategy B floods `81.6%` of the rural scene and `49.5%` of Rangoon `f=230`; Strategy C rejects overflowing ROI masks (`outside_detector_box_spill`) and restores clean detector boxes.
2. **Non-Road False Positives Blocked ([dev_survey_f0020.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_seg_comparison_viz/dev_survey_f0020.png)):** Strategy B misclassifies a motor-scooter in traffic (`cov=0.8%`), whereas Strategy C restricts segmentation to detector proposals, yielding clean masks on `3` foreground potholes (`acc=3, fb=1`).
3. **Detector-Limited Recall Ceiling ([dev_rangoon_f0020.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_seg_comparison_viz/dev_rangoon_f0020.png), [dev_rangoon_f0740.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_seg_comparison_viz/dev_rangoon_f0740.png)):** Any pothole missed by `YOLOv8n` detector receives zero ROI attempts.
4. **Spurious Detector Box Vulnerability ([dev_rural_f0075.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_seg_comparison_viz/dev_rural_f0075.png)):** When the detector itself emits a misaligned box covering shoulder grass, a `YOLOv8n-seg` grass mask sitting inside that box passes geometric box-overlap checks (`acc=1`), proving visual acceptance does not guarantee semantic correctness.

---

## 5. RTX 2050 Performance & Real-Time Suitability Benchmark

Measured over `108` timed passes (`36` frames × `3` loops after CUDA warmup) on `NVIDIA GeForce RTX 2050` (`4 GB` VRAM):

### 5.1 Stage-by-Stage Latency & Sequential vs. Batched Comparison

| Pipeline Stage | Sequential ROI (`batch_roi=False`) Mean / Med / P95 (ms) | Batched ROI (`batch_roi=True`) Mean / Med / P95 (ms) |
|---|---:|---:|
| **Detector (`YOLOv8n`)** | `16.73` / `11.70` / `34.29` | `13.17` / `11.35` / `19.60` |
| **ROI Extraction & Letterbox** | `1.81` / `1.46` / `4.91` | `1.89` / `1.53` / `4.81` |
| **Segmentation Inference (All ROIs)** | `56.19` / `42.45` / `159.24` (`23.79 ms/crop`) | `34.65` / `27.49` / `82.87` |
| **Mask Unletterbox & Reconstruction** | `1.10` / `0.48` / `3.93` | `1.03` / `0.44` / `3.66` |
| **Quality Assessment & Fallback** | `0.86` / `0.38` / `3.91` | `0.86` / `0.37` / `4.07` |
| **Combined Cascade Total** | **`76.68` / `63.25` / `199.83` (`13.04 FPS`)** | **`51.61` / `45.50` / `111.17` (`19.38 FPS`)** |
| **Peak GPU VRAM (Alloc / Reserved)** | `95.13 MB` / `124.0 MB` | `330.90 MB` / `534.0 MB` |

*(For comparison, standalone Strategy B Full-Frame `YOLOv8n-seg` runs at `17.44 ms` / `57.33 FPS`, `77.35 MB` allocated VRAM, while Strategy A `YOLOv8n` Detector runs at `~13.17–16.73 ms`.)*

### 5.2 Latency Scaling vs. Number of Potholes per Frame (`batch_roi=True` vs `False`)

| Proposals / Frame | Samples | Mean Proposals | Sequential Mean / P95 (FPS) | Batched Mean / P95 (FPS) |
|---|---:|---:|---:|---:|
| `0` potholes | `12` | `0.0` | `16.85 ms` / `46.17 ms` (`59.34 FPS`) | `12.11 ms` / `17.83 ms` (`82.57 FPS`) |
| `1` pothole | `24` | `1.0` | `50.89 ms` / `106.35 ms` (`19.65 FPS`) | `29.37 ms` / `36.87 ms` (`34.05 FPS`) |
| `2` potholes | `21` | `2.0` | `66.85 ms` / `186.56 ms` (`14.96 FPS`) | `44.05 ms` / `49.88 ms` (`22.70 FPS`) |
| `3` potholes | `21` | `3.0` | `102.07 ms` / `302.94 ms` (`9.80 FPS`) | `57.45 ms` / `68.63 ms` (`17.41 FPS`) |
| `4+` potholes | `30` | `5.5` | `110.34 ms` / `186.94 ms` (`9.06 FPS`) | `86.40 ms` / `118.18 ms` (`11.57 FPS`) |

**Real-Time Suitability Verdict:** Even with CUDA batching (which improves throughput `1.49×` while staying safely under `534 MB` of `4 GB` VRAM), the full cascade averages `19.38 FPS` across mixed scenes and drops to **`11.57 FPS` (`86.40 ms` mean, `118.18 ms` P95)** on damaged multi-pothole roads — and **`14.04 FPS` (`71.23 ms`)** across the `202`-frame unseen rural demo video (`ruralRoad_potHoles.mp4`). Because the full RoadVision pipeline also runs ByteTrack tracking, MiDaS depth inference, and planar road calibration, **this experimental cascade is not yet suitable for 25–30 FPS real-time deployment** on the RTX 2050.

---

## 6. Research-Only Video Demonstrations (`Task 8`)

Generated via standalone [demo_roi_segmentation_video.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/demo_roi_segmentation_video.py) (`--batch-roi`, `det_conf=0.25`, `seg_conf=0.25`):
1. **Unseen Rural Dashcam ([roi_demo_rural_road_unseen.mp4](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_demo_rural_road_unseen.mp4)):** `202` frames (`8.08 s`, `640×360 @ 25 FPS`), `914` raw per-frame proposals → `34` accepted ROI masks (`3.72%`) and `880` bbox fallbacks (`96.28%`), `71.23 ms` mean (`14.04 FPS`).
2. **Rangoon Road Clip ([roi_demo_rangoon_300f.mp4](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_demo_rangoon_300f.mp4)):** `300` frames (`12.0 s`, `640×360 @ 25 FPS`), `401` raw per-frame proposals → `94` accepted ROI masks (`23.44%`) and `307` bbox fallbacks (`76.56%`), `34.75 ms` mean (`28.77 FPS`).
*(Raw per-frame counts do not represent unique physical potholes because tracking is disabled.)*

---

## 7. Final Phase 4A.4 Recommendation

**Recommendation B — ROI-guided segmentation requires domain adaptation.**

Detector-guided ROI gating + geometric quality fallback successfully eliminated perspective whole-road mask collapse and traffic false positives while keeping `100%` of detector recalls, proving the *gating architecture* works safely as a guard rail. However, `88.78%` fallback (`69.39%` empty ROI masks) on the 36-frame dashcam suite and `96.28%` fallback on unseen rural road footage show that cropping alone cannot overcome the top-down `Pothole440` domain gap. RoadVision should **retain detection-only operation in production** until a domain-adapted dashcam segmentation model is trained and validated using the annotation plan in [dashcam_segmentation_annotation_plan.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/dashcam_segmentation_annotation_plan.md).
