# Phase 4A.3 — Frozen Segmentation Evaluation Protocol

**Status:** Frozen Pre-Evaluation Specification (Authored Prior to Test Split Execution)  
**Target Split:** `dataset_pothole440_seg_grouped/data.yaml` → `split=test` (`15` unaugmented original images; reconstructed grouped benchmark)  
**Target Model:** `runs/segment/pothole440_yolov8n_seg_v1/weights/best.pt` (selected at epoch `52` strictly on `val`)  
**Rule:** Thresholds, interpolation settings, boundary tolerances, and metric definitions in this document are locked prior to reading or running evaluation on the grouped `test` split. No checkpoint retraining, hyperparameter tuning, or threshold adjustment may occur after examining `test` outputs.

---

## 1. Verified Artifact Checksums & Provenance

| Artifact | Path | Verified SHA256 | Role |
|---|---|---|---|
| **Trained Segmentation Checkpoint** | [best.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1/weights/best.pt) | `73f525322681468322dd129589279b016eeeab1197879a8876e3071e8e0d6d6d` | Validation-selected `YOLOv8n-seg` (`3,390,771` params, `6.7` MiB) |
| **Dataset Manifest** | [dataset_manifest.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_pothole440_seg_grouped/manifests/dataset_manifest.json) | `cff0e21e8341a96a7c95c114aa1ea1509874c1047dfae1a9b2ea85c1fe972a16` | Leakage-controlled dataset inventory (`seed=42`) |
| **Per-Image Split Manifest** | [split_manifest.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_pothole440_seg_grouped/manifests/split_manifest.csv) | `63b2a1ca49e49b1fd859c8f9b809a9bbba67ff51c629f0092ea0bd7bd8dae643` | Maps `INCLUDED_TEST` filenames to original Pothole440 PNG masks |
| **Dataset YAML** | [data.yaml](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_pothole440_seg_grouped/data.yaml) | `bd97956a2912ece0a30500ac040c86380aae6b8327d696ec170688a027cb8cfc` | `nc: 1`, `{0: 'pothole'}` |
| **Phase 2 YOLOv8n Detector (Task 4)** | [best.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt) | `ddd6a5be27ac21f3ef17ffe17fbac5ea901cf2a7f7706956c5e551a669490343` | Controlled `YOLOv8n` dashcam/unified detector (`3,138,003` params) |

---

## 2. Frozen Inference & Post-Processing Configuration

### 2.1 Primary Region-Level Inference Configuration (`conf=0.50`)
Locked directly from Phase 4A.2 validation findings (`val` sensitivity sweep showed `conf=0.50` increased macro IoU from `0.7646` to `0.7871` and reduced false-positive regions from `2` to `1` with virtually unchanged recall):
- **Input Image Size (`imgsz`):** `640`
- **Primary Confidence Threshold (`conf`):** `0.50`
- **Secondary Reference Threshold Logged:** `0.25` (logged strictly for cross-comparison with Phase 4A.2 `val` primary runs; **not** used to select or alter the primary test claim)
- **NMS IoU Threshold (`iou_nms`):** `0.70`
- **Mask Upsampling (`retina_masks`):** `True`
- **Spatial Resizing & Binarization:** When `masks.data` spatial dimensions `[H_m, W_m]` differ from original image shape `(H, W)` (`800 × 1200`), each instance soft mask is resized with bilinear interpolation (`cv2.resize(..., (W, H), interpolation=cv2.INTER_LINEAR)`) and thresholded at `>= 0.50`. Image-level foreground mask is the pixel-wise logical union (`np.any`) across all retained instances.
- **Empty Predictions:** Any image with `0` retained detections above `conf` produces an all-`False` `(H, W)` binary mask (`empty_prediction_images` tracked explicitly) and is included in all macro and micro summaries.
- **Ground-Truth Reference:** Original unmodified Pothole440 binary PNG masks (`Pothole440-main/Pothole440-main/<orig_split>/mask/<filename>`, `gt_raw > 0` at `1200 × 800`), avoiding polygon-rasterization rounding artifacts.

### 2.2 Separate Ultralytics Standard AP Evaluation Protocol
Average Precision (`mAP@50`, `mAP@50-95`) and PR-curve Precision/Recall/F1 are evaluated via Ultralytics `model.val(...)`:
- `data=dataset_pothole440_seg_grouped/data.yaml`, `split="test"`, `imgsz=640`, `batch=4`, `iou=0.70`
- Ultralytics PR-curve confidence sweep uses Ultralytics `val` standard (`conf=0.001` default sweep across precision-recall thresholds, `max_det=300`), distinct from the fixed operating point `conf=0.50` used for binary mask reconstruction.
- Evaluated against YOLO polygon ground-truth labels (`dataset_pothole440_seg_grouped/labels/test/`).

---

## 3. Predefined Quantitative Metrics & Uncertainty Protocol

### 3.1 Ultralytics Instance Detection & Segmentation Metrics
- Box: `Precision`, `Recall`, `F1`, `mAP@50`, `mAP@50-95`
- Mask: `Precision`, `Recall`, `F1`, `mAP@50`, `mAP@50-95`

### 3.2 Region-Level Foreground & Error Metrics (`1200 × 800` Original Resolution)
- **Macro (Per-Image Equal Weight, `n=15`):** Mean, Std (`ddof=1`), Median, Min, Max of `IoU` and `Dice`; Mean `Pixel Precision` and `Pixel Recall`.
- **Micro (Pixel-Pooled Across All `15` Test Images):** Pooled `IoU`, `Dice`, `Pixel Precision`, `Pixel Recall`.
- **Connected-Component Region Counts (`8`-connectivity):**
  - `gt_components`: connected components in GT binary mask.
  - `missed_gt_regions`: GT components with zero overlapping predicted foreground pixels.
  - `pred_components`: predicted binary components.
  - `fp_pred_regions`: predicted components (`area >= 4` px to ignore isolated sub-pixel specks) with zero overlapping GT pixels.
- **Boundary Quality at Multiple Predefined Tolerances (`tol_px ∈ {2, 5, 10}`):**
  - `1`-px interior boundary `B = mask - erode(mask, 3×3)`.
  - Symmetrically dilated with circular morphological kernel `cv2.MORPH_ELLIPSE((2*tol+1, 2*tol+1))`.
  - Report `Boundary IoU` and `Boundary F1` at **2 px, 5 px, and 10 px** tolerances (treating both-empty as `1.0` and one-empty as `0.0`).
- **Uncertainty Intervals & Disclosed Caveats:**
  - Compute non-parametric percentile bootstrap `95%` confidence intervals (`10,000` resamples, `seed=42`) over the `n=15` per-image scores for macro IoU, Dice, Pixel Precision, Pixel Recall, and `2/5/10`-px Boundary F1.
  - Explicitly report `n=15` small-sample sensitivity and uncertain physical independence (grouped capture IDs may share same-day sessions or nearby pavement stretches).

---

## 4. Box-vs-Mask Representation Comparison Protocol (Task 3)

To isolate the geometric value of segmentation masks over axis-aligned bounding boxes without confounding detector differences:
- **Identical Predictions:** Run `YOLOv8n-seg` (`best.pt`) once per test image at `imgsz=640`, `conf=0.50`, `iou_nms=0.70`, `retina_masks=True` (and secondary reference `conf=0.25`).
- **Matched Rectangular Box Mask:** For the exact set of surviving instances, construct a union binary mask by filling each instance's `boxes.xyxy` rectangle `[max(0, floor(x1)):min(W, ceil(x2)), max(0, floor(y1)):min(H, ceil(y2))]`.
- **Matched Segmentation Mask:** Union of resized (`INTER_LINEAR`, `>= 0.50`) instance masks from the exact same predictions.
- **Metrics Reported Side-by-Side:**
  - Macro & Micro `IoU` and `Dice` for Box Mask vs Segmentation Mask.
  - False-Positive Pavement Area (`fp_px` total/mean and percentage of non-pothole background pixels misclassified).
  - False-Negative Cavity Area (`fn_px` total/mean and percentage of ground-truth pothole cavity pixels missed).
  - Absolute and relative improvement in IoU, Dice, and False-Positive Pavement Area reduction.

---

## 5. Pre-Registered Dashcam Domain-Transfer & Video Demo Rules (Tasks 4 & 5)

- **No Fabricated Segmentation Metrics:** Dashcam / phone road datasets lack pixel segmentation masks; segmentation IoU/Dice will **not** be computed on dashcam frames.
- **Empty Label Handling:** Any dataset directory with unreviewed empty `.txt` labels (specifically `evaluation/ground_truth_labels/` which contains `40` `PENDING_MANUAL_LABEL` empty files) will **never** be treated as verified negative ground truth.
- **Verified Detection Sets Only:** Detection metrics (if reported alongside visual comparison) are computed strictly on sets with non-empty verified labels (`dataset_v2_candidates/` `35` Rangoon frames with `76` reviewed boxes and `dataset_phone_potholes/images/val` `14` frames with `52` boxes), accompanied by explicit annotation-quality and exploratory-set caveats.
- **Video Demonstration (`experiments/demo_pothole_segmentation_video.py`):** Research-only standalone script preserving input dimensions, reporting per-frame inference latency and **per-frame raw detection counts** (never misrepresented as unique physical pothole counts without validated multi-object tracking).
