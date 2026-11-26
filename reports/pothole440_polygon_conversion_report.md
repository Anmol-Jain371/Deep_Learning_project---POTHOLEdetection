# Phase 4A.1: Pothole440 Binary Mask to YOLO Polygon Conversion Report

> **Conversion Script:** [`experiments/build_pothole440_seg_grouped.py`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/build_pothole440_seg_grouped.py)  
> **Visual Inspection Overlays:** `24` individual overlays + `4` grids in [`runs/benchmark/pothole440_seg_preparation/`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_preparation)

---

## 1. Conversion Parameters & Topology Handling

* **Contour Extraction:** `cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` on `{0, 255}` binary foreground masks (`1200×800`).
* **Simplification Setting:** `cv2.approxPolyDP(cnt, epsilon = 0.0005 * arcLength, closed=True)` with automatic fallback to raw contour if fewer than `3` vertices result. All normalized $(x/W, y/H)$ coordinates are clamped strictly to `[0.0, 1.0]` and written under `class_id = 0` (`pothole`).
* **Small-Region Preservation:** Only contours with `< 4.0 px²` area (single-pixel noise) were rejected; zero valid small potholes were discarded (`dropped_degenerate_contours = 0` across all `331` exported images).
* **Empty Masks:** All `6` empty-mask augmented crops across source splits were visually verified and quarantined (`QUARANTINED_EMPTY_MASK`).

---

## 2. Disconnected Components, Holes & Instance-vs-Semantic Limitation

Inspection of the original binary masks established:
* **Disconnected Components:** `9` source images (`2022_05_22_pothole_16_vertical_aug27` [`2`], `2022_05_25_pothole_8*` [`3`], `2022_06_03_pothole_18*` [`4`]) contain `2–4` disconnected foreground regions ([`multicomp_holes_audit.jpg`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_preparation/multicomp_holes_audit.jpg)). These were preserved explicitly as separate polygon lines (`351` total polygons across `331` images: `321` train, `15` val, `15` test).
* **Semantic vs. True Instance Caveat:** Because the source Pothole440 masks are single-channel binary `{0, 255}` masks without per-instance IDs, disconnected fragments near edges (e.g., `2022_05_22_pothole_16_vertical_aug27`) vs. clustered adjacent cavities (e.g., `2022_06_03_pothole_18`) cannot be authoritatively separated from ground truth alone. Thus, **Pothole440 natively supports single-class binary/semantic pothole region segmentation rather than guaranteed multi-instance separation**.
* **Interior Holes Limitation:** `31` source images contain `1–2` tiny interior background holes (`cv2.RETR_CCOMP` parent hierarchy check). Because standard YOLO-Seg `.txt` format represents each object as a single non-holed outer polygon, `RETR_EXTERNAL` fills interior pinholes during conversion; this slight area fill is directly quantified below.

---

## 3. Mask Reconstruction Accuracy (Original Binary Mask vs. YOLO Polygon)

All `331` exported `.txt` label files were parsed, rasterized back into `1200×800` masks via `cv2.fillPoly`, and evaluated pixel-wise against the original binary PNG masks:

| Metric | Mean | Median | 10th Percentile | 5th Percentile | Minimum | Maximum | Flagged (`IoU < 0.96`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **IoU** | **0.99838** | **0.99834** | **0.99722** | **0.99620** | **0.99113** | **1.00000** | **0 / 331** |
| **Dice** | **0.99919** | **0.99917** | **0.99861** | **0.99810** | **0.99555** | **1.00000** | — |

* **Integrity Checks:** `0` image-label pairing mismatches; `0` malformed polygon rows; `0` out-of-range (`<0` or `>1`) coordinates.
* **Visual Verification:** Inspected `24` training RGB/polygon overlays across [`train_overlays_grid_1.jpg`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_preparation/train_overlays_grid_1.jpg)–[`grid_4.jpg`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_preparation/train_overlays_grid_4.jpg) (covering the lowest-IoU `0.9911` hole-containing `2022_06_02_pothole_5` samples, multi-component `pothole_8`/`pothole_18` samples, and high-precision `1.0000` boundaries). No test images or model predictions were visualized or used for tuning.
