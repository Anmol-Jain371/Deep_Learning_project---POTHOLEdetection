# Phase 4A.3 — Independent Grouped Test Segmentation Report

**Status:** Complete (Single-Pass Frozen Protocol Evaluation — No Post-Hoc Tuning or Retraining)  
**Benchmark Type:** Reconstructed Grouped Test Benchmark (`dataset_pothole440_seg_grouped/data.yaml`, `split=test`, `15` unaugmented original images)  
**Protocol Specification:** [pothole440_segmentation_evaluation_protocol.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/pothole440_segmentation_evaluation_protocol.md)  
**Raw Results JSON:** [pothole440_segmentation_test.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_segmentation_test.json)  
**Per-Image CSV:** [pothole440_segmentation_test_per_image.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_segmentation_test_per_image.csv)  
**Evaluation Script:** [evaluate_pothole_segmentation_test.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/evaluate_pothole_segmentation_test.py)

---

## 1. Verified Checksums & Frozen Test Conditions

- **Checkpoint:** [best.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1/weights/best.pt) (`SHA256: 73f525322681468322dd129589279b016eeeab1197879a8876e3071e8e0d6d6d`, `3,390,771` parameters, epoch `52` validation selection)
- **Frozen Manifest Checksums:**
  - `dataset_manifest.json`: `cff0e21e8341a96a7c95c114aa1ea1509874c1047dfae1a9b2ea85c1fe972a16`
  - `split_manifest.csv`: `63b2a1ca49e49b1fd859c8f9b809a9bbba67ff51c629f0092ea0bd7bd8dae643`
  - `data.yaml`: `bd97956a2912ece0a30500ac040c86380aae6b8327d696ec170688a027cb8cfc`
- **Inference & Ground Truth Protocol:** `imgsz=640`, `retina_masks=True`, `iou_nms=0.70`, primary region operating threshold `conf=0.50` (pre-registered from `val`), bilinear mask upsampling to `1200×800` and binarized at `>= 0.5`, evaluated against original Pothole440 binary PNG masks (`15` unaugmented test images). Empty predictions (`0` detections) are retained as zero masks.

---

## 2. Quantitative Test Summary

### 2.1 Ultralytics Standard Instance AP Evaluation (`test_eval`)

Saved in [runs/benchmark/pothole440_seg_ultralytics_test/test_eval/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_ultralytics_test/test_eval):

| Head | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 | Validation Baseline (Phase 4A.2) |
|---|---:|---:|---:|---:|---:|---|
| **Box** | `0.6245` | `0.6667` | `0.6449` | `0.4998` | `0.2602` | `mAP@50 = 0.9637`, `mAP@50-95 = 0.6047` |
| **Mask** | `0.6873` | `0.7326` | `0.7092` | `0.6316` | `0.3212` | `mAP@50 = 0.9637`, `mAP@50-95 = 0.7045` |

### 2.2 Region-Level Binary Mask Metrics (`1200×800`, Primary `conf=0.50`)

| Metric | Value (`conf=0.50` Primary) | 95% Bootstrap CI (`10,000` Resamples) | Reference `conf=0.25` | Phase 4A.2 `val` (`conf=0.50`) |
|---|---:|---:|---:|---:|
| **Empty Predictions** | `4 / 15` | — | `3 / 15` | `1 / 15` |
| **Macro Mean IoU** | `0.5996` (std `0.3878`) | `[0.4042, 0.7764]` | `0.6343` `[0.4480, 0.7965]` | `0.7871` |
| **Macro Median / Min / Max IoU** | `0.7128` / `0.0000` / `0.9456` | — | `0.8192` / `0.0000` / `0.9456` | `0.8625` / `0.0000` / `0.9286` |
| **Macro Mean Dice** | `0.6564` (std `0.4147`) | `[0.4455, 0.8447]` | `0.6984` | `0.8525` |
| **Macro Median Dice** | `0.8323` | — | `0.9006` | `0.9262` |
| **Macro Pixel Precision** | `0.6778` | `[0.4520, 0.8734]` | `0.7393` | `0.8603` |
| **Macro Pixel Recall** | `0.6503` | `[0.4367, 0.8477]` | `0.6862` | `0.8519` |
| **Micro-Pooled IoU** | `0.6015` | — | `0.6534` | `0.8190` |
| **Micro-Pooled Dice** | `0.7512` | — | `0.7904` | `0.9005` |
| **Micro Precision / Recall** | `0.9344` / `0.6280` | — | `0.9343` / `0.6849` | `0.9291` / `0.8735` |
| **Missed GT Pothole Regions** | `4 / 15` | — | `3 / 15` | `1 / 15` |
| **False-Positive Mask Regions** | `1` | — | `1` | `1` |

Pooled pixel totals across all `15` test images (`14,400,000` pixels at `conf=0.50`): `TP = 981,070`, `FP = 68,896`, `FN = 581,133`, `TN = 12,768,901`.

### 2.3 Multi-Scale Boundary Quality (`1200×800`, Primary `conf=0.50`)

| Boundary Tolerance | Macro Boundary IoU | Macro Boundary F1 | 95% Bootstrap CI (Boundary F1) | Reference `conf=0.25` Boundary F1 |
|---|---:|---:|---:|---:|
| **2 px** | `0.1568` | `0.2758` | `[0.1785, 0.3695]` | `0.2838` |
| **5 px** | `0.2675` | `0.4377` | `[0.2857, 0.5826]` | `0.4545` |
| **10 px** | `0.3626` | `0.5413` | `[0.3604, 0.7114]` | `0.5652` |

Because `4` of `15` images have `0.0000` boundary score due to missed detections, conditional boundary accuracy on the `11` detected cavities is substantially higher (e.g., up to `0.5395` at 2 px, `0.8214` at 5 px, and `0.9615` at 10 px on `test_03`/`test_05`/`test_10`).

---

## 3. Complete Per-Image Test Results (`n=15`, `conf=0.50`)

| # | Filename | Orig Split | Inst | GT Area (%) | Pred Area (%) | IoU | Dice | Prec | Rec | B-F1 (2/5/10 px) | Missed / FP Reg |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 01 | `2022_05_22_pothole_16.png` | `train` | `2` | `17.184` | `16.525` | `0.8683` | `0.9295` | `0.9480` | `0.9117` | `0.361 / 0.532 / 0.665` | `0 / 0` |
| 02 | `2022_05_23_pothole_1.png` | `train` | `1` | `4.494` | `7.146` | `0.6248` | `0.7691` | `0.6264` | `0.9960` | `0.221 / 0.377 / 0.467` | `0 / 0` |
| 03 | `2022_05_25_pothole_6.png` | `train` | `2` | `10.937` | `11.272` | `0.9308` | `0.9642` | `0.9498` | `0.9789` | `0.429 / 0.780 / 0.959` | `0 / 0` |
| 04 | `2022_06_01_pothole_11.png` | `train` | `1` | `10.087` | `10.162` | `0.8971` | `0.9458` | `0.9423` | `0.9493` | `0.306 / 0.487 / 0.771` | `0 / 0` |
| 05 | `2022_06_01_pothole_19.png` | `train` | `1` | `3.416` | `3.367` | `0.9190` | `0.9578` | `0.9648` | `0.9510` | `0.514 / 0.821 / 0.961` | `0 / 0` |
| 06 | `2022_06_01_pothole_24.png` | `train` | `0` | `20.798` | `0.000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.000 / 0.000 / 0.000` | `1 / 0` |
| 07 | `2022_06_03_pothole_32.png` | `train` | `1` | `15.785` | `14.962` | `0.8795` | `0.9359` | `0.9616` | `0.9115` | `0.231 / 0.416 / 0.686` | `0 / 0` |
| 08 | `2022_06_03_pothole_34.png` | `train` | `0` | `6.224` | `0.000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.000 / 0.000 / 0.000` | `1 / 0` |
| 09 | `2022_06_03_pothole_6.png` | `train` | `0` | `7.261` | `0.000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.000 / 0.000 / 0.000` | `1 / 0` |
| 10 | `2022_06_03_pothole_9.png` | `train` | `1` | `23.404` | `24.002` | `0.9456` | `0.9721` | `0.9600` | `0.9845` | `0.540 / 0.778 / 0.856` | `0 / 1` |
| 11 | `2022_06_04_pothole_13.png` | `train` | `1` | `10.158` | `6.791` | `0.6447` | `0.7840` | `0.9782` | `0.6541` | `0.420 / 0.592 / 0.648` | `0 / 0` |
| 12 | `2022_05_22_pothole_15.png` | `val` | `0` | `14.096` | `0.000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.000 / 0.000 / 0.000` | `1 / 0` |
| 13 | `2022_05_22_pothole_24.png` | `val` | `1` | `3.236` | `3.544` | `0.8631` | `0.9265` | `0.8863` | `0.9705` | `0.450 / 0.708 / 0.836` | `0 / 0` |
| 14 | `2022_06_02_pothole_3.png` | `test` | `2` | `10.270` | `7.598` | `0.7077` | `0.8289` | `0.9746` | `0.7211` | `0.429 / 0.644 / 0.735` | `0 / 0` |
| 15 | `2022_06_03_pothole_33.png` | `test` | `1` | `5.380` | `4.002` | `0.7128` | `0.8323` | `0.9756` | `0.7257` | `0.237 / 0.430 / 0.534` | `0 / 0` |

---

## 4. Qualitative Failure Analysis ([Test Visualizations](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_test_viz))

Visual inspection of test overlays explains the generalization gap from `val` (`0.9637` mask `mAP@50`) to `test` (`0.6316` mask `mAP@50`):

1. **Four Complete False Negatives (`IoU = 0.0` at `conf = 0.50`):**
   - [test_06_2022_06_01_pothole_24.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_test_viz/test_06_2022_06_01_pothole_24.png): Large light-gray aggregate-filled triangular pothole (`20.798%` area) with minimal luminance contrast against surrounding pavement and an adjacent dark circular bitumen patch.
   - [test_08_2022_06_03_pothole_34.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_test_viz/test_08_2022_06_03_pothole_34.png): Shallow dry star-shaped alligator-crack depression (`6.224%` area) lacking crisp shadow rims.
   - [test_09_2022_06_03_pothole_6.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_test_viz/test_09_2022_06_03_pothole_6.png): Shallow cracked pothole crossed by **dappled foliage/tree shadows** (`7.261%` area) — reproducing the exact failure mode observed on validation sample `val_12`.
   - [test_12_2022_05_22_pothole_15.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_test_viz/test_12_2022_05_22_pothole_15.png): Sunlit gray pothole (`14.096%` area) flanked vertically by two black measurement bars/rods resting on the pavement.
2. **Dark Tar-Sealant Over-Segmentation ([test_02_2022_05_23_pothole_1.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_test_viz/test_02_2022_05_23_pothole_1.png), `IoU = 0.6248`):**
   - High recall (`0.9960`) but `25,632` FP pixels (`Precision = 0.6264`) because the mask leaks outward along dark tar crack-sealant strips intersecting the bottom corners of the pothole.
3. **Interior Island Dropout & Clipping ([test_11_2022_06_04_pothole_13.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_test_viz/test_11_2022_06_04_pothole_13.png), `IoU = 0.6447`):**
   - High precision (`0.9782`, `1,419` FP px) but truncated lower neck and an interior circular hole (`33,734` FN px, `Recall = 0.6541`) over a pale gray asphalt patch inside the cavity.
4. **High-Quality Cavity Fit ([test_10_2022_06_03_pothole_9.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_test_viz/test_10_2022_06_03_pothole_9.png), `IoU = 0.9456`, `Dice = 0.9721`):**
   - Captures `221,189` TP pixels of a large `23.404%`-area dark cavity (`10`-px B-F1 `0.856`), with one detached small FP region (`fp_pred_regions = 1`) over a dark bottom-right stain.

---

## 5. Uncertainty, Sample Size & Independence Caveats

1. **Wide Confidence Intervals (`n=15`):** Macro mean IoU on `test` is `0.5996` with `95%` bootstrap CI `[0.4042, 0.7764]` (and `macro Dice = 0.6564`, `95% CI [0.4455, 0.8447]`). Because four binary misses score `0.0`, performance is bimodal (`median IoU = 0.7128` vs `mean IoU = 0.5996`; conditional mean IoU across the `11` detected images is `0.8175`).
2. **Val-to-Test Variance Expected at Small `n`:** While grouped stratification prevented augmentation leakage across splits, `val` (`n=15`) happened to contain only `1` shadow-occluded miss (`val_12`) whereas `test` (`n=15`) contained `4` low-contrast / shadowed / bar-flanked misses (`test_06`, `test_08`, `test_09`, `test_12`), dropping `mask mAP@50` from `0.9637` to `0.6316`.
3. **Uncertain Physical Pothole Independence:** As documented in [pothole440_grouped_split_report.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/pothole440_grouped_split_report.md), base capture groups within the same capture date (`e.g. 2022_06_03_pothole_*`) could come from nearby road segments with shared asphalt/lighting characteristics despite zero perceptual hash duplicates.
