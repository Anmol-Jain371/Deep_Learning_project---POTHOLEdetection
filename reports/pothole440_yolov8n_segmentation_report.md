# Phase 4A.2 — YOLOv8n Pothole Segmentation Baseline Report

**Status:** Complete (Validation Only — Grouped Holdout Test Split Untouched)  
**Model:** `YOLOv8n-seg` (initialized strictly from COCO-pretrained `yolov8n-seg.pt`)  
**Dataset:** [dataset_pothole440_seg_grouped/data.yaml](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_pothole440_seg_grouped/data.yaml)  
**Training Run:** [runs/segment/pothole440_yolov8n_seg_v1/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1)  
**Validation Report JSON:** [runs/benchmark/pothole440_segmentation_validation.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_segmentation_validation.json)  
**Per-Image Validation CSV:** [runs/benchmark/pothole440_segmentation_per_image.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_segmentation_per_image.csv)

---

## 1. Executive Summary

A lightweight `YOLOv8n-seg` instance segmentation baseline was trained from standard COCO-pretrained weights on the leakage-controlled `dataset_pothole440_seg_grouped` training split (`301` images across `68` base capture groups, `321` polygons) and evaluated exclusively on the unaugmented validation split (`15` original images across `15` base capture groups). The grouped holdout test split (`15` images) remains untouched.

Key takeaways:
- **Checkpoint & Selection:** Early stopping triggered at epoch `77` (`591.92 s` / `9.87 min` on RTX 2050, `batch=8`). Validation fitness selected **epoch 52** (`best.pt`: `3,390,771` parameters, `6.7` MiB on disk).
- **Instance Mask Detection (Ultralytics `val`):** Achieved `0.9536` mask precision, `0.8667` mask recall (`F1 = 0.9081`), `0.9637` mask `mAP@50`, and `0.7045` mask `mAP@50-95`.
- **Region-Level Foreground Quality vs Original 1200×800 PNG Masks (`conf=0.25`):** Across all `15` validation images (including empty predictions scored as zero masks), macro mean IoU is `0.7646` (median `0.8625`), macro mean Dice is `0.8337` (median `0.9262`), micro-pooled IoU is `0.7911`, and micro-pooled Dice is `0.8834`.
- **Boundary Precision Limit:** 2-px boundary F1 averages `0.3613` (`mean boundary IoU = 0.2089`), reflecting smooth `640×640` prototype mask upsampling onto irregular `1200×800` asphalt edges rather than catastrophic localization errors.
- **Standalone RTX 2050 Latency:** `14.65 ms` mean predict-only (`68.24 FPS`) and `15.87 ms` mean including `1200×800` binary mask union reconstruction (`62.99 FPS`), using `70.24 MB` peak allocated VRAM.
- **Metric Scope Clarification:** Polygon-rasterization fidelity measured during dataset preparation (`mean IoU = 0.99838`, `mean Dice = 0.99919`) quantifies ground-truth label conversion accuracy only and is not model accuracy. Likewise, standalone `YOLOv8n-seg` speed does not represent end-to-end RoadVision pipeline latency (which also includes ByteTrack, MiDaS depth, and planar road calibration).

---

## 2. Pre-Training Verification & Environment

All pre-training integrity checks passed before launching training via [experiments/train_pothole_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/train_pothole_segmentation.py):

- **Environment & Hardware:**
  - PyTorch `2.6.0+cu124`, CUDA available (`True`), Ultralytics `8.4.211`
  - GPU: `NVIDIA GeForce RTX 2050` (`4095.5 MB` VRAM)
- **Frozen Dataset Manifests Verified:** ([dataset_manifest.sha256](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_pothole440_seg_grouped/manifests/dataset_manifest.sha256))
  - `dataset_manifest.json`: `cff0e21e8341a96a7c95c114aa1ea1509874c1047dfae1a9b2ea85c1fe972a16`
  - `split_manifest.csv`: `63b2a1ca49e49b1fd859c8f9b809a9bbba67ff51c629f0092ea0bd7bd8dae643`
  - `data.yaml`: `bd97956a2912ece0a30500ac040c86380aae6b8327d696ec170688a027cb8cfc`
- **Split & Label Correspondence (Train/Val Only):**
  - `train`: `301` images, `301` label files, `321` valid YOLO segmentation polygons
  - `val`: `15` unaugmented images, `15` label files, `15` valid YOLO segmentation polygons
  - Ultralytics `check_det_dataset` passed (`nc: 1`, `{0: 'pothole'}`)
- **Base Checkpoint Provenance:**
  - Weights: standard COCO-pretrained `yolov8n-seg.pt` (downloaded cleanly; existing pothole detection weights untouched)
  - SHA256: `a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`
  - File size: `6.7` MiB (`7,071,756` bytes), `task: segment`, `nc: 80`, `epoch: -1`, `train_args.data: coco.yaml`, `3,409,968` parameters

---

## 3. Training Configuration, VRAM Probing & Convergence

### 3.1 Hyperparameters & Conservative Augmentation Policy

Because `dataset_pothole440_seg_grouped` already contains `234` offline-augmented training images (`77.7%` of `301`), online geometric mixing and distortion augmentations were disabled or kept mild to prevent compounding warping artifacts:

| Setting | Value | Rationale |
|---|---|---|
| Base Model | `yolov8n-seg.pt` | COCO-pretrained lightweight baseline |
| Image Size / Epochs | `640` / `100` (`patience=25`) | Standard resolution; early stopping enabled |
| Optimizer / LR Schedule | `SGD`, `lr0=0.01`, `cos_lr=True` | Controlled optimization with cosine decay |
| Batch Size | `8` (probed safely from `4`) | Auto-batch VRAM probe verified headroom |
| Mixed Precision / Seed | `amp=True`, `seed=42`, `workers=2` | RTX 2050 memory efficiency & reproducibility |
| Disabled Augmentations | `mosaic=0.0`, `mixup=0.0`, `copy_paste=0.0`, `cutmix=0.0`, `degrees=0.0`, `shear=0.0`, `perspective=0.0`, `flipud=0.0`, `erasing=0.0` | Avoid stacking distortions onto offline-augmented samples |
| Conservative Augmentations | `fliplr=0.5`, `translate=0.05`, `scale=0.15`, `hsv_h=0.01`, `hsv_s=0.3`, `hsv_v=0.25` | Mild spatial & photometric regularization only |

*(Note: `close_mosaic=10` was configured as specified, though `mosaic=0.0` remained disabled throughout training. Exact run arguments are logged in [args.yaml](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1/args.yaml) and [phase4a2_train_telemetry.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1/phase4a2_train_telemetry.json).)*

### 3.2 VRAM Probing & GPU Utilization

Short CUDA memory probes ran prior to full training on the RTX 2050 (`4095.5 MB` total):
- `batch=4`: `646.25 MB` peak allocated, `824.0 MB` peak reserved (`20.12%` of VRAM) — passed
- `batch=8`: `1188.42 MB` peak allocated, `1400.0 MB` peak reserved (`34.18%` of VRAM) — passed and selected
- **Full Training Peak VRAM:** `1282.07 MB` allocated / `1520.0 MB` reserved (`37.11%` of `4095.5 MB`), zero OOM events
- **Training Duration:** `591.92 s` (`9.87 min`) for `77` completed epochs

### 3.3 Training & Validation Convergence

Training stopped at **epoch 77** via early stopping (`patience=25`), restoring **epoch 52** selected by Ultralytics validation fitness (`0.1*(mAP50_B + mAP50_M) + 0.9*(mAP50-95_B + mAP50-95_M)`).

| Metric | Best Epoch (`52`) | Final Epoch (`77`) | Observation |
|---|---:|---:|---|
| `train/box_loss` | `0.3720` | `0.2221` | Monotonic convergence on training split |
| `train/seg_loss` | `0.5698` | `0.4549` | Smooth mask loss reduction |
| `train/cls_loss` | `0.2800` | `0.1819` | Steady foreground/background separation |
| `train/dfl_loss` | `0.8316` | `0.7890` | Gradual box distribution refinement |
| `val/box_loss` | `1.2013` | `1.2591` | Plateaus then drifts slightly upward |
| `val/seg_loss` | `1.5732` | `1.7868` | Overfitting signal after epoch ~52–60 |
| `val/cls_loss` | `0.6830` | `1.2692` | Classification confidence divergence late in run |
| `val/dfl_loss` | `1.2433` | `1.2422` | Flat |
| `mask/mAP50` | `0.9637` | `0.9194` | Peaks at epoch 52 (tied at epoch 56) |
| `mask/mAP50-95` | `0.7045` | `0.6401` | Peaks at epoch 52 |

- Trained artifact: [best.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1/weights/best.pt) (`SHA256: 73f525322681468322dd129589279b016eeeab1197879a8876e3071e8e0d6d6d`, `3,390,771` parameters; fused eval summary: `86` layers, `3,385,123` parameters, `11.9 GFLOPs`).
- Visual training curves are available in [results.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1/results.png).

---

## 4. Quantitative Validation Results (`val` Split Only)

Validation was executed using [experiments/evaluate_pothole_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/evaluate_pothole_segmentation.py). Two complementary metric families are reported separately because Ultralytics `val` evaluates instance detection/mask predictions against YOLO polygon labels (`nc=1`), whereas RoadVision downstream area/geometry depends on full-resolution (`1200×800`) binary mask fidelity against the original Pothole440 PNG ground-truth masks.

### 4.1 Ultralytics Instance Validation Metrics (`val_eval`)

Artifacts saved in [runs/benchmark/pothole440_seg_ultralytics_val/val_eval/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_ultralytics_val/val_eval):

| Head | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| **Box** | `0.9536` | `0.8667` | `0.9081` | `0.9637` | `0.6047` |
| **Mask** | `0.9536` | `0.8667` | `0.9081` | `0.9637` | `0.7045` |

Notably, `mask mAP@50-95` (`0.7045`) exceeds `box mAP@50-95` (`0.6047`), indicating that predicted segmentation masks often fit the irregular pothole cavities more tightly at higher IoU thresholds than axis-aligned bounding boxes.

### 4.2 Region-Level Binary Mask Metrics vs Original `1200×800` PNG Masks

Predicted instance masks (`retina_masks=True`, `imgsz=640`, `iou_nms=0.7`) were unioned per image at `1200×800` and compared pixel-for-pixel against original Pothole440 binary PNG masks. Images with zero detections (`empty_prediction_images=1`) are included in all macro and micro averages as empty masks rather than excluded.

#### Primary Operating Point (`conf=0.25`) vs Confidence Sensitivity (`0.15`, `0.50`)

| Metric | `conf=0.15` | `conf=0.25` (Primary) | `conf=0.50` |
|---|---:|---:|---:|
| Empty-Prediction Images | `1 / 15` | `1 / 15` | `1 / 15` |
| **Macro Mean IoU** | `0.7620` | `0.7646` (std `0.2506`) | `0.7871` (std `0.2287`) |
| **Macro Median IoU** | `0.8625` | `0.8625` (min `0.0`, max `0.9286`) | `0.8625` (max `0.9286`) |
| **Macro Mean Dice** | `0.8323` | `0.8337` (std `0.2512`) | `0.8525` (std `0.2396`) |
| **Macro Median Dice** | `0.9262` | `0.9262` | `0.9262` |
| Macro Mean Pixel Precision | `0.8338` | `0.8364` | `0.8603` |
| Macro Mean Pixel Recall | `0.8520` | `0.8520` | `0.8519` |
| Macro Mean 2-px Boundary IoU | `0.2073` | `0.2089` | `0.2131` |
| Macro Mean 2-px Boundary F1 | `0.3592` | `0.3613` | `0.3682` |
| **Micro-Pooled IoU** | `0.7885` | **`0.7911`** | `0.8190` |
| **Micro-Pooled Dice** | `0.8818` | **`0.8834`** | `0.9005` |
| Micro-Pooled Pixel Precision | `0.8901` | `0.8934` | `0.9291` |
| Micro-Pooled Pixel Recall | `0.8736` | `0.8736` | `0.8735` |
| Completely Missed GT Regions | `1 / 15` | `1 / 15` | `1 / 15` |
| False-Positive Mask Regions | `3` | `2` | `1` |

Sensitivity takeaway:
- Raising `conf` from `0.25` to `0.50` increases macro mean IoU (`+0.0225`), micro-pooled IoU (`+0.0279`), and cuts false-positive mask regions (`2 -> 1`) without losing recall (`micro pixel recall` drops only `0.8736 -> 0.8735`).
- Lowering `conf` to `0.15` does **not** recover the single missed pothole (`completely_missed_gt_regions` stays `1`) while adding an extra false-positive mask region (`2 -> 3`).

### 4.3 Per-Image Validation Breakdown (`conf=0.25`)

Full CSV: [runs/benchmark/pothole440_segmentation_per_image.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_segmentation_per_image.csv)

| Validation Image | Pred Inst | GT Area (%) | Pred Area (%) | IoU | Dice | Pixel Prec | Pixel Rec | 2px B-F1 | Missed GT / FP Regions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2022_05_22_pothole_17.png` | `2` | `3.006` | `7.225` | `0.3901` | `0.5613` | `0.3974` | `0.9552` | `0.1909` | `0 / 1` |
| `2022_05_22_pothole_19.png` | `1` | `2.103` | `2.211` | `0.8863` | `0.9397` | `0.9168` | `0.9639` | `0.5300` | `0 / 0` |
| `2022_05_23_pothole_4.png` | `1` | `7.180` | `7.607` | `0.9286` | `0.9630` | `0.9359` | `0.9916` | `0.4318` | `0 / 0` |
| `2022_05_25_pothole_3.png` | `1` | `7.558` | `8.180` | `0.8625` | `0.9262` | `0.8910` | `0.9643` | `0.3102` | `0 / 0` |
| `2022_06_01_pothole_16.png` | `1` | `6.263` | `6.203` | `0.8307` | `0.9075` | `0.9119` | `0.9031` | `0.3978` | `0 / 0` |
| `2022_06_01_pothole_23.png` | `1` | `12.276` | `8.754` | `0.6962` | `0.8209` | `0.9860` | `0.7031` | `0.3157` | `0 / 0` |
| `2022_06_01_pothole_25.png` | `1` | `8.840` | `8.683` | `0.9190` | `0.9578` | `0.9664` | `0.9493` | `0.3698` | `0 / 0` |
| `2022_06_01_pothole_7.png` | `1` | `2.549` | `2.314` | `0.8222` | `0.9024` | `0.9481` | `0.8609` | `0.3952` | `0 / 0` |
| `2022_06_02_pothole_4.png` | `1` | `9.837` | `10.149` | `0.9011` | `0.9480` | `0.9334` | `0.9630` | `0.4868` | `0 / 1` |
| `2022_06_03_pothole_1.png` | `1` | `13.828` | `14.379` | `0.9052` | `0.9503` | `0.9320` | `0.9692` | `0.3807` | `0 / 0` |
| `2022_06_03_pothole_21.png` | `2` | `2.349` | `2.014` | `0.7930` | `0.8845` | `0.9582` | `0.8214` | `0.4252` | `0 / 0` |
| `2022_06_03_pothole_23.png` | `1` | `6.560` | `6.467` | `0.8985` | `0.9465` | `0.9534` | `0.9398` | `0.4892` | `0 / 0` |
| `2022_06_03_pothole_26.png` | `1` | `2.787` | `2.439` | `0.8626` | `0.9262` | `0.9922` | `0.8684` | `0.3012` | `0 / 0` |
| `2022_06_03_pothole_5.png` | `0` | `3.798` | `0.000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1 / 0` |
| `2022_06_04_pothole_18.png` | `1` | `2.219` | `2.499` | `0.7723` | `0.8715` | `0.8226` | `0.9266` | `0.3954` | `0 / 0` |

---

## 5. Qualitative Visual Error Analysis

All `15` validation visualizations were generated in [runs/benchmark/pothole440_seg_val_viz/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_val_viz). Six representative cases were visually inspected:

1. **Strong Core Cavity Segmentation ([val_03_2022_05_23_pothole_4.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_val_viz/val_03_2022_05_23_pothole_4.png), IoU `0.9286`, Dice `0.9630`):**
   - Vertical crack-aligned pothole localized cleanly (`1` instance, `68,349` TP px, `577` FN px). Errors consist almost exclusively of a thin `4,679`-pixel perimeter halo (`FP`) where the upsampled prototype mask is slightly smoother than the jagged ground-truth boundary (explaining `2-px B-F1 = 0.4318` despite `92.86%` IoU).
2. **Clean Horizontal Cavity Fit ([val_06_2022_06_01_pothole_25.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_val_viz/val_06_2022_06_01_pothole_25.png), IoU `0.9190`, Dice `0.9578`):**
   - Dark cavity along a pavement seam captured with `80,559` TP px, `2,799` FP px, and `4,305` FN px. Minor mismatches occur along fine serrations on the upper edge and right corner.
3. **Boundary Spill / Crack Fragmentation ([val_08_2022_06_02_pothole_4.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_val_viz/val_08_2022_06_02_pothole_4.png), IoU `0.9011`, Dice `0.9480`):**
   - Single detected instance (`inst=1`) captures the dark left-bounded pothole (`90,941` TP px) but produces `3` disconnected binary mask components (`1` counted FP region): a small detached sliver leaks onto a branching crack near the top right (`6,491` total FP px).
4. **Partial Under-Segmentation in Textured Interior ([val_05_2022_06_01_pothole_23.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_val_viz/val_05_2022_06_01_pothole_23.png), IoU `0.6962`, Dice `0.8209`):**
   - Very high pixel precision (`0.9860`, `1,175` FP px) paired with low pixel recall (`0.7031`, `34,989` FN px). The model segments the darker upper depression accurately but truncates the lower narrow tail containing light gray stone/debris whose appearance blends with adjacent weathered asphalt.
5. **Curb/Seam False-Positive Over-Prediction ([val_01_2022_05_22_pothole_17.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_val_viz/val_01_2022_05_22_pothole_17.png), IoU `0.3901`, Dice `0.5613`):**
   - High recall (`0.9552`, `27,565` TP px) but heavy over-prediction (`41,795` FP px, `inst=2`). Beyond the labeled recessed pocket (`3.006%` area), a second prediction spans an adjacent weathered/shadowed strip along the horizontal concrete curb joint (`7.225%` predicted area). Increasing confidence threshold to `0.50` eliminates one FP mask region across the validation set.
6. **Complete Miss Under Dappled Foliage Shadows ([val_12_2022_06_03_pothole_5.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_val_viz/val_12_2022_06_03_pothole_5.png), IoU `0.0000`, Dice `0.0000`):**
   - Complete false negative (`inst=0`, `36,456` FN px) even at `conf=0.15`. Dappled tree shadows cross a shallow, cracked pothole (`3.798%` area) with minimal color contrast relative to surrounding weathered concrete/asphalt, suppressing objectness/classification score below detection threshold.

---

## 6. Standalone Latency Benchmark (RTX 2050)

Measured over `150` timed `640×640` GPU inference passes (`15` validation images × `10` passes after warmup, `retina_masks=True`) on the `NVIDIA GeForce RTX 2050`:

| Benchmark Mode | Mean (ms) | Median (ms) | p95 (ms) | Throughput (FPS) |
|---|---:|---:|---:|---:|
| **Predict Only** (preprocess + forward + NMS + mask upsample) | `14.65` | `13.02` | `21.62` | `68.24` |
| **Predict + Binary Union Reconstruction (`1200×800`)** | `15.87` | `14.35` | `22.82` | `62.99` |

- **Peak Inference VRAM:** `70.24 MB` allocated / `208.0 MB` reserved.
- **Scope:** Standalone `YOLOv8n-seg` image inference only. Total RoadVision runtime latency also includes ByteTrack tracking, MiDaS depth inference, annular donut depth extraction, and planar road calibration.

---

## 7. Scientific Limitations & Readiness Notes

1. **Small Validation Sample Size (`n=15`):** Because each base capture group contributes a single unaugmented original frame to `val`, one complete miss (`val_12`) lowers macro mean IoU by roughly `~0.05–0.06` (`0.7646` macro vs `0.8625` median). Model comparisons should inspect both macro/median and micro-pooled metrics.
2. **Binary Mask Granularity vs Instance Ground Truth:** Original Pothole440 annotations are single-channel foreground PNGs without instance IDs (`15` connected GT regions across `15` validation frames). Multi-instance overlaps or split predictions on a single cavity (e.g., `2022_06_03_pothole_21.png` with `pred_instances=2`, or `val_08` crack fragment) are captured via `gt_components`, `missed_gt_regions`, `fp_pred_regions`, and region-union IoU/Dice, complementing Ultralytics per-instance `mAP`.
3. **Training-Split Augmentation Composition & Mild Overfitting:** `train` includes `234` offline-augmented variants across `68` base capture groups; `val_seg_loss` and `val_cls_loss` began rising after epochs `52–60` while training losses continued dropping, justifying early stopping (`epoch 52` selection).
4. **Domain Shift Relative to RoadVision Dashcam Streams:** Pothole440 consists of close-range top-down/near-nadir `1200×800` pavement captures rather than forward-facing dashcam perspectives (`dataset_unified` / Rangoon frames). Integrating Pothole440-trained segmentation directly into production dashcam inference without cross-domain validation is premature.
5. **Holdout Test Split Preserved:** `dataset_pothole440_seg_grouped/images/test` and `labels/test` (`15` unaugmented base-group frames) were not evaluated or touched during this phase.

---

## 8. Deliverables & Preserved Assets

- **New / Updated Phase 4A.2 Artifacts:**
  - Training script: [experiments/train_pothole_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/train_pothole_segmentation.py)
  - Validation evaluation script: [experiments/evaluate_pothole_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/evaluate_pothole_segmentation.py)
  - Training run & telemetry: [runs/segment/pothole440_yolov8n_seg_v1/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1) (`best.pt`, `results.csv`, `results.png`, `phase4a2_train_telemetry.json`)
  - Validation JSON & CSV: [pothole440_segmentation_validation.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_segmentation_validation.json), [pothole440_segmentation_per_image.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_segmentation_per_image.csv)
  - Ultralytics `val` plots: [runs/benchmark/pothole440_seg_ultralytics_val/val_eval/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_ultralytics_val/val_eval)
  - Qualitative overlays (`15` PNGs): [runs/benchmark/pothole440_seg_val_viz/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_val_viz)
- **Untouched / Preserved:** `Pothole440-main/`, `dataset_pothole440_seg_grouped/` frozen splits & untouched `test` split, `dataset_unified/`, existing YOLOv8 detection weights/runs, `roadvision_engine.py`, `server.py`, `config/camera_params.json`, and prior research reports.

---

## 9. Recommended Next Step (Awaiting Approval)

Phase 4A.2 is complete. Candidate options for the next step (do not proceed until approved):
- **Option A — Controlled `YOLOv8s-seg` Comparison (Val Only):** Train `yolov8s-seg.pt` under identical leakage-controlled splits and conservative augmentations to test whether extra capacity resolves dappled-shadow false negatives (`val_12`) and textured under-segmentation (`val_05`) without worsening curb false positives (`val_01`).
- **Option B — Shadow/Contrast Regularization Check:** Test whether mild shadow/contrast augmentation (`hsv_v`, random local shadow/brightness perturbation) improves robustness on shadow-occluded shallow potholes (`val_12`) before locking the final segmentation model and evaluating once on the untouched grouped `test` split.
