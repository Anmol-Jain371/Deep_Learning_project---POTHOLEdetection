# Phase 13 — Detection Model Evaluation (YOLOv8n vs YOLOv8s)

**Run ID:** `phase13_full_20261127`  
**Output Artifact:** `runs/evaluation/phase13/phase13_full_20261127/result_detection.json`, `tables/detection_comparison.csv`, `figures/fig1_detection_comparison.png`  
**Evaluation Settings:** `conf = 0.25`, `iou = 0.50`, `imgsz = 640`, `device = cuda` (NVIDIA GeForce RTX 2050, 4 GB VRAM)  
**Generalization Status:** **`BLOCKED` for independent generalization** (`evaluation/ground_truth_labels/` has `0/40` annotated files; `dataset_unified` exhibits `328` cross-split perceptual near-duplicate pairs `dHash <= 6` and prior Phase 2 inspection). Metrics below are **EXPLORATORY ONLY**.

---

## 1. Controlled Accuracy Comparison (Exploratory Splits)

| Model Checkpoint | Params | Split (Role, Independent?) | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---|---:|---:|---:|---:|---:|
| `yolov8n_100e_sgd` | 3,138,003 | `dataset_unified_val` (dev val, No) | 0.5236 | 0.3569 | 0.4245 | 0.2533 | 0.0884 |
| `yolov8s_100e` | 11,135,987 | `dataset_unified_val` (dev val, No) | 0.4888 | 0.4212 | 0.4525 | 0.2885 | 0.1073 |
| `yolov8n_100e_sgd` | 3,138,003 | `dataset_unified_test` (exploratory, No) | 0.6100 | 0.4565 | 0.5222 | 0.3643 | 0.1442 |
| `yolov8s_100e` | 11,135,987 | `dataset_unified_test` (exploratory, No) | 0.5404 | 0.4783 | 0.5074 | 0.3581 | 0.1427 |
| `yolov8n_100e_sgd` | 3,138,003 | `dataset_unified_external_rangoon` (exploratory clip, No) | 0.5588 | 0.2500 | 0.3455 | 0.1560 | 0.0493 |
| `yolov8s_100e` | 11,135,987 | `dataset_unified_external_rangoon` (exploratory clip, No) | 0.5366 | 0.2895 | 0.3761 | 0.1728 | 0.0541 |

*(Note: Precision/Recall/mAP are extracted directly from `val_res.box`; `external_rangoon` consists of `35` continuous frames from `Watch Video_ Terrible potholes in Rangoon Road.mp4` with documented incomplete ground-truth coverage.)*

## 2. Synchronized GPU Runtime Comparison (`N = 60` images, RTX 2050)

| Model | Mean (ms) | Median (ms) | P95 (ms) | Throughput (FPS) | Peak Alloc VRAM (MB) | Peak Reserved VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| `yolov8n_100e_sgd` | 12.383 | 11.021 | 18.955 | **80.75** | **68.49** | 646.0 |
| `yolov8s_100e` | 25.710 | 24.629 | 33.058 | 38.90 | 200.77 | 1074.0 |

`yolov8n_100e_sgd` achieves **2.08x higher sustained throughput** and uses **2.93x less peak allocated VRAM**, leaving headroom inside 4 GB VRAM to share GPU memory cleanly with DAv2 Small / MiDaS depth inference.

## 3. Qualitative Exploratory Proposal Observations
- `rangoon_clip` frame `60` (`water/shadow/urban_boundary`): `3` detector proposals (`1` boundary-clipped, confidence `0.38–0.47`); frames `150` & `330`: `0` proposals.
- `ruralRoad_202f` (`unpaved_texture/distant_small`): small distant and gravel-texture regions produce varying candidate counts without ground-truth labels (`accuracy` NOT computed on unlabeled footage).
