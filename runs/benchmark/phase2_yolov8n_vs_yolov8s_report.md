# Phase 2 Controlled Architecture Comparison: YOLOv8n vs. YOLOv8s

- **Dataset Manifest:** `dataset_unified_v1_frozen` (`SHA256: 125384470fb2e9636349fb6de75faec998043248884d5a2b352dc1380fbaa8d7`)
- **Shared Settings:** `SGD`, `lr0=0.01`, `cos_lr=True`, `epochs=100`, `imgsz=640`, `close_mosaic=10`, `patience=30`, `amp=True`, `workers=2`, `seed=42`, `device=0`
- **Note on Evaluation Splits:** `val` (153 imgs) was used strictly for checkpoint selection (`best.pt`). `test` (153 imgs) and `external_rangoon` (35 imgs) were previously inspected during Phase 1/exploration and are explicitly reported as **Exploratory Benchmark Splits**; an additional untouched final test set is required for final unbiased model certification.

## 1. Model Capacity, Training Telemetry & Hardware Footprint

| Metric | YOLOv8n (`yolov8n.pt` -> SGD 100e) | YOLOv8s (`yolov8s.pt` -> SGD 100e) |
|---|:---:|:---:|
| Parameters | 3,138,003 (3.138M) | 11,135,987 (11.136M) |
| Checkpoint Size | 6.21 MB | 21.48 MB |
| Batch Size Used | 8 | 8 |
| Epochs Completed | 89 | 87 |
| Best Epoch (Fitness) | 59 | 57 |
| Best Epoch (mAP@50) | 59 | 83 |
| Training Duration | 34.64 min | 57.1 min |
| Peak GPU Allocated | 1050.37 MB | 1772.96 MB |
| Peak GPU Reserved | 1396.0 MB | 2404.0 MB |
| Predict Latency (No TTA) | 15.74 ms (63.54 FPS) | 22.88 ms (43.71 FPS) |
| Predict Latency (TTA) | 38.53 ms (25.95 FPS) | 50.01 ms (20.0 FPS) |

## 2. Losses at Best Validation Epoch & Final Epoch

| Loss Term | YOLOv8n (Best / Final) | YOLOv8s (Best / Final) |
|---|:---:|:---:|
| Train Box Loss | 1.7698 / 1.6131 | 1.6595 / 1.3156 |
| Val Box Loss | 2.2551 / 2.4359 | 2.3887 / 2.4668 |
| Train Cls Loss | 1.4885 / 1.2122 | 1.2205 / 0.8718 |
| Val Cls Loss | 1.9823 / 2.0148 | 1.9224 / 2.0037 |

## 3. Detection Accuracy Across Identical Splits (No TTA vs. TTA)

| Split & Mode | Model | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Val (153 imgs) — No TTA | YOLOv8n | 0.4876 | 0.3794 | 0.4267 | 0.3794 | 0.1287 |
| Val (153 imgs) — No TTA | YOLOv8s | 0.5262 | 0.3762 | 0.4387 | 0.3646 | 0.1316 |
| Val (153 imgs) — TTA | YOLOv8n | 0.4097 | 0.3955 | 0.4025 | 0.3465 | 0.1241 |
| Val (153 imgs) — TTA | YOLOv8s | 0.4354 | 0.3987 | 0.4162 | 0.3513 | 0.1298 |
| Test Exploratory (153 imgs) — No TTA | YOLOv8n | 0.5547 | 0.4969 | 0.5242 | 0.4865 | 0.1960 |
| Test Exploratory (153 imgs) — No TTA | YOLOv8s | 0.5658 | 0.4317 | 0.4897 | 0.4519 | 0.1785 |
| Test Exploratory (153 imgs) — TTA | YOLOv8n | 0.5339 | 0.4969 | 0.5147 | 0.4819 | 0.2102 |
| Test Exploratory (153 imgs) — TTA | YOLOv8s | 0.5509 | 0.4317 | 0.4841 | 0.4738 | 0.1917 |
| External Rangoon Exploratory (35 imgs) — No TTA | YOLOv8n | 0.4173 | 0.2895 | 0.3418 | 0.2359 | 0.0756 |
| External Rangoon Exploratory (35 imgs) — No TTA | YOLOv8s | 0.4653 | 0.3421 | 0.3943 | 0.2758 | 0.0853 |
| External Rangoon Exploratory (35 imgs) — TTA | YOLOv8n | 0.4813 | 0.4211 | 0.4492 | 0.3488 | 0.1012 |
| External Rangoon Exploratory (35 imgs) — TTA | YOLOv8s | 0.4600 | 0.3684 | 0.4092 | 0.3434 | 0.1071 |

## 4. Artifact Paths (Confusion Matrices & PR Curves)

- **YOLOv8n Training Curves & PR/CM:** `D:\DeepLearning\Deep_Learning_project---POTHOLEdetection\runs\detect\pothole_unified_yolov8n_100e_sgd\results.png`, `D:\DeepLearning\Deep_Learning_project---POTHOLEdetection\runs\detect\pothole_unified_yolov8n_100e_sgd\BoxPR_curve.png`, `D:\DeepLearning\Deep_Learning_project---POTHOLEdetection\runs\detect\pothole_unified_yolov8n_100e_sgd\confusion_matrix.png`
- **YOLOv8s Training Curves & PR/CM:** `D:\DeepLearning\Deep_Learning_project---POTHOLEdetection\runs\detect\pothole_unified_yolov8s_100e\results.png`, `D:\DeepLearning\Deep_Learning_project---POTHOLEdetection\runs\detect\pothole_unified_yolov8s_100e\BoxPR_curve.png`, `D:\DeepLearning\Deep_Learning_project---POTHOLEdetection\runs\detect\pothole_unified_yolov8s_100e\confusion_matrix.png`
