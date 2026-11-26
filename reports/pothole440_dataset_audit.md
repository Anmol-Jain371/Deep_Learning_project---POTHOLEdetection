# Phase 3B: Pothole440 Dataset Audit Report

> **Dataset Path Audited:** `./Pothole440-main/Pothole440-main/`  
> **Inspection Script:** [`experiments/inspect_pothole440.py`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/inspect_pothole440.py)  
> **Inventory JSON:** [`dataset_inventory.json`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_inspection/dataset_inventory.json)

---

## 1. Folder Structure & Documentation Audit

The dataset archive extracts with a nested root (`./Pothole440-main/Pothole440-main/`) containing:
* `train/`, `val/`, `test/` — each holding `rgb/`, `mask/`, and `depth/` PNG directories.
* `model/` — PyTorch network definitions (`3DpredicNet.py`, `unet_model.py`, `yolov11.py`, `yolov11l.py`, `SegFormer.py`, `SwinTransformerSys.py`, `mobilenetv3_small.py`).
* `README.md`, `LICENSE`, `university-of-manitoba-logo.jpg`.
* **Missing Documented Directory:** `README.md` lists an `annotations/` directory (`COCO and Pascal VOC annotation files`), which is **not present** in this extracted release (`annotations_dir_present: false`).

### Git Tracking & `.gitignore` Status
* `git ls-files Pothole440-main` confirmed **0 files** were tracked in Git.
* `/Pothole440-main/` was **not** previously excluded; it has now been added to [`.gitignore`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/.gitignore) and verified via `git check-ignore -v Pothole440-main`.

---

## 2. File Inventory, Pairing & Mask Geometry

| Split | RGB | Mask | Depth | Matched Triplets | Unmatched / Corrupted | Aligned (`1200×800`) | Empty Masks | Valid YOLO Contours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 350 | 350 | 350 | 350 | 0 / 0 | 350 (100%) | 3 | 347 |
| **Val** | 50 | 50 | 50 | 50 | 0 / 0 | 50 (100%) | 2 | 48 |
| **Test** | 40 | 40 | 40 | 40 | 0 / 0 | 40 (100%) | 1 | 39 |
| **Total** | **440** | **440** | **440** | **440** | **0 / 0** | **440 (100%)** | **6** | **434** |

### RGB & Segmentation Mask Properties
* **Format & Dtype:** 100% PNG (`uint8`). RGB images are 3-channel (`800×1200×3`); masks are single-channel binary (`{0, 255}`).
* **Foreground Coverage (% of frame):**
  * Train: min `0.0%`, median `6.245%`, mean `8.679%`, max `59.562%`.
  * Val: min `0.0%`, median `6.456%`, mean `7.812%`, max `33.336%`.
  * Test: min `0.0%`, median `6.405%`, mean `8.511%`, max `33.748%`.
* **Connected Components:** Predominantly single-instance (`338/350` train, `48/50` val, `39/40` test have `1` component; `9` train images have `2–4` components; `6` augmented images across splits have `0` foreground pixels).
* **YOLO Instance Segmentation Feasibility:** All `434` non-empty binary masks have clean contours extractable via `cv2.findContours`, making conversion to normalized YOLO polygon format straightforward. *(The 6 empty-mask files correspond to augmentation crops/transforms such as `2022_05_22_pothole_15_aug20.png` and must be treated carefully or excluded as background frames.)*

---

## 3. Duplicate, Augmentation & Cross-Split Leakage Audit (Critical Finding)

1. **Augmented Expansion of Only 98 Base Captures:**
   * Filename inspection (`_horizantal`, `_vertical`, `_aug*`) shows that **343 of 440 images (78.0%)** are augmented variants of only **`98` unique base pothole captures** across `7` capture dates (`2022_05_22` to `2022_06_04`):
     * `train`: `274 / 350` augmented (`76` base-named)
     * `val`: `39 / 50` augmented (`11` base-named)
     * `test`: `30 / 40` augmented (`10` base-named)
2. **Severe Cross-Split Base-Capture Leakage:**
   * **`43` base pothole IDs** appear in **both `train` and `val`**.
   * **`34` base pothole IDs** appear in **both `train` and `test`** (meaning ~85% of the 40-image test split comes from potholes seen during training!).
   * **`16` base pothole IDs** overlap between **`val` and `test`**.
   * Capture session dates (`2022_05_22`, `2022_05_25`, `2022_06_01`–`2022_06_04`) overlap across all three splits.
3. **Exact SHA256 & Perceptual-Hash Duplicates:**
   * **Exact RGB SHA256 duplicates:** `0`.
   * **Exact Mask SHA256 duplicates:** `2` groups (one group of `6` identical all-zero empty masks spanning `train`, `val`, and `test`; one pair of identical non-empty masks `train/2022_05_22_pothole_15_horizantal_vertical_aug22.png` & `test/2022_05_22_pothole_15_vertical_aug23.png`).
   * **Exact Depth SHA256 duplicates:** `3` groups, including **2 cross-split `train`/`test` pairs** (`pothole_15_horizantal_vertical_aug22` vs `pothole_15_vertical_aug23`, and `pothole_17_horizantal_aug29` vs `pothole_17_aug28`).
   * **Cross-split RGB perceptual near-duplicates (`dHash <= 6` & `pHash <= 6`):** `10` pairs detected (several with Hamming distance `0`, e.g., `train/...pothole_17.png` vs `test/...pothole_17_aug28.png`). Flipped augmentations (`horizantal`/`vertical`) evade unflipped perceptual hashing and are additionally caught by base-ID grouping.

---

## 4. Licensing & Citation Requirements

* **`LICENSE` file:** MIT License (`Copyright (c) 2026 Youngjincha`).
* **`README.md` License Section (More Restrictive):** Explicitly permits use/modification **solely for non-commercial academic research**, prohibits selling, reselling, commercial use, or redistribution/reposting of the code or data (in whole or part), and mandates citing:
  > Ali, R., Saeed, Q. B., Buyukozturk, O., Lee, S., & Cha, Y. (2026). "Monocular Computer Vision-Based Simultaneous Pothole Segmentation and 3-D Volume Prediction Using 3DpredicNet." *IEEE Transactions on Industrial Informatics*, 1–12. DOI: `10.1109/TII.2026.3706348`.
* Because `README.md` contains non-commercial and anti-redistribution terms alongside the MIT `LICENSE` file, this project treats Pothole440 strictly as **academic research data** and excludes `/Pothole440-main/` from Git redistribution via `.gitignore`.
