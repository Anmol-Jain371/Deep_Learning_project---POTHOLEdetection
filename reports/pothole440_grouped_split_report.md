# Phase 4A.1: Pothole440 Grouped Split & Leakage Control Report

> **Builder Script:** [`experiments/build_pothole440_seg_grouped.py`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/build_pothole440_seg_grouped.py)  
> **Derived Dataset:** `./dataset_pothole440_seg_grouped/` (`data.yaml` portable relative path)  
> **Frozen Manifest:** [`dataset_manifest.json`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_pothole440_seg_grouped/manifests/dataset_manifest.json) (`SHA256: cff0e21e8341a96a7c95c114aa1ea1509874c1047dfae1a9b2ea85c1fe972a16`)  
> **Per-Image Audit CSV:** [`split_manifest.csv`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_pothole440_seg_grouped/manifests/split_manifest.csv)

---

## 1. Base Capture Grouping & Physical Identity Findings

* **Regex Base-ID Extraction:** Every filename in `./Pothole440-main/Pothole440-main/{train,val,test}/rgb/` follows `^(\d{4}_\d{2}_\d{2}_pothole_\d+)(.*)$`. Stripping `_horizantal`, `_vertical`, `_horizantal_vertical`, and `_aug*` groups all `440` source triplets into **`98` unique base capture IDs** across `7` capture dates (`2022_05_22` to `2022_06_04`).
* **Special Group Without Original:** Exactly `1` base group (`2022_06_03_pothole_39`) has no unaugmented filename (only `2022_06_03_pothole_39_horizantal_vertical.png`); it was pinned to `train`.
* **Cross-Group Physical Similarity Check:** Orientation-invariant perceptual hashing (minimum Hamming distance across `{orig, hflip, vflip, hvflip}` with `dHash <= 8` and `pHash <= 8`) found **`0`** near-identical pairs across different base capture IDs.
* **Caveat on Physical Independence:** Adjacent capture IDs recorded on the same day (`e.g., 2022_06_03_pothole_1..39`) could still include nearby or different-angle captures of the same road segment that evade global perceptual hashing; therefore, **these `98` base groups are reported as capture groups rather than proven independent physical potholes**.

---

## 2. Empty-Mask Inspection & Quarantine

Visual inspection of all `6` empty-mask images ([`empty_masks_audit.jpg`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_seg_preparation/empty_masks_audit.jpg)) confirmed that **all `6` are zoomed/rotated augmentation crops (`_aug*`)** (`2022_05_22_pothole_11_horizantal_aug5.png`, `pothole_15_aug20.png`, `pothole_21_horizantal_vertical_aug46.png`, `pothole_12_vertical_aug11.png`, `pothole_19_vertical_aug39.png`, `pothole_11_vertical_aug7.png`), several exhibiting black rotation padding corners or cropped crack/edge textures rather than verified clean-road scenes. All `6` were **quarantined** (`QUARANTINED_EMPTY_MASK`) rather than injected as negative training samples.

---

## 3. Capture-Date Holdout Feasibility vs. Chosen Split Strategy

| Capture Date | Base Groups | With Unaugmented Original | Total Source Images | Assigned Train / Val / Test Base Groups |
|---|---:|---:|---:|---|
| `2022_05_22` | 16 | 16 | 118 | `11` / `2` / `3` |
| `2022_05_23` | 4 | 4 | 16 | `2` / `1` / `1` |
| `2022_05_25` | 7 | 7 | 28 | `5` / `1` / `1` |
| `2022_06_01` | 23 | 23 | 92 | `16` / `4` / `3` |
| `2022_06_02` | 6 | 6 | 24 | `4` / `1` / `1` |
| `2022_06_03` | 35 | 34 | 137 | `25` / `5` / `5` |
| `2022_06_04` | 7 | 7 | 25 | `5` / `1` / `1` |
| **Total** | **98** | **97** | **440** | **`68` (69.4%) / `15` (15.3%) / `15` (15.3%)** |

* **Why Pure Capture-Date Holdout Was Not Chosen:** Capture dates are heavily skewed (`4` to `35` base groups per date, and `2022_05_22` holds almost all heavy crop/rotate `_aug*` variants). Holding out whole dates would concentrate lighting/pavement appearance bias exclusively in val or test.
* **Chosen Strategy (`seed=42`, Stratified Base-Group Split):** Partitioned the `97` original-containing base groups proportionally across all `7` dates into **`15` val / `15` test / `67` train** (`+ 1` forced group = **`68` train**). Validation and test retain **only the unaugmented original image** per group (`15` + `15`), while all **`103` val/test augmented variants** (`51` val, `52` test) were strictly **excluded**.
* **Benchmark Naming:** Because this test set was reconstructed by regrouping the contaminated official Pothole440 splits, all evaluations on `dataset_pothole440_seg_grouped/images/test` must be reported explicitly as a **reconstructed grouped benchmark**.

---

## 4. Final Dataset Inventory & Leakage Verification

| Split | Base Groups | Total Images | Originals | Augmented | Excluded Augs | Quarantined Empty | Polygons |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 68 | 301 | 67 | 234 | 0 | 3 | 321 |
| **Val** | 15 | 15 | 15 | 0 | 51 | (2 in excl.) | 15 |
| **Test** | 15 | 15 | 15 | 0 | 52 | (1 in excl.) | 15 |
| **Total** | **98** | **331** | **97** | **234** | **103** | **6** | **351** |

* **Cross-split base-group overlap:** `0` (`train_val=[]`, `train_test=[]`, `val_test=[]`).
* **Cross-split exact RGB SHA256 overlap:** `0`.
* **Cross-split perceptual matches (`dHash <= 6` & `pHash <= 6`):** `0`.
* **Remaining Potential Leakage / Limitations:** Capture sessions (`7` dates) intentionally span all splits due to stratification; unflipped visually similar street scenes with Hamming distance `> 6` remain theoretically possible across same-day captures; `val` and `test` have small sample sizes (`15` unaugmented originals each), increasing metric variance compared to large benchmarks.
* **Git Exclusion:** `/Pothole440-main/`, `dataset_pothole440_seg_grouped/images/`, and `dataset_pothole440_seg_grouped/labels/` are excluded in [`.gitignore`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/.gitignore); manifests (`split_manifest.csv`, `dataset_manifest.json`, `.sha256`), `data.yaml`, and `README.md` are lightweight metadata.
* **Ultralytics Compatibility:** `ultralytics.data.utils.check_det_dataset('dataset_pothole440_seg_grouped/data.yaml')` passed (`nc=1`).

---

## 5. Recommended Segmentation Training Configuration (RTX 2050 4 GB VRAM)

* **Primary / Control Models:** Start with `yolov8n-seg.pt` (`~3.26M` params) and compare against `yolov8s-seg.pt` (`~11.8M` params) if `batch=4` fits comfortably inside 4 GB VRAM with segmentation proto-head memory.
* **Recommended Settings:** `data=dataset_pothole440_seg_grouped/data.yaml`, `imgsz=640`, `batch=4` (probe peak VRAM before testing `batch=8`, keep CUDA OOM fallback), `optimizer='SGD'`, `lr0=0.01`, `cos_lr=True`, `epochs=100`, `patience=30`, `close_mosaic=10`, `amp=True`, `workers=2`, `seed=42`, `device=0`.
* *(Note: Because `val` contains `15` single-instance images, monitor `mAP50-95(M)` and loss stability carefully, and consider 5-fold grouped CV over the `98` base groups if high checkpoint-selection variance is observed.)*
