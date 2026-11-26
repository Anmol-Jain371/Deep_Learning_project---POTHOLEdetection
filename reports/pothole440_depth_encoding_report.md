# Phase 3B: Pothole440 Depth Encoding & Physical Units Report

> **Dataset Inspected:** `./Pothole440-main/Pothole440-main/` (`train` + `val` depth distribution audit; `test` inspected only for dtype/shape/min-max integrity)  
> **Inventory Data:** [`dataset_inventory.json`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_inspection/dataset_inventory.json)  
> **Visualizations:** `12` multi-panel figures in [`runs/benchmark/pothole440_inspection/`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_inspection)

---

## 1. Observed Depth File Format & Native Pixel Properties

Reading all `440` depth maps via `cv2.imread(path, cv2.IMREAD_UNCHANGED)` and inspecting PNG headers with `PIL.Image` establishes:
* **Data Type & Channels:** `100%` of depth maps (`350` train, `50` val, `40` test) are **8-bit grayscale PNG (`uint8`, `1` channel, PIL mode `L`)** at `1200×800` resolution.
* **Embedded Metadata:** **`0` PNG metadata chunks** (`PIL.Image.info == {}`). No EXIF, scale tags, camera intrinsics, extrinsics, or Go!SCAN scanner calibration headers are present.
* **Observed Value Ranges Across Splits:**
  * `train`: `[0, 190]`
  * `val`: `[0, 183]`
  * `test`: `[0, 189]` (structural check only)

---

## 2. Physical Polarity & Cropped Scan-Patch Encoding

Statistical analysis of all `395` non-empty `train + val` masks plus visual inspection of 12 diverse multi-panel visualizations (`sample_01`–`sample_12`) reveals the exact structure of the depth maps:

1. **Polarity (Higher `uint8` Value = Deeper Cavity):**
   * Across **`395 / 395` (`100%`)** non-empty `train + val` samples, `inside_mean > outside_mean`.
   * Inside mask mean intensity: range **`[109.15, 175.75]`**, median **`146.04`**.
   * Outside mask mean intensity: range **`[2.00, 140.88]`**, median **`91.04`**.
   * Cross-section profiles (`sample_02`, `sample_04`, `sample_06`, `sample_07`, `sample_10`, `sample_12`) confirm smooth positive relief ridges peaking inside the crater floor (`~160–187`) relative to surrounding pavement inside the scanned region (`~80–140`). This matches **downward sensor-to-surface distance** (larger = deeper below camera), **opposite in sign to MiDaS inverse disparity** ($1/Z$, which decreases inside cavities).

2. **Cropped Go!SCAN Scan Patch with Sample-Dependent Constant Padding (Major Artifact):**
   * Visual inspection of the depth maps and histograms exposes a sharp rectangular/quadrilateral discontinuity in every inspected scan: only a **cropped patch around the pothole** contains real Go!SCAN 3D surface geometry.
   * Outside that scan patch, **`398 / 400` (`99.5%`)** of `train + val` depth images are padded with a **single dominant constant pixel value** covering a median **`79.8%`** (up to `100.0%`) of the entire frame.
   * Critically, this constant padding value is **not fixed at `0`** — it varies per sample from **`1` to `147` (median `86.0`)**.
   * In **`186` of `400`** `train + val` samples, the segmentation mask extends onto pixels equal to that constant fill value (visible as secondary background spikes inside the crater histogram in `sample_06` and `sample_12`).

3. **Supporting Evidence from Repository Model Code:**
   * Included architectures in `Pothole440-main/Pothole440-main/model/` (`3DpredicNet.py`, `yolov11.py`) terminate depth heads in **`torch.sigmoid()` / `nn.Sigmoid()`**, regressing normalized `[0, 1]` targets derived by dividing the `uint8` image intensities rather than predicting unnormalized physical millimetres directly.

---

## 3. Verification of Physical Units, Intrinsics & Dimensions

| Metadata / Quantity | Present in `./Pothole440-main/`? | Evidence |
|---|---|---|
| Physical depth unit (`mm` / `cm` per pixel unit) | **No** | No header, CSV, config, or README scaling formula |
| Per-scan normalization bounds (`min_mm`, `max_mm`) | **No** | `PIL.info == {}`; background fill varies `1–147` across samples |
| Camera intrinsics ($f_x, f_y, c_x, c_y$, sensor size) | **No** | Not included in dataset or code |
| Camera extrinsics ($H_{\text{cam}}$, pitch/tilt angle) | **No** | Not included in dataset or code |
| Go!SCAN 3D raw meshes / point clouds (`.ply`/`.obj`) | **No** | Only 8-bit PNG images provided |
| Physical surface area ($cm^2$) or volume ($cm^3$/L) GT | **No** | Absent from directory (`annotations/` also missing) |

---

## 4. Scientific Conclusion on Depth Encoding

The Pothole440 depth files are **uncalibrated, spatially cropped 8-bit quantized intensity projections (`uint8`)** of Go!SCAN depth surfaces padded outside the scan window by sample-varying constants (`1–147`).

**They do NOT encode verifiable metric cavity depth in cm/mm without external per-scan calibration metadata**, and any numeric factor mapping pixel intensity `[0–255]` to physical millimetres cannot be recovered or assumed from the dataset without risking fabricated metrics.
