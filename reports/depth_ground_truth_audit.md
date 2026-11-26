# Phase 3A: Independent Physical Depth Ground-Truth Audit

> **Audit Date:** 2026-11-26  
> **Audited Files:** `roadvision_engine.py`, `validate_physical_depth.py`, `depth_ground_truth_manifest.csv`, `runs/depth_validation_report.json`, `runs/depth_validation_results.csv`, `config/camera_params.json`, `analyze_camera_sensitivity.py`

---

## 1. Audited Metric Depth Estimation Architecture

RoadVision AI estimates cavity depth in metric centimetres through three coupled stages in [`roadvision_engine.py`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_engine.py):

1. **`AnnularDonutDepthExtractor`** (lines 190–262): Expands each bounding box by `mx = max(15, min(25, 0.25*w))`, `my = max(15, min(25, 0.25*h))`, masks out the crater interior, fits a 2D planar road disparity baseline $z(y, x) = a y + b x + c$ via `np.linalg.lstsq` (falling back to border median if `<20` rim pixels), and computes the 85th-percentile depression $\Delta d_{\text{raw}} = \max(0, P_{85}(\hat{z}_{\text{road}} - d_{\text{crater}}))$.
2. **`MetricDepthCalibrator`** (lines 101–184) + [`config/camera_params.json`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/config/camera_params.json): Maps row position $y_2$ and raw disparity delta to metric depth using pinhole ground-plane geometry (`sedan_dashcam` defaults: $H_{\text{cam}} = 1.25\text{ m}$, $\text{VFOV} = 60^\circ$, `horizon_ratio = 0.40`, `horizon_cutoff_ratio = 0.15`, $k_{\text{disp}} = 0.082$, $\gamma = 1.35$, clip $= 35.0\text{ cm}$):
   $$f_y = \frac{H / 2}{\tan(\text{VFOV}/2)}, \qquad Z_{\text{road}}(y_2) = \frac{H_{\text{cam}} \cdot f_y}{\max(15.0,\; y_2 - y_{\text{horizon}})}, \qquad \Delta Z_{\text{cm}} = \Delta d_{\text{raw}} \cdot Z_{\text{road}}^{\,1.35} \cdot 0.082$$
3. **`TemporalEMASmoother`** (lines 370–479): Maintains a 10-frame rolling history per ByteTrack ID, applies moving-median outlier suppression, blends with proximity-weighted EMA ($\alpha = 0.45$ if approaching hood else $0.20$), and classifies severity (`<4.0` MILD, `4.0–7.9` MODERATE, `>=8.0` SEVERE, plus aquaplaning flag if $>30\%$ frames trigger HSV glare/Laplacian smoothness).

---

## 2. Audit of `depth_ground_truth_manifest.csv` (`P01`–`P15`)

### 2.1 Inventory & Missing Field Metadata
* **Sample Count:** 15 entries (`P01`–`P15`), all referencing `Watch Video_ Terrible potholes in Rangoon Road.mp4` (`640×360 @ 25.0 FPS`, `1,005` frames) at monotonically increasing timestamps `0.8s` to `16.0s`.
* **Surface & Range Spread:** `13` dry (`NO`), `2` water-present (`P08` at `7.8 m`, `P15` at `13.5 m`); `Physical_Depth_cm` spans `2.8–11.0 cm`; `Estimated_Depth_cm` spans `3.2–11.8 cm`; `Approx_Camera_Dist_m` spans `3.2–13.5 m`.
* **Missing / Unverifiable Metadata:**
  * **Zero unique physical pothole group IDs or Track IDs:** Repeated observations of the same physical road cavity along the continuous `0.8s–16.0s` sequence cannot be distinguished from distinct cavities.
  * **No measurement method verification:** No straightedge/ruler photographs, field survey dates, or observer logs exist in the repository.
  * **No camera/detection provenance per row:** Missing `vehicle_profile`, `camera_height_m`, `vfov_deg`, `horizon_ratio`, `lighting`, `raw_disparity_delta`, and bounding-box pixel coordinate (`y2`).
  * **`validate_physical_depth.py` does not invoke the perception engine:** It compares the CSV's pre-filled static `Estimated_Depth_cm` column against `Physical_Depth_cm` without running MiDaS or `MetricDepthCalibrator`.

### 2.2 Provenance & Reproducibility Findings (Direct Forensic Checks)

1. **Synthetic Harness Overlap (`P01` & `P02`):**
   * Commit `d7dfd87` (`2026-10-10`) left `depth_ground_truth_manifest.csv` empty (`P01`–`P15` blank) and defined `generate_synthetic_test_data()` inside [`validate_physical_depth.py`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/validate_physical_depth.py#L404-L417) with synthetic cases `T01` (`dist=3.2`, `phys=3.0`, `est=3.4`) and `T02` (`dist=4.1`, `phys=5.5`, `est=5.1`).
   * Commit `1600610` (`2026-11-19`) populated `P01` (`3.2, 3.0, 3.4`) and `P02` (`4.1, 5.5, 5.1`) with **identical numeric triplets**.
2. **Non-Reproducibility by Live Pipeline at Manifest Timestamps (`0.8s–16.0s`):**
   * Executing `RoadVisionPipeline` (`MiDaS_small` + `AnnularDonutDepthExtractor` + `MetricDepthCalibrator` with `sedan_dashcam`) on `Watch Video_ Terrible potholes in Rangoon Road.mp4` at the exact frame indices for `0.8s–16.0s` fails to reproduce `Estimated_Depth_cm` and `Approx_Camera_Dist_m` using either `pothole_rdd1666_50e` or `pothole_combined_v2`:
     * At `13.2s` (`frame 330`, `P13`), **both detectors produce `0` detections** (`[]`), whereas the manifest records `Estimated_Depth_cm = 11.4 cm`.
     * At `6.0s` (`frame 150`, `P07`), `pothole_rdd1666_50e` produces `0` detections (`pothole_combined_v2` produces `35.0 cm` saturated at `Z_road = 25.98 m`), whereas the manifest records `7.7 cm` at `6.9 m`.
     * On detected frames (`e.g., P08, P10, P11, P12, P14, P15`), actual pipeline depth estimates range from `0.0 cm` to `34.84 cm` (`Z_road` frequently hitting `25.98 m` near `horizon_y = 144 px`), vs. the manifest's uniformly tight `±0.4–0.9 cm` residuals (`0.4`–`0.9 cm` on every single row).
3. **Inconsistency with `analyze_camera_sensitivity.py`:**
   * [`analyze_camera_sensitivity.py`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/analyze_camera_sensitivity.py#L22-L63) lists hardcoded `"raw_disp"` values for `P01–P15` (`0.42` to `1.12`) and evaluates `raw_disp * (z ** 1.35) * 0.082 * 1.48` (containing an undocumented `* 1.48` multiplier), which still evaluates to `0.245 cm` for `P01` rather than `3.4 cm`.
4. **Provenance of `k_disp = 0.082`, `depth_exponent = 1.35`, and `k_disp = 0.07664`:**
   * `k_disp = 0.082` and `depth_exponent = 1.35` were already hardcoded in `roadvision_engine.py` at commit `d7dfd87` (`2026-10-10`) *before* `depth_ground_truth_manifest.csv` was populated in commit `1600610` (`2026-11-19`); therefore, `0.082` and `1.35` were **not derived from `P01–P15`**.
   * Conversely, `suggested_k_disp = 0.07664` (`0.082 × 0.9346`) reported in `runs/depth_validation_report.json` **was directly fitted in-sample on all 15 rows (`P01–P15`)**, creating in-sample calibration leakage if reported without held-out evaluation.

### 2.3 Baseline Reproduction (`validate_physical_depth.py`)
Running `python validate_physical_depth.py` reproduces the prior baseline identically on the existing static manifest columns:
* **Samples:** `15` | **MAE:** `0.56 cm` | **RMSE:** `0.59 cm` (`0.5888`) | **MBE:** `+0.43 cm` | **MAPE:** `9.5%` (`9.55%`) | **Severity Concordance:** `100.0%` (`15/15`).

---

## 3. Research Integrity Conclusion for Depth Validation

Because sample independence cannot be verified (`independent_physical_identity_verified = False`), repeated potholes along a continuous video cannot be excluded, two entries match pre-existing synthetic test cases, and the static manifest values cannot be reproduced from the archived video frames via the current pipeline, **existing depth validation results must be classified as uncertified / provenance-incomplete rather than independent field proof**.

True certified validation requires collecting **30–50 genuinely independent, ruler-measured potholes** with paired raw video frames, bounding boxes, vehicle calibration specs (`camera_height_m`, `vfov_deg`, `horizon_ratio`), raw MiDaS disparity deltas, physical straightedge photographs, and grouped pothole IDs.

---

## 4. Separate Research Limitations (Unresolved Phase 2 Dataset & Annotation Issues)

Independently of Phase 3A depth auditing, the following unresolved Phase 2 dataset limitations remain documented and preserved:
1. **Cross-Split Perceptual Near-Duplicates in `dataset_unified`:** While exact SHA256 duplicates across `train`, `val`, `test`, and `external_rangoon` are `0`, perceptual hashing (`verify_and_freeze_dataset.py`) identified `337` cross-split image pairs with `dHash <= 6` or `pHash <= 6` (of which `14` pairs satisfy both `dHash <= 6` and `pHash <= 6`, and `16` pairs have `dHash <= 2` or `pHash <= 2`). Because RDD2022 India filenames omit dashcam trip/sequence IDs, visually adjacent road frames cannot be grouped out automatically.
2. **Incomplete & Domain-Shifted `external_rangoon` Annotations:** All `35` frames in `external_rangoon` originate from a single continuous flooded-road clip (`Watch Video_ Terrible potholes in Rangoon Road.mp4`, median box area `10.54%`, `12` boxes `>30%` area, `21` edge-touching boxes). Visual inspection ([`external_rangoon_audit_sheet.jpg`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_unified/manifests/external_rangoon_audit_sheet.jpg)) confirmed missed unannotated visible craters in at least `rangoon_v2_01_f0000.jpg`, `rangoon_v2_02_f0040.jpg`, and `rangoon_v2_05_f0120.jpg`.
3. **Absence of Untouched Final Test Set:** Both `dataset_unified/test` (`153` imgs) and `external_rangoon` (`35` imgs) were inspected during Phase 1/Phase 2 exploratory comparisons, and `evaluation/ground_truth_labels/` contains `0` annotated files.
