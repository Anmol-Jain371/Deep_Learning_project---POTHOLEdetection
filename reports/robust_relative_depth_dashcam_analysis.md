# Dashcam Development, Temporal Stability, Pothole440 Feasibility & RTX 2050 Benchmark Analysis (Phase 5A)

Executed via [benchmark_robust_relative_depth.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_robust_relative_depth.py); structured results saved to [robust_relative_depth_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_relative_depth_results.json) and [robust_relative_depth_per_region.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_relative_depth_per_region.csv).

> [!WARNING]
> **No Metric Depth Claims:** All values below compare relative inverse-depth residuals (`MiDaS_small`), within-track relative CV, synthetic perturbations, and stage latencies. Manual metric ground truth remains unavailable, and historical `depth_ground_truth_manifest.csv` rows are NOT used.

---

## 1. Comparative Benchmark on 36 Dashcam Development Frames (98 YOLOv8n Regions)

Using identical cached Phase 2 YOLOv8n (`yolov8n_pothole_v1`) boxes and `MiDaS_small` predictions across all 5 methods (`conf >= 0.25`, 36 sampled frames across `ruralRoad_potHoles.mp4` and `raw_survey.mp4`):

| Method | Valid Regions | Invalid Regions | Top Failure Reasons (`failure_reasons`) | Ref Fit MAD (Median) | Inlier Ratio (Median) | Valid Cavity P75 (Median / IQR) | Mean CPU Region Time (ms) | 4+ Boxes Frame Time (ms / FPS) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `median_baseline` | `13/98` (`13.27%`) | `85` | `unexpected_residual_polarity`: 85, `water`: 3 | `56.9555` | `1.000` | `68.2604` / `74.5707` | `3.18` | `13.54 ms` / `73.85` |
| `existing_annular` | `39/98` (`39.80%`) | `59` | `unexpected_residual_polarity`: 57, `water`: 3 | `2.9218` | `1.000` | `6.1245` / `5.8994` | `4.12` | `18.46 ms` / `54.16` |
| `ols_plane` | `37/98` (`37.76%`) | `61` | `unexpected_residual_polarity`: 58, `water`: 3, `unstable`: 1 | `3.9618` | `0.9904` | `8.4879` / `11.3247` | `4.80` | `21.85 ms` / `45.77` |
| **`ransac_plane`** | **`49/98` (`50.00%`)** | **`49`** | `unexpected_residual_polarity`: 29, `low_inlier`: 22, `water`: 3, `unstable`: 3 | **`2.5754`** | `0.5527` | `7.9628` / `9.8988` | `10.00` | `49.39 ms` / `20.25` |
| `huber_irls_plane` | `35/98` (`35.71%`) | `63` | `unexpected_residual_polarity`: 60, `water`: 3, `unstable`: 1 | `3.6205` | `0.9849` | `8.4788` / `9.6610` | `22.03` | `96.99 ms` / `10.31` |

**Interpretation:**
- `median_baseline` collapses on tilted dashcam road planes (`56.96` median ref MAD, `85/98` polarity rejections).
- `ransac_plane` achieves the **highest valid region rate (`50.00%`) and lowest median reference MAD (`2.5754`)**, cutting `unexpected_residual_polarity` rejections nearly in half (`29` vs `58–60`) by excluding adjacent grass shoulders, mud ridges, and traffic clutter (`median inlier_ratio = 0.5527`), while gating off severely non-planar rings via `low_inlier_ratio` (`22`).
- `huber_irls_plane` is the slowest CPU method (`22.03 ms/region`, `96.99 ms` on dense frames) due to iterative weighted least-squares solves over large annular rings.

---

## 2. Sensitivity & Contamination Robustness on Real Dashcam Frames

| Stress Test | Metric | `ols_plane` | `ransac_plane` | `huber_irls_plane` |
| :--- | :--- | :---: | :---: | :---: |
| **Margin `0.25` vs `0.45`** | Valid count / Median P75 shift | `40` / `12.55%` | `53` / `13.71%` | `39` / `12.04%` |
| **Margin `0.70` vs `0.45`** | Valid count / Median P75 shift | `34` / `14.40%` | `46` / `14.75%` | `33` / `14.44%` |
| **±8% Box Perturbation** | Retained valid / Median P75 shift | **`37/37` / `5.40%`** | `45/49` / `11.21%` | `35/35` / `5.74%` |
| **Injected Curb Step (`+30..80`)** | Retained valid / Median Rel P75 Err | `37/37` / **`4,734.81%`** | **`25/49` / `17.15%`** | `35/35` / **`5,439.91%`** |

**Trade-off Exposed:** Under clean pavement jitter (`±8%` box shifts), global `ols_plane` (`5.40%`) and `huber_irls_plane` (`5.74%`) shift less than `ransac_plane` (`11.21%`) because RANSAC samples a discrete consensus subset of slightly non-planar MiDaS road surfaces. Conversely, under high-leverage border/curb step contamination (`+30..80` injected along a ring band), `ols_plane` (`4,734.81%`) and `huber_irls_plane` (`5,439.91%`) fail catastrophically, whereas `ransac_plane` holds median P75 error to **`17.15%`** (and rejects badly corrupted frames cleanly via `low_inlier_ratio`).

---

## 3. Within-Track Temporal Relative-Depth Stability (`ruralRoad_potHoles.mp4`, Frames 0–119)

> [!CAUTION]
> Lower within-track Coefficient of Variation (CV) measures **frame-to-frame consistency of relative P75 units only — NOT physical metric accuracy**.

| Method | Unique Tracks Seen (`YOLOv8n` + ByteTrack) | Valid Observations | Tracks with `>=4` Valid Obs | Mean Within-Track CV | Median Within-Track CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `median_baseline` | `22` | `27` | `2` | `0.1951` | `0.1951` |
| `existing_annular` | `22` | `131` | `5` | `0.4628` | `0.4545` |
| **`ols_plane`** | `22` | `126` | `5` | **`0.3542`** | **`0.2998`** |
| `ransac_plane` | `22` | **`190`** | **`14`** | `0.5912` | `0.5598` |
| `huber_irls_plane` | `22` | `126` | `5` | `0.3639` | `0.3550` |

Notice the track-selection difference: `ols_plane` and `huber_irls_plane` retain `5` stronger tracks (`mean CV 0.3542–0.3639`, beating legacy `0.4628`), whereas `ransac_plane` admits `14` tracks (`190` valid obs, including distant/marginal potholes whose relative inverse depth naturally grows non-linearly as the vehicle approaches!). Un-normalized MiDaS inverse depth varies strongly with vehicle approach distance, so track-level temporal stability requires per-frame scale stabilization or disparity-distance normalization in future work.

---

## 4. Pothole440 Permitted Validation Set Feasibility Check

Evaluated **exclusively** on the 15 unaugmented originals in `dataset_pothole440_seg_grouped/val` (`test` split NEVER accessed):

- **Samples with `>25%` of Adaptive Annular Ring Overlapping Constant Padding:** **`11 / 15`** (up to **`89.12%`** on `2022_06_03_pothole_23.png` and **`80.70%`** on `2022_06_03_pothole_1.png`).
- **Interior Matching Dominant Value:** Border-connected flood fill cleanly separated outer constant padding without deleting genuine interior scan pixels (e.g., `2022_05_25_pothole_3.png`: `3.24%` valid interior pixels shared `dom_pad_val = 126`), proving border-fill isolation is safe — **but cannot recover pavement cropped outside the handheld Go!SCAN frame**.
- **Ordinal Agreement (`Go!SCAN cavity > ring` vs `MiDaS cavity < ring`):** **`8 / 15`**.
- **Mean / Median Spearman $\rho$ (`MiDaS` vs negated `uint8 Go!SCAN` inside valid scan):** **`-0.0401`** / **`-0.0194`** (range `-0.4709` to `+0.4616`).
- **Status:** **`BLOCKED`** — Pothole440 `uint8` maps cannot serve as physical or residual ground truth for dashcam relative depth due to missing per-scan physical scaling, missing intrinsics, severe off-road scan crops, and near-zero pixelwise correlation with `MiDaS_small`.

---

## 5. RTX 2050 End-to-End Performance & Latency Budget

Measured on NVIDIA RTX 2050 (~4 GB VRAM) + `640×360` frames:
- **Separate `MiDaS_small` cold load (startup):** `5717.62 ms`
- **YOLOv8n (`imgsz=640`):** **`13.68 ms`** mean (`12.05 ms` median, `20.52 ms` p95 -> **`73.11 FPS`**)
- **`MiDaS_small` active frames:** **`28.03 ms`** mean (`26.47 ms` median, `40.54 ms` p95 -> **`35.67 FPS`**)
- **`MiDaS_small` amortized over all 36 frames:** **`24.92 ms`** (`40.13 FPS`)
- **Peak GPU Memory:** **`144.26 MB`** allocated / **`164.00 MB`** reserved (very light VRAM footprint alongside YOLOv8n).
- **End-to-End Projected Pipeline Latency (YOLO `13.68 ms` + MiDaS `28.03 ms` + CPU Ref Extraction on 4+ box frame):**
  - With `ols_plane` (`21.85 ms`): **`~63.6 ms/frame` (`~15.7 FPS`)**
  - With `ransac_plane` (`49.39 ms`): **`~91.1 ms/frame` (`~11.0 FPS`)**
  - Note: Subsampling `ransac_sample_limit` or running `ransac_plane` on downsampled 2x coordinates would halve CPU runtime while retaining outlier immunity.

---

## 6. Visual Error Analysis from 9-Panel Diagnostic Plots (`robust_depth_viz/*.png`)

Direct inspection of the 8 rendered `ransac_plane` 9-panel figures highlights four clear real-world behaviors:
1. **Clean Planar Rejection of Adjacent Clutter ([dev_rural_f0015_r0_valid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_depth_viz/dev_rural_f0015_r0_valid.png), [dev_rural_f0030_r1_valid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_depth_viz/dev_rural_f0030_r1_valid.png)):** RANSAC masks out left-side grassy/berm shoulders (`Inlier: 53.5%–58.5%`) in red, fits the sloped mud track in green (`Ref MAD: 1.36–3.35`), and isolates crisp positive cavity residuals (`P75: 6.23–12.19`, `PosFrac: 0.617–0.809`).
2. **Neighbor Box Exclusion in Dense Clusters ([dev_rural_f0060_r1_valid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_depth_viz/dev_rural_f0060_r1_valid.png)):** Orange neighbor box (`1050 px` removed) cleanly notches the lower-left annular ring (`n=5950`, `Inlier: 75.1%`, `P75: 13.98`), triggering `neighbors_excluded_from_ring` cleanly without contaminating the plane.
3. **Specular Puddle Polarity & Water Gate ([dev_rural_f0030_r0_invalid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_depth_viz/dev_rural_f0030_r0_invalid.png), [dev_survey_f0060_r1_invalid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_depth_viz/dev_survey_f0060_r1_invalid.png), [dev_survey_f0120_r0_invalid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_depth_viz/dev_survey_f0120_r0_invalid.png)):** Mirror-like sky reflections inside puddles trick MiDaS into predicting raised surfaces (`PosFrac: 0.029`, `unexpected_residual_polarity`) or trigger `SpecularAquaplaneDetector` (`suspected_water_reflection`), proving that explicit `is_valid=False` rejection prevents spurious alerts.
4. **Wide Edge-Touching / Dry-Shadow Cavities ([dev_rural_f0000_r0_valid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_depth_viz/dev_rural_f0000_r0_valid.png), [dev_rural_f0030_r2_invalid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/robust_depth_viz/dev_rural_f0030_r2_invalid.png)):** In `dev_rural_f0000_r0`, a wide bottom-truncated YOLO box encompassing dark grass shoulder shows `P75: 169.5` due to perspective change across a large tall box (`boundary_truncated:bottom`), while `dev_rural_f0030_r2` (foreground dark dry patch) exhibits slight inverted residual polarity (`PosFrac: 0.084`), reflecting genuine monocular MiDaS failure on dark shallow textures.

---

## 7. Final Phase 5A Recommendation & Stop Gate

- **Recommended Reference Strategy:** Adopt **Adaptive Neighbor-Excluded Annular Sampling + `ransac_plane` (MSAC with neighbor finite-difference `local_sigma`) + 11-Point Quality Gating** as the primary experimental relative-depth extractor wherever robustness to road-shoulder vegetation, curbs, and adjacent potholes is required (`12/12` synthetic accuracy, `0.0003` mean normalized RMSE, `50.0%` valid dashcam coverage, `17.15%` curb-step error vs `4,734%+` for OLS/Huber).
- **Important Practical Caveat:** For smooth, latency-critical real-time streams (`~15.7 FPS` vs `~11.0 FPS`), `ols_plane` offers lower frame-to-frame box jitter (`5.40%` vs `11.21%`) and lower clean-track CV (`0.354` vs `0.591`), so future optimization should downsample RANSAC candidate coordinates (2–4× stride) and consider distance-normalized temporal smoothing **for valid observations only**.
- **Production Integration Decision:** **Do NOT modify `roadvision_engine.py` or `server.py` yet.** Phase 5A deliverables are complete and ready for review; physical metric calibration remains suspended until controlled field measurements become available.
