# RoadVision AI — Phase 7B: Real-Video Exploratory Horizon Evaluation & Final Decision

> [!WARNING]
> **Exploratory Status Only:** None of the five local footage sources have physical horizon labels, pitch extrinsics, camera intrinsic matrices `K`, or surveyed road plane truth. Every measurement in this report is an **exploratory proxy evaluation** of estimator stability, reliability gating, failure behavior, and visual alignment ([phase7b_horizon_results.json](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_horizon_results.json), [phase7b_per_frame_horizon.csv](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_per_frame_horizon.csv)).

---

## 1. Evaluated Local Footage Corpus (657 Frames Total)

| Source ID | Frames & Rate | Resolution | Provenance & Visual Character |
|---|---:|---|---|
| `ruralRoad_202f` | `202` @ `25 FPS` | `640×360` | Unpaved rural road with grass verges and puddle-filled potholes (true visual vanishing apex sits high in frame ~`0.05–0.15`, above `0.40`). |
| `rangoonRoad_f200_399` | `200` @ `25 FPS` | `640×360` | Shadowed congested Indian urban corridor (`f200-319` pre-examined + `f320-399`), weak/occluded lane markings, roadside clutter. |
| `rawSurvey_160f_10fps` | `160` @ `10 FPS` | `1280×720` | Multi-clip stock montage mixing normal dashcam shots (`F0016`), close-up puddles (`F0056`), steep wheel-level angles (`F0104`), and no-sky shots (`F0144`). Native `10 FPS` tests timestamp expiry. |
| `phonePotholes_70f` | `70` (`25 FPS` assm.) | `1280×720` | Handheld sequential phone frames (`frame_0000..0069.jpg`), strong pitch/perspective variation. |
| `testClip_25f` | `25` @ `25 FPS` | `640×360` | Short 25-frame sanity clip. |

---

## 2. Detector Mode Comparison on Real Footage

| Source | `lsd_320` VALID / LowC / Unav (%) | `hough_320` VALID (%) | `hybrid_320` VALID (%) | `lsd_fullres` VALID (%) | `lsd_320` Mean `horizon_y` & Mean ms |
|---|---:|---:|---:|---:|---:|
| `ruralRoad_202f` | **`91.09%`** / `1.49%` / `7.43%` | `70.30%` | `83.66%` | `73.76%` | `0.177` (`7.53 ms`) |
| `rangoonRoad_f200_399` | `34.00%` / `33.50%` / **`32.50%`** | `31.50%` | `51.50%` | **`2.00%`** | `0.285` (`7.28 ms`) |
| `rawSurvey_160f_10fps` | `65.00%` / `35.00%` / `0.00%` | `55.63%` | `71.88%` | `12.50%` | `0.342` (`14.80 ms`) |
| `phonePotholes_70f` | **`91.43%`** / `8.57%` / `0.00%` | `25.71%` | `91.43%` | **`0.00%`** | `0.271` (`9.63 ms`) |
| `testClip_25f` | `64.00%` / `36.00%` / `0.00%` | `32.00%` | `100.00%` | `0.00%` | `0.231` (`7.04 ms`) |

**Key Real-World Findings:**
- **`lsd_fullres` collapses on noisy 720p roads (`0.0%`–`12.5%` VALID):** At full `720p`, LSD fractures coarse asphalt aggregate into hundreds of tiny short edges that fail `min_line_len_ratio` (`62.68 ms/frame` mean cost for worse reliability!). Downscaling to **`320px` acts as natural structural smoothing**, boosting both speed (`~6×`) and `VALID` rates (`91.43%` vs `0.00%` on `phonePotholes_70f`).
- **`rangoonRoad_f200_399` exposes urban failure mode (`34.0% VALID`):** Heavy shadows, uneven shopfronts/curbs, vehicles, and absent lane lines trigger **`66` `low_inlier_count`**, **`35` `implausible_vp_position`**, and **`27` `insufficient_inlier_support`** flags.

---

## 3. Temporal Policy Comparison (`lsd_320`)

Observed trajectories are plotted in [real_video_horizon_trajectories.png](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_viz/real_video_horizon_trajectories.png):

| Policy | Metric | `ruralRoad_202f` | `rangoonRoad_f200_399` | `rawSurvey_160f_10fps` | `phonePotholes_70f` |
|---|---|---:|---:|---:|---:|
| **`A_fixed`** | Availability / Jitter | `100%` / `0.000` | `100%` / `0.000` | `100%` / `0.000` | `100%` / `0.000` |
| **`B_dynamic_raw`** | Availability / Mean Jitter | `92.57%` / `0.068` | `67.50%` / `0.080` | `100.0%` / `0.017` | `100.0%` / `0.095` |
| **`C_dynamic_smoothed`** | Availability / Mean Jitter | `100.0%` / `0.014` | `73.50%` / `0.009` | `81.87%` / `0.007` | `100.0%` / `0.021` |
| **`D_dynamic_fallback`** | Availability / Fallback % / Jitter | **`100%`** / `0.0%` / `0.014` | **`100%`** / **`26.5%`** / `0.008` | **`100%`** / **`18.12%`** / `0.009` | **`100%`** / `0.0%` / `0.021` |

Smoothing (`C` / `D`) cuts frame-to-frame raw horizon jitter by **`5×–9×`** (`0.068→0.014`, `0.080→0.008`), while `D_dynamic_fallback` seamlessly switches to `0.40` on `52` expired urban frames in `rangoonRoad` (`26.5%`) and `9` expired frames (`+20 LOW_CONFIDENCE`) in `rawSurvey` (`18.12%`).

---

## 4. Visual Inspection Summary (Comparative Demo Previews)

Inspecting the generated comparative overlay previews ([ruralRoad grid](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_viz/phase7b_preview_grid_ruralRoad_202f.png), [rawSurvey grid](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_viz/phase7b_preview_grid_rawSurvey_160f.png)) and demo videos (`phase7b_horizon_demo_*.mp4`):
- **True Improvement Case (`ruralRoad_202f` `F0020`):** `A_fixed=0.40` (white dashed line) bisects mid-road puddles far below the visible vanishing apex, whereas `D_dynamic_fallback` (`VALID`, `0.134`, green line) rises cleanly near the true horizon!
- **Distractor & Failure Cases (`ruralRoad` `F0070`/`F0181`, `rawSurvey` `F0104`/`F0144`):** Linear puddle reflections and close-up pothole cracks align opportunistically, occasionally pulling the VP crosshair sideways (`STALE`/`VALID` around `0.10–0.47` without semantic road awareness).

---

## 5. Final Outcome Recommendation: **Outcome B**

Based objectively on all verified Phase 7B evidence, **Outcome B** is selected:

> **Outcome B — Dynamic horizon estimation works reliably only in structured perspective or lane-marked scenes and MUST remain research-only until stronger road-plane cues or true camera calibration are available.**

**Justification:**
1. While `lsd_320` achieves **`90.48% VALID` at `0.003` error** on clean synthetic perspective (`0/4` false `VALID`s) and tracks true upper road geometry on structured rural/handheld clips (`~91% VALID`), **valid coverage drops to `34.00%`** in real Indian unmarked/shadowed urban traffic (`rangoonRoad_f200_399`).
2. Water puddle rims and pavement cracks in steep close-ups (`rawSurvey` `F0144`) can produce spurious line intersections that mimic road convergence without semantic road-plane segmentation.
3. Crucially, as proven in [phase7b_geometry_sensitivity.md](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/phase7b_geometry_sensitivity.md), substituting `D_dynamic_fallback` into `MetricDepthCalibrator` **does not stabilize metric scaling or fix approach-distance drift** (`within-track CV 0.396 → 0.404`, `|Spearman rho|` still `0.473`) and introduces amplification spikes near small `dy`.

*Prerequisites to revisit toward integration:* (1) calibrated camera intrinsics `K` & mounting extrinsics (or road-surface segmentation masking out non-road clutter/puddles), (2) horizon-relative `dy` clamp defined as a normalized ratio rather than `15.0` raw pixels, and (3) true physical pothole metric ground truth.
