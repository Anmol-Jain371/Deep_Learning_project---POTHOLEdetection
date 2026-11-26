# Phase 5C Shadow-Mode Integration & Multi-Video A/B Report

Experimental orchestrator: [experiments/shadow_depth_pipeline.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/shadow_depth_pipeline.py) | Benchmark harness: [experiments/benchmark_shadow_depth_pipeline.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_shadow_depth_pipeline.py) | Results: [shadow_depth_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/shadow_depth_results.json), [shadow_depth_per_frame.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/shadow_depth_per_frame.csv), [shadow_depth_per_track.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/shadow_depth_per_track.csv).

---

## 1. Phase 5B Artifact Verification, Production Hashes & Denominator Clarification (Task 1)

- **Automated Test Verification:** All 15 prior Phase 5A/5B tests (`test_optimized_relative_depth.py`, `test_validity_aware_temporal_filter.py`, `test_robust_relative_depth.py`) plus all 6 new Phase 5C tests (`tests/test_shadow_depth_pipeline.py`) pass (**21/21 `OK`**).
- **Production & Experimental File SHA256 Hashes Verified Before & After:**
  - `roadvision_engine.py`: `3805808750134A38B3FA9BCFC42FB6583A843DDA575E70AEF28FF205C0E2A51E`
  - `server.py`: `D8772504E03F949E1B92CF8C76DA65327035F2B1A5AE713385D5EB60FDE91EFF`
  - `config/camera_params.json`: `0C7633F330A8B5C15A1696B85D81E9CA6D32E3CAC1C1C237CFFF8E576E789AF1`
  - `runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt`: `DDD6A5BE27AC21F3EF17FFE17FBAC5EA901CF2A7F7706956C5E551A669490343`
  - `experiments/robust_relative_depth.py`: `41683E8FC78EBB03787570111EEFA4AEA45128B0E2850849D48870C069770E47`
  - `experiments/optimized_relative_depth.py`: `48D36735AD80A1CEE584FF40E8BC255FF99D5AEABA4536FB29B97ABA9A9F58FF`
  - `experiments/validity_aware_temporal_filter.py`: `8DA5FFBE1F282DC2610798BB6E4DC25C6410F80968156F330F2613A165384FC6`
- **Denominator Clarification Note (Phase 5B `ols_plane` `37/37` vs `37/98`):** In [optimized_relative_depth_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_relative_depth_results.json) and [matched_track_depth_stability.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/matched_track_depth_stability.md), the **total Phase 5A dashcam candidate denominator is `98` regions**, of which `ols_plane` achieved **`37 / 98` (`37.76%`)** valid coverage (`valid_count = 37`), while **`37 / 37`** refers solely to `ols_plane`'s **`box_perturbation_8pct` and `curb_contamination` valid-retention subset** (retaining all 37 of its own base-valid detections under perturbation/step stress). Any prior narrative phrasing showing `37/37` under admitted coverage was a typographical conflation of the retention ratio (`37/37`) with the full-corpus coverage ratio (**`37/98`**).

---

## 2. Measurement Contract & Shadow Architecture (Tasks 2 & 4)

`ShadowDepthPipeline` executes **YOLOv8n + ByteTrack** and **`MiDaS_small`** **once per frame**, evaluates specular water flags once per box, and passes identical shared inputs to:
- **Method A (`A_legacy_annular_ema`):** Unmodified `AnnularDonutDepthExtractor` + `MetricDepthCalibrator` + `TemporalEMASmoother` (`25 px` margin, silent `0.0` fallback).
- **Method B (`B_opt_ransac_valid_ema`):** `extract_optimized_relative_depth_for_box` (`stratified_quadrant`, `512`) + `ValidityAwareTrackFilter("validity_aware_ema")`.
- **Method C (`C_opt_ransac_rollmed_ema`):** Same optimized RANSAC extraction + `ValidityAwareTrackFilter("rolling_median_validity_ema", window=5, alpha=0.35, max_stale=3, reset=8)`.
- **Method D (`D_opt_ransac_adaptive_1024`):** Optional diagnostic escalation (`512 -> 1024`) when `inlier_ratio < 0.65` or `populated_quadrants < 3` with `available_ref_samples > 1024`.

Every detection returns a strict `ShadowMeasurementResult` supporting explicit states (`VALID`, `INVALID`, `STALE`, `EXPIRED`, `NOT_EVALUATED`), keeping unavailable B/C/D metrics strictly `None`/`null` and catching any experimental exception cleanly without disturbing Method A.

---

## 3. Contiguous Multi-Video Evaluation (Tasks 5 & 9 — 567 Frames, 1,178 Detections)

| Sequence | Res @ FPS | Frames / Dets | Prior Exposure | `A_legacy` Valid | `B/C_opt` Valid / Stale / Exp / Inv | Shadow Wall FPS |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `ruralRoad_202f` | `640×360 @ 25` | `202` / `690` | Unseen in detector/seg train | `542` (`78.6%`) | `441 / 141 / 43 / 65` (`63.9%` valid) | `16.93` |
| `rangoonRoad_180f` | `640×360 @ 25` | `180` / `214` | `6/180` in `dataset_v2_candidates`; `174` unseen | `187` (`87.4%`) | `47 / 50 / 110 / 7` (`22.0%` valid) | `19.74` |
| `rawSurvey_720p_160f` | `1280×720 @ 10` | `160` / `249` | `0` MD5 matches in train; 720p urban | `192` (`77.1%`) | `19 / 0 / 0 / 230` (**`7.6%` valid**) | `14.76` |
| `testClip_25f` | `640×360 @ 25` | `25` / `25` | Development upload clip | `25` (`100%`) | `3 / 9 / 13 / 0` (`12.0%` valid) | `19.88` |

### Aggregate A/B/C/D Comparison (`n = 1,178` Detections across `57` Tracks)

| Metric | `A_legacy_annular_ema` | `B_opt_ransac_valid_ema` | `C_opt_ransac_rollmed_ema` | `D_opt_ransac_adaptive_1024` |
| :--- | :---: | :---: | :---: | :---: |
| Status Split (`VAL / STL / EXP / INV %`) | `80.31 / 0.00 / 0.00 / 19.69` | `43.29 / 16.98 / 14.09 / 25.64` | `43.29 / 16.98 / 14.09 / 25.64` | `43.46 / 15.87 / 15.70 / 24.96` |
| Admitted P75 Median / IQR | `4.1562` / `7.9514` | `8.1345` / `11.2478` | `8.1345` / `11.2478` | `8.3806` / `11.5442` |
| **Matched Common-ABC (`n=442`) P75 Med / Ref MAD Med** | `6.3020` / `N/A` | **`8.2332` / `2.6068`** | **`8.2332` / `2.6068`** | `8.1951` / `2.6101` |
| Mean Track Med `\|Δ\|` (Raw -> Smoothed) | `2.5805` -> `0.6655` | `11.8653` -> `2.0981` | `11.8653` -> **`1.1011`** | `7.9431` -> `1.5398` |
| Mean Track Smooth CV / MAD | `0.4341` / `1.3397` | `0.3863` / `7.8607` | `0.3423` / `2.9764` | **`0.2595`** / `3.4270` |
| Approach Spearman `ρ` | `0.2123` | `0.3441` | `0.3441` | `0.3559` |
| Extraction Latency Mean / P95 (ms) | `1.07` / `2.84` | `3.49` / `8.67` | `3.49` / `8.67` | `5.60` / `16.90` |

**Key Comparative Insights:**
- Notice Method `C` reduces smoothed track CV (`0.3423`) and MAD (`2.9764`) strongly below Method `B` (`0.3863` CV, `7.8607` MAD) and cuts frame-to-frame delta `~10.8×` (`11.865 -> 1.101`) without fabricating `0.0` updates during invalid frames.
- **Crucial Validity Bottleneck Exposed:** While Method `A` emits `>0` on `80.3%` of detections, experimental gating (`B/C`) admits `63.9%` on rural potholes (`ruralRoad_202f`) but drops sharply to **`22.0%` on `rangoonRoad_180f`** and **`7.6%` (`19/249`) on `rawSurvey_720p_160f`**, driven overwhelmingly by **`unexpected_residual_polarity` (`511` occurrences)**, `low_inlier_ratio` (`190`), and `suspected_water_reflection` (`65`). Side-by-side plots ([shadow_ab_status_and_dispersion.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase5c_shadow_viz/shadow_ab_status_and_dispersion.png)) and demo frame inspections ([preview_rural_f45.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase5c_shadow_viz/preview_rural_f45.png), [preview_rangoon_f60.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase5c_shadow_viz/preview_rangoon_f60.png)) show flat muddy water craters in `rangoonRoad` and shallow shadows/perspective warp in `720p rawSurvey` triggering polarity rejection.

---

## 4. Research-Only Demonstration Artifacts (Task 10)

Rendered with relative-unit overlays only (zero `cm`/`mm`/volume strings):
- [shadow_demo_ruralRoad_202f.mp4](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase5c_shadow_viz/shadow_demo_ruralRoad_202f.mp4) (`202` frames, `9.57 MB`)
- [shadow_demo_rangoonRoad_180f.mp4](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase5c_shadow_viz/shadow_demo_rangoonRoad_180f.mp4) (`180` frames, `4.54 MB`)
