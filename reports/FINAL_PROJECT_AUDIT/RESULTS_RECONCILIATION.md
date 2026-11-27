# Experimental Results Reconciliation — RoadVision AI 2.0

All discrepancies between earlier phase reports, legacy V1 files, Phase 13 outputs, and direct audit recomputations are reconciled below from actual inspected artifacts.

---

## 1. Detection mAP@0.50 & Precision Discrepancies (`YOLOv8n` / `YOLOv8s` / Legacy Model #3)

| Source / Artifact | Checkpoint | Dataset Split | Protocol / Threshold | Reported mAP@0.50 | Recomputed / Verified Reality | Resolution |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **Legacy `server.py:L45` & `BenchmarkComparison.jsx:L22-27`** | `pothole_rdd1666_50e/weights/best.pt` (`6,249,123 B`) | Unspecified ("RDD1666") | Claimed "+TTA / Precision 92.4%", UI calls it "YOLOv8s (11.13M, 5.96 MB)" | `84.7%` / `82.1%` | `runs/detect/pothole_rdd1666_50e/results.csv` best epoch 50 = **`0.45765` mAP50**, **`0.56055` P**, **`0.43922` R**. File size `5.96 MB` = **YOLOv8n**, NOT YOLOv8s! | **DISPUTED / DO NOT CITE.** Unreconciled legacy claim & model-name mismatch. |
| **Phase 2 Controlled Eval** (`phase2_yolov8n_vs_yolov8s_controlled.json`) | `pothole_unified_yolov8n_100e_sgd/weights/best.pt` | `dataset_unified/test` (`153` imgs) | Ultralytics `model.val()` default `conf~0.001` PR curve integration (no TTA) | **`0.4865`** | Confirmed verbatim in `phase2_yolov8n_vs_yolov8s_controlled.json` (`P=0.5547, R=0.4969`) | **EXPLORATORY.** Standard AP integration across full `[0.001..1.0]` confidence sweep. |
| **Phase 13 Evaluation** (`result_detection.json`) | `pothole_unified_yolov8n_100e_sgd/weights/best.pt` | `dataset_unified/test` (`153` imgs) | Thresholded `model.val(conf=0.25, iou=0.50, imgsz=640)` | **`0.3643`** (`0.3581` for YOLOv8s) | Confirmed verbatim (`conf=0.25` truncates low-conf recall tail of PR curve -> lower AP). **Bug:** `tp=0, fp=0, fn=0` emitted in `result_detection.json` | **EXPLORATORY + MINOR HARNESS BUG.** Explain `conf=0.25` vs `conf~0.001`; report only as exploratory due to `328` near-dups & blocked `evaluation/ground_truth_labels`. |

---

## 2. Optimized MSAC/RANSAC Latency (`1.445 ms` / `2.660 ms` vs `4.514 ms`)

- **Phase 5B (`optimized_relative_depth_results.json`):** `1.445 ms` mean across 48 synthetic regions; `2.660 ms` across 98 dashcam regions.
- **Phase 13 (`result_geometry.json`):** `4.514 ms` mean for `D_optimized_msac_ransac` vs `9.014 ms` for `C_ransac_baseline` (`~2.0x`).
- **Root Cause Found in Code (`experiments/phase13/geometry_ablation.py:L192-202`):** Inside the single `dt_d = time.perf_counter() - t0` timer block for `D_optimized_msac_ransac`, the script executed **`build_and_sample_annular_region` + `fit_ransac_optimized`**, and THEN immediately executed **`extract_cavity_with_strategy(...)`** (which internally runs **`build_and_sample_annular_region` + `fit_ransac_optimized` a SECOND time**!).
- **Verdict:** Phase 13 `4.514 ms` reflects **double execution**, not true single-call latency (~`2.0–2.66 ms`). Must be corrected in `geometry_ablation.py` and re-run before reporting Phase 13 timing/speedup numbers in the paper.

---

## 3. Depth Coverage & Historical Phase 8A `0.0%` Anomaly

- **Phase 6B (`phase6b_resolution_scaling.json`):** `MiDaS 256` valid coverage `39.30%` (`N=575` boxes using `full_box`); `DAv2 336` `59.65%`.
- **Phase 13 (`result_depth_cavity.json`):** `MiDaS 256` valid coverage `28.57%` (`12/42` boxes on 17 frames using stricter `elliptical_core`); `DAv2 336 FP16` `59.52%` (`25/42`); `DAv2 518 FP16` `66.67%` (`28/42`). Differences reflect sample size (`N=42` vs `N=575`) and cavity core strategy.
- **Historical `runs/benchmark/phase8a_integration_results.json`:** Recorded **`0.0%` `fresh_valid_coverage_pct` (`237 FRESH_INVALID, 0 FRESH_VALID`)** on `balanced` mode due to a historical schema key mismatch (`core_cavity_stats`) subsequently fixed in `roadvision_v2/pipeline.py`. Current execution verified cleanly: DB job `0f286074` (`254 FRESH_VALID`, `177 FRESH_INVALID`, `318 CACHED_VALID` across `784` obs) and fresh Smoke Test C (`7 FRESH_VALID`, `12 FRESH_INVALID`, `9 CACHED_VALID` across `39` obs).

---

## 4. Scheduler Call Reduction & Simulated vs Measured FPS

- **Phase 7A (`phase7a_scheduler_results.json`):** `66.80%` call reduction across `482` frames (simulated `16.07 FPS`).
- **Task 10 Live Benchmark (`balanced` with `fp16_weights`):** `63.54%` call reduction across `ruralRoad` (`20.65`–`24.30` measured wall FPS).
- **Phase 13 (`result_scheduler_temporal.json`):** `80 / 150` calls = **`46.67%` call reduction**, `36.21%` fresh (`168/464`), **`71.34%` effective availability (`(168+163)/464`)**, **`18.63` simulated FPS** (`fp16_autocast` on `ruralRoad[:150]`, using `bytetrack.yaml` instead of `road_tracker.yaml`).
- **Fresh Isolated Smoke Test C (`balanced`, `fp16_weights`, 30f):** `10/30` depth calls, **`14.67 FPS` E2E** (includes video decode, model loop, annotation video write).
- **Verdict:** All numbers reconcile once precision (`fp16_weights` vs `fp16_autocast`), tracker (`road_tracker.yaml` vs `bytetrack.yaml`), and scope (`simulated_sustained_fps` vs full E2E FPS) are stated explicitly.

---

## 5. Phase 9 Quality Subset (`145` vs `105` `HIGH_QUALITY`)

- Verified exact counts (`145 C_deployed` vs `105 E_conservative_gated`) in `result_quality_geo_maint.json`, **with scope caveat**: `quality_geo_maint_eval.py:L155` sliced **`obs_list[:300]`** out of **`784`** total observations in job `0f286074`, leaving 484 observations unevaluated in that table.
- Verified `0f286074/geometry/snap_t15_g0_f000048.json` genuinely exhibits **`quality_index = 0.8516 (HIGH_QUALITY)`** despite `detection_quality = 0.4133` and `temporal_consistency = null`, confirming the Phase 9 available-weight renormalization limitation.

---

## 6. Phase 10 Geometry Snapshot Reference Masks (`42` Historical vs New Runs)

- Direct inspection of `artifacts_v2/` confirms **all 42 historical snapshots** (`0f286074`: 24, `acc1a7dd`: 13, `b3043fd0`: 3, `e122ded7`: 2) have **`0` True `reference_mask_grid` cells** due to the pre-fix `valid_ring` key mismatch.
- Fresh controlled Smoke Tests B, C, and D with current `roadvision_v2/geometry/surface.py` verified **`5/5` (Test B)**, **`5/5` (Test C)**, and **`2/2` (Test D)** snapshots contain non-empty `reference_mask_grid` cells (`100%` recovery after fix).
