# Phase 7A — Integration Readiness & Final Recommendation (Task 14)

All production files ([roadvision_engine.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_engine.py), `server.py`, dashboard) remain strictly untouched, verified by [test_adaptive_depth_scheduler.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_adaptive_depth_scheduler.py) and `git diff`. Cross-phase unit verification across all 41 tests is green.

---

## 1. Selected Outcome & Policy Verdict

### Selected Outcome: **B — Scheduling improves average throughput but requires further deadline, freshness or lifecycle improvements**

- **Recommended Experimental Architecture for Further Iteration:**
  - **Depth Backbone & Precision:** `DepthAnythingV2 Small`, `input_size = 336` (`336 × 602`), `fp16_weights` (`model.half()`, `torch.inference_mode()`), paired with `elliptical_core` cavity extraction + `stratified_quadrant` `512`-sample RANSAC.
  - **Scheduling Policy Baseline:** `D_track_aware` (or `E_budget_aware` when latency throttling is needed cleanly) with `TrackDepthCache` (`max_cache_age_sec = 0.45 s`, generation-ID anti-reuse) + **`TimeAwareEMA` ($\tau = 0.16\,\text{s}$)**.
  - **Why Neither Outcome A Nor C Alone Is Optimal Today:**
    1. **Outcome A rejected for immediate production readiness:** Although `DAv2 336` + `BudgetAware` achieves **`20.65 FPS` live (`48.06 ms` mean)** on `362` mixed frames (faster amortized than `MiDaS` candidate-gated at `17.47 FPS`) and **`27.95 FPS` (`35.8 ms` mean)** on `300f` 360p, **`36.5%`–`51.0%` of frames still breach the `40 ms` (`25 FPS`) deadline (`P95 = 120–143 ms`)**, `BudgetAware` suppresses `144` valid refresh requests (`fresh_valid` falls `63.8% -> 23.9%`), and `54.5 ms` mean / `109.9 ms` P95 GPU contention persisted in the async worker probe.
    2. **Outcome B preferred over static Outcome C (`MiDaS continuous`):** Retaining `MiDaS 256` fails `53.7%` of Primary detections (`12.3%` on urban `rawSurvey`) while STILL missing **`79.3%` (`live`) to `86.7%` (`482f`)** of `40 ms` deadlines due to full-frame decode + YOLO + RANSAC + MiDaS latency (`56.75 ms` live mean). Meanwhile, **`DAv2 336 TrackAware`** lifts effective availability from `46.3%` (`MiDaS`) to **`67.8%` (`Primary`) / `97.7%` (`Semi-Indep`)** with `0` starved tracks and `50.2%–74.7%` fewer depth forward passes (`demo` clips saved **`38.6%` [rural]** and **`60.0%` [urban 10 FPS]**!). Production keeps `MiDaS` untouched until Phase 7B fixes keyframe stalling.

---

## 2. Specific Technical Gaps to Address Before Production Integration

1. **Keyframe Latency & Single-GPU Async Contention:** Reduce `DAv2 336` `preproc + infer` (`~70 ms`, especially `30.6 ms` CPU preprocessing on `720p` frames) via GPU-resident resize/normalization (`torchvision`/CUDA bilinear resize avoiding CPU NumPy copies) and/or **CUDA-12-matched TensorRT FP16** compilation, or chunked multi-frame depth amortization so P95 frame latency drops below `40 ms`.
2. **FP16 Local Stability Margin:** On 98 boxes across 36 frames, `fp16_weights` matches `FP32` well (`0.9922` Spearman at `336`), yet produces `3/98` boundary validity flips and `15.55%` P95 local `scale_norm_p75` delta when `ref_mad` is tiny.
3. **Budget-Triggered Freshness Drop:** `E_budget_aware` achieves higher wall FPS mainly by dropping fresh observations (`38.4% -> 23.9%`) and allowing `115` `NOT_EVALUATED` detections (`P95` cache age `320 ms`, `P95` missed score change `1.52`). Priority-based track budgeting (prioritizing highest-conf / unmeasured / largest-shift tracks instead of whole-frame budget skipping when depth runs) would protect freshness.
4. **Independent Dataset Evaluation & Extrinsic Calibration:** No truly unseen video session exists on disk; physical metric depth accuracy remains unverified.

---

## 3. Summary Table of Final Selected Reference Numbers

| Dimension | Value Observed | Source |
| :--- | :--- | :--- |
| Selected Experimental Config | `DAv2 336` (`fp16_weights`) + `D_track_aware` / `E_budget_aware` | [phase7a_scheduler_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_scheduler_results.json) |
| Live Sustained Throughput (`362f` mixed) | `14.75 FPS` (`TrackAware`) \| `20.65 FPS` (`BudgetAware`) | Live verification Task 10 |
| P95 Latency (`482f` / `362f` live) | `166.7 / 144.6 ms` (`TrackAware`) \| `143.1 / 120.1 ms` (`Budget`) | Task 10 |
| `25 FPS` (`40 ms`) Deadline Misses | `62.2% / 52.5% live` (`TrackAware`) \| `51.0% / 36.5% live` (`Budget`) | Task 10 |
| Fresh Valid / Eff. Valid Avail (`482f`) | `38.4% / 67.8%` (`TrackAware`) \| `23.9% / 69.5%` (`Budget`) | Task 8 |
| P95 Cache Age (`482f` / `300f`) | `120 ms / 160 ms` (`TrackAware`) \| `320 ms / 160 ms` (`Budget`) | Task 8 |
| Demo Videos Verified | [ruralRoad demo](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_viz/phase7a_demo_ruralRoad_dav2_336_track_aware.mp4) (`38.6%` saved), [rawSurvey 10fps](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_viz/phase7a_demo_rawSurvey_dav2_336_track_aware.mp4) (`60.0%` saved) | Task 12 |
