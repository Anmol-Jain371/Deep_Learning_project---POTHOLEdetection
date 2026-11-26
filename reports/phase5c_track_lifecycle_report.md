# Phase 5C Temporal Lifecycle, Fault Injection & Runtime Scaling Report

Executed via [benchmark_shadow_depth_pipeline.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_shadow_depth_pipeline.py) (`Tasks 6 & 8`) and [test_shadow_depth_pipeline.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_shadow_depth_pipeline.py). Plotted in [shadow_latency_scaling_and_stages.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase5c_shadow_viz/shadow_latency_scaling_and_stages.png).

---

## 1. Temporal Lifecycle & Failure Injection Stress Suite (Task 6)

All 11 controlled failure, lifecycle, and environment-transition tests executed deterministically (`max_stale_frames = 3`, `max_missed_frames_reset = 8`, `alpha = 0.35`, `rolling_window = 5`):

| # | Stress Scenario | Observed Behavior & Key Numbers | Status |
| :-: | :--- | :--- | :---: |
| 1 & 2 | Consecutive Invalid Frames | Frame 0 `VALID` (`9.6439`); Frames 1–3 `STALE` (score frozen at `9.6439`, stale age `1, 2, 3`); Frames 4–5 **`EXPIRED`** (`smoothed = None`) | PASS |
| 3 | Temporary Detection Loss (`< 8` f) | 2-frame miss preserves blended prior history upon reappearance (`smoothed = 7.0262` vs raw `12.3992`) | PASS |
| 4 & 5 | Track Expiry & ID Reuse (`>= 8` f) | 9-frame absence resets track completely; reused ID=9 begins fresh at `track_age_frames = 0` (`zero_leak_diff = 0.0`) | PASS |
| 6 | Abrupt Cavity Step (`6 -> 22`) | 90% step response lag: `B` (`validity_aware_ema`) = **`5` frames**, `C` (`rolling_median_validity_ema`) = **`7` frames** | PASS |
| 7 | Missing & NaN Depth Inputs | `depth_map=None` -> `NOT_EVALUATED` (B/C) while Legacy A finishes; `>50%` crater NaN -> `INVALID`, `smoothed = None` | PASS |
| 8 | Extreme Annular Ring Outliers (`+500`) | Method C ignores `+500` block (`is_valid = True`, `ref_mad = 7e-6`) | PASS |
| 9 & 10 | `360p -> 720p` + Neighbor + Profile Switch | Mid-run resolution (`1280×720`), `commercial_truck_bus` profile, and overlapping neighbors handled cleanly (`VALID`, zero crash) | PASS |
| 11 | Experimental Fault Hook Injection | `RuntimeError` injected inside `B_filter` returns `NOT_EVALUATED` for B while **Legacy A and sibling C remain `VALID`** | PASS |

---

## 2. End-to-End Runtime & Candidate-Scaling Breakdown (Task 8 — `567` Video Frames)

Un-instrumented per-stage latencies measured separately from the isolated `tracemalloc` memory pass (`peak shadow CPU = 3,415.04 KB`; `peak GPU VRAM = 178.14 MB alloc / 194.00 MB res`):

### Stage-by-Stage Latency Distribution

| Pipeline Stage | Scope | Mean (ms) | Median (ms) | P95 (ms) | Equivalent FPS |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `video_decode` | Per frame | `1.20` | `0.50` | `3.08` | `836.0` |
| `det_and_bytetrack` | Per frame | `14.93` | `13.57` | `20.54` | `67.0` |
| `midas_preprocess_active` | Active frames (`RGB`) | `6.29` | `4.30` | `13.01` | `158.9` |
| `midas_inference_active` | Active frames (RTX 2050) | `20.01` | `17.84` | `28.40` | `50.0` |
| `annular_sampling_standalone` | Per region (CPU) | `0.40` | `0.34` | `0.72` | `2,518.7` |
| `ref_fitting_standalone` | Per region (CPU) | `0.81` | `0.75` | `1.22` | `1,230.5` |
| `cavity_extraction_standalone` | Per region (CPU) | `0.26` | `0.23` | `0.50` | `3,816.4` |
| `quality_gating_standalone` | Per region (CPU) | `0.20` | `0.17` | `0.39` | `4,931.7` |
| `temporal_filtering` (B / C) | Per region (CPU) | `0.01` / `0.02` | `0.01` / `0.01` | `0.02` / `0.03` | `>60,000` |
| `csv_logging` / `demo_rendering` | Per frame (Optional) | `0.11` / `4.07` | `0.10` / `3.80` | `0.25` / `5.34` | `—` |

### Frame Throughput Scaling Across Candidate Counts (`0..4+` Detections)

| Candidates / Frame | Shadow Mode (`A+B+C+D`) Mean / P95 (ms) | Standalone Opt (`C`) Mean / P95 (ms) | Standalone Legacy (`A`) Mean / P95 (ms) |
| :---: | :---: | :---: | :---: |
| `0` (gating skips MiDaS) | `14.66` / `18.33` (`68.2 FPS`) | **`14.65`** / `18.31` (`68.3 FPS`) | `14.65` / `18.31` (`68.3 FPS`) |
| `1` | `54.07` / `103.05` (`18.5 FPS`) | **`46.47`** / `75.98` (`21.5 FPS`) | `42.44` / `63.34` (`23.6 FPS`) |
| `2` | `67.01` / `109.72` (`14.9 FPS`) | **`55.69`** / `78.78` (`18.0 FPS`) | `49.56` / `64.99` (`20.2 FPS`) |
| `3` | `62.98` / `79.22` (`15.9 FPS`) | **`56.56`** / `73.70` (`17.7 FPS`) | `49.92` / `64.56` (`20.0 FPS`) |
| `4+` | `58.77` / `74.29` (`17.0 FPS`) | **`51.85`** / `63.55` (`19.3 FPS`) | `45.51` / `55.58` (`22.0 FPS`) |

- Actual measured multi-video **shadow-mode throughput** (including CSV bookkeeping & rendering overhead on 382/567 frames) ranged from **`14.76 FPS` (`720p rawSurvey`) to `16.93–19.88 FPS` (`360p`)**.
- Stripping shadow A/B/D overhead leaves **standalone optimized Method C within `~4–6 ms/frame` of legacy production speed (`~17.7–21.5 FPS` active, `68.3 FPS` empty-road)** — proving runtime cost is **already near-parity**.
