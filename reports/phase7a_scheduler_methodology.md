# Phase 7A — Adaptive Track-Aware Depth Scheduler & Cache Methodology (Tasks 3, 4, 5 & 9)

Modules: [track_depth_cache.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/track_depth_cache.py), [adaptive_depth_scheduler.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/adaptive_depth_scheduler.py) | Unit Tests: [test_track_depth_cache.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_track_depth_cache.py), [test_adaptive_depth_scheduler.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_adaptive_depth_scheduler.py).

> [!IMPORTANT]
> **Relative Inverse-Depth & Causality Rules:** All decisions use strictly current or past timestamps `t <= t_cur` (zero future lookahead), expose relative scores only (never cm/mm), never warp old depth maps across frames, and never feed cached or invalid observations into temporal EMA states.

---

## 1. Track-Level Depth Cache (`TrackDepthCache`)

Each tracked pothole is indexed by `(video_session_id, track_id, track_generation_id)` inside `TrackCacheEntry`:
- **Stored State:** Last valid relative score (`scale_norm_p75`), `raw_p75`, `iqr_norm_p75`, `last_valid_frame_id/idx/ts`, `last_attempted_frame_idx/ts`, `last_attempt_valid`, `ref_diagnostics`, `depth_model_id`, `depth_input_size`, `inference_precision`, `last_evaluated_box_xyxy`, `cache_age_sec`, `fresh_valid_count`, `cached_read_count`, and expiration status.
- **Explicit Output States:** `FRESH_VALID`, `FRESH_INVALID`, `CACHED_VALID`, `EXPIRED`, `NOT_EVALUATED`.
- **Timestamp Expiration Across Heterogeneous FPS:** Cache validity expires strictly in wall-clock time (`max_cache_age_sec = 0.45 s`; shortened to `max_invalid_stale_sec = 0.25 s` if the latest attempt failed) so `10 FPS` (`raw_survey.mp4`) and `25 FPS` (`ruralRoad`, `Rangoon`) expire cleanly at identical physical durations. Expired entries return `None` (`null`), never `0.0`.
- **Track-ID Reuse Protection:** Reappearance after `max_absence_reset_sec = 0.60 s` OR after a `> 0.20 s` gap with zero IoU and normalized centroid jump `> 1.75 × bbox_diag` bumps `track_generation_id` and resets numeric cache + temporal filter states.

---

## 2. Five Scheduler Strategies & Causal Trigger Policy

| Strategy | Invocation Rule | Explicit Trigger Reasons Tracked |
| :--- | :--- | :--- |
| **`A_always_on`** | Every frame regardless of candidates | `ALWAYS_ON` |
| **`B_candidate_gated`** | Skips when zero boxes exist | `CANDIDATE_PRESENT`, `NO_CANDIDATES` |
| **`C_fixed_interval`** | Every `k ∈ {2, 3, 4, 5}` candidate frames | `FIXED_INTERVAL`, `FIXED_INTERVAL_SKIP`, `NO_CANDIDATES`, `SCENE_CHANGE` |
| **`D_track_aware`** | Runs 1 shared full-frame forward pass when any active track triggers | `NEW_TRACK`, `MEASUREMENT_EXPIRED` (`>= 0.45 s`), `BOX_AREA_CHANGE` (`>= 30%`), `LOW_BOX_OVERLAP` (`IoU < 0.48`), `BOX_POSITION_CHANGE` (`>= 35% diag`), `INVALID_RETRY` (`0.12–0.36 s` exponential backoff), `PERIODIC_REFRESH` (`0.16 s`, `0.12 s` if prev `inlier_ratio < 0.55`), `SCENE_CHANGE` (`thumb L1 >= 38`), `CAMERA_CHANGE`, `CACHED_HOLD` |
| **`E_budget_aware`** | `D_track_aware` + latency budget gate (`ema > 38 ms` or `dt < 0.08–0.10 s`), overridden on `starvation >= 0.40 s` / scene cut | All of `D` plus `SKIPPED_BY_BUDGET` *(detections always preserved!)* |

---

## 3. Temporal Filter Compatibility Under Irregular Intervals (Task 9)

On skipped keyframes, `read_non_keyframe_state()` returns cached EMA outputs read-only (`is_fresh=False, is_cached=True`) without touching `ema_fixed_score`, `rollmed_ema_score`, or `ema_timeaware_score`. On `FRESH_VALID` updates only, we compare fixed-step `ValidityAwareEMA` ($\alpha = 0.35$), `RollingMedianEMA` ($W=5$), and continuous-time irregular-interval EMA ($\alpha_{\text{eff}} = 1 - e^{-\Delta t / \tau}$, $\tau = 0.16\,\text{s}$):

| Test Cadence (`25 FPS`, step `4.0 -> 12.0` at `t=1.0 s`) | `ema_fixed_score` ($\alpha=0.35$) 90% Lag | `rollmed_ema_score` 90% Lag | `ema_timeaware_score` ($\tau=0.16\,\text{s}$) 90% Lag |
| :--- | :---: | :---: | :---: |
| Dense fresh updates (`dt = 0.04 s`) | **`0.20 s`** | `0.28 s` | `0.32 s` |
| Irregular skipped updates (`dt = 0.12–0.20 s`) | `0.76 s` *(+3.8× slower!)* | `1.08 s` *(+3.9× slower!)* | **`0.44 s`** *(consistent response time!)* |
