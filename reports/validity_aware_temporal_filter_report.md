# Validity-Aware Temporal Filtering Report (Phase 5B)

Experimental module: [experiments/validity_aware_temporal_filter.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/validity_aware_temporal_filter.py); automated verification: [tests/test_validity_aware_temporal_filter.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_validity_aware_temporal_filter.py) (`5/5` tests passing); trajectory plot: [matched_track_temporal_filters.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_depth_viz/matched_track_temporal_filters.png).

> [!IMPORTANT]
> All temporal filters operate strictly on **relative inverse-depth P75 residuals** (not metric cm). Production `roadvision_engine.py` (`TemporalEMASmoother`) was not modified.

---

## 1. Filter Designs & State Contracts

Using identical configuration (`alpha = 0.35`, `rolling_window = 5`, `max_stale_frames = 3`, `max_missed_frames_reset = 8`):

1. **`raw`:** Returns current valid relative residual or `None` when invalid (never fabricates zero).
2. **`legacy_ema_zero_clamp`:** Emulates production's failure mode where invalid/suppressed detections inject `0.0` into EMA state.
3. **`validity_aware_ema`:** Updates $v_t = 0.35\,x_t + 0.65\,v_{t-1}$ **strictly when `is_valid=True`** and $x_t$ is finite. Invalid/corrupted inputs increment `consecutive_invalid_count` and hold $v_{t-1}$ marked **`is_stale=True`** up to `max_stale_frames = 3`; at gap `>= 4`, emits **`None` (`is_expired=True`)** and resets at `>= 8` missed/invalid frames.
4. **`rolling_median_validity_ema`:** Feeds the causal rolling median of the last `5` valid observations into the same staleness-bounded validity-aware EMA, rejecting transient valid spikes cleanly.

Every update records `track_age_frames`, `track_age_sec`, `frames_since_last_valid`, `time_since_last_valid_sec`, `did_update`, `is_stale`, and `is_expired`.

---

## 2. Controlled Synthetic Sequence Evaluation

| Synthetic Scenario | Metric | `raw` | `legacy_ema_zero_clamp` | `validity_aware_ema` | `rolling_median_validity_ema` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`stable_noisy`** (`40` steps, $\sigma=1.2$) | MAE / `mean |Δ|` | `1.0617` / `1.5961` | `0.4746` / `0.4263` | `0.4746` / `0.4263` | **`0.3449` / `0.1222`** |
| **`abrupt_step`** (`6 -> 18`, `45` steps) | MAE / 90% Lag (frames) | **`0.4669` / `0`** | `0.7244` / `5` | `0.7244` / `5` | `1.1263` / `7` |
| **`invalid_dropouts_with_glitches`** | MAE (emitted) / `mean |Δ|` | `0.9000` / `1.1868` | **`2.6193`** / `1.3749` | **`0.4291`** / `0.2809` | `0.4368` / **`0.1357`** |
| **`prolonged_invalid_expiry`** (`11`-step gap) | MAE / Stale / Expired | `0.3108` / `11` / `0` | **`4.9368`** / `11` / `0` | `0.2371` / `3` / **`8`** | **`0.1903`** / `3` / **`8`** |
| **`track_gap_reset`** (`12`-frame miss -> new obj) | Post-reset first Err | `0.0338` | `0.0338`* | `0.0338` | `0.0338` |

*(`*` Under the unified harness expiry reset `max_missed_frames_reset = 8`, absent track IDs clear state cleanly; without disappearance/reset, in-track invalid frames still corrupt `legacy_ema_zero_clamp` heavily toward `0.0`, seen in `MAE = 2.62–4.94` vs `0.19–0.44`.)*

---

## 3. Evaluation on Real `ruralRoad_potHoles.mp4` Matched ByteTrack Stream

Evaluated on all **309 region steps across 22 unique ByteTrack IDs** using `optimized_ransac` (`valid = 180`, `invalid = 129`):

| Filter Mode | Emitted Non-`None` Steps (`/309`) | Stale Held Steps | Expired (`None`) Steps | Mean `|Δ|` (All Emitted) | Median `|Δ|` (All Emitted) | Mean `|Δ|` (Valid Steps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `raw` | `180` | `104` (post-valid invalid) | `0` | `14.4877` | `2.9095` | `14.4877` |
| `legacy_ema_zero_clamp` | `309` (never `None`!) | `129` (pulled to `0`) | `0` | `6.5302` | `1.0732` | `6.7682` |
| `validity_aware_ema` | `254` | `74` | `30` | `5.5293` | `0.3795` | `8.0708` |
| **`rolling_median_validity_ema`** | **`254`** | `74` | **`30`** | **`4.4142`** | **`0.1486`** | **`6.4431`** |

Visual inspection of **Real Track #12** in [matched_track_temporal_filters.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_depth_viz/matched_track_temporal_filters.png) illustrates both behaviors directly: during the dropout gaps before frame 75, `legacy_ema_zero_clamp` (red) repeatedly collapses near `0` and under-reports frame 76 (`< 350` instead of `~500`), whereas `validity_aware_ema` (blue) and `rolling_median_validity_ema` (green) pause numeric updates, mark `is_stale=True` briefly, emit `None` past `max_stale_frames = 3` (`30` expired steps prevented stale ghost alerts), and cleanly resume at `~500` when valid observations return.

---

## 4. Recommendation & Integration Guardrail

- **Recommended Filter Strategy:** Use **`validity_aware_ema`** when responsiveness to rapid approach growth (`lag = 5` vs `7` frames) outweighs spike immunity, OR **`rolling_median_validity_ema` (`window=5, alpha=0.35, max_stale=3`)** when jitter suppression (`median |Δ| = 0.1486`) is preferred.
- **Stop Gate:** Keep all Phase 5B modules experimental until approved for production refactor, keeping explicit invalid state (`None` + staleness flags) rather than clamping to zero.
