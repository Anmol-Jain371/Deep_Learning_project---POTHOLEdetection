# RoadVision AI — Phase 7B: Vanishing-Point & Horizon-Proxy Methodology and Runtime

> [!NOTE]
> All dynamic horizon estimation algorithms, reliability states, and temporal fallback policies live exclusively in the isolated experimental module [dynamic_horizon_estimator.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/dynamic_horizon_estimator.py). No production files were modified.

---

## 1. Classical CV Vanishing-Point Pipeline

Because Phase 7A concluded that GPU compute budget on the RTX 2050 (4 GB VRAM) is already constrained by YOLOv8n + DAv2 336 keyframes, dynamic horizon estimation was designed to run purely on **lightweight CPU classical computer vision**:

1. **ROI & Reduced-Resolution Preprocessing:**
   - Resizes horizontally to configurable `process_width` (`320 px` recommended default vs full resolution), converts to grayscale, crops perspective ROI (`y ∈ [0.22, 0.92]`, `x ∈ [0.02, 0.98]`), and applies `5×5` Gaussian blur (`gaussian_ksize=5`).
2. **Line Segment Detection Modes Tested:**
   - `lsd_320`: OpenCV `cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)` on `320px`-wide input (`edge_ms ≈ 0.05–0.06 ms`, subpixel line accuracy; handled OpenCV `5.0.0` 2D `(N, 4)` array output via `np.asarray(...).reshape(-1, 4)`).
   - `hough_320`: Canny (`50, 150`) + `cv2.HoughLinesP` (`threshold=28`, minimum length `4.5%` ROI diagonal).
   - `hybrid_320`: Union of both segment sets.
   - `lsd_fullres`: Full native video resolution (`640×360` or `1280×720`) ablation baseline.
3. **Perspective Orientation & Bank Filtering:**
   - Rejects segments shorter than `0.045` normalized length (`too_short`), nearly horizontal bars/cracks/wires with `angle < 12°` (`too_horizontal`), and vertical poles/trees/building edges with `angle > 80°` (`too_vertical`).
   - Classifies surviving oriented lines into left-converging (`bank="left"`, `dx < 0` downward) and right-converging (`bank="right"`, `dx > 0` downward) groups.
4. **Vectorized MSAC RANSAC + Weighted Refit:**
   - Forms pairwise non-parallel intersection hypotheses (`sin(Δθ) >= sin(12°)`), boosting cross-bank `(left × right)` hypotheses (`2×` weight) and capping candidate evaluations deterministically at `ransac_iters=120`.
   - Scores MSAC consensus combining normalized orthogonal point-to-line distance (`<= 0.022`) and midpoint-to-VP ray angle alignment (`<= 5.0°`).
   - Refits the final winning inlier set using length-weighted least squares `_weighted_ls_vp(inliers)` and computes system condition number \(\kappa = \sqrt{\lambda_{\max} / \lambda_{\min}}\).

---

## 2. Geometric Conditioning & Explicit Reliability States

Every frame returns one explicit estimator state and full diagnostic reasons (`reasons` list):

| State | Trigger Conditions | Interpretation & Caveat |
|---|---|---|
| **`VALID`** | `inliers >= 5`, `left >= 1` AND `right >= 1`, `cond <= 28.0`, `dispersion <= 0.055`, plausible VP bounds (`vx ∈ [-0.20, 1.20]`, `vy ∈ [0.05, 0.68]`), temporal jump `<= 0.10`, and median cross-bank `y` agrees within `0.08`. | Strong bilateral perspective convergence in image space. **Not** a calibrated probability or guarantee against foreground parallel edges mimics. |
| **`LOW_CONFIDENCE`** | Plausible intersection (`inliers >= 3`, `cond <= 120.0`), but weak bilateral geometry (`low_inlier_count`, `single_sided_line_support`, `high_condition_number`, `high_vp_dispersion`, `cross_group_unverified`, `cross_group_disagreement`, or `large_temporal_jump`). | Partial perspective cue present (e.g., curb on one side only); excluded from EMA update by default (`accept_low_conf_in_ema=False`). |
| **`UNAVAILABLE`** | `< 2` oriented lines (`insufficient_oriented_lines`), all pairs near-parallel (`parallel_or_degenerate_lines`), `< 3` inliers (`insufficient_inlier_support`), out-of-bounds VP (`implausible_vp_position`), or `cond > 120.0` (`degenerate_conditioning`). | No usable horizon cue; triggers temporal stale/expiry/fallback rules. |

Heuristic confidence is emitted strictly as an engineering diagnostic (`heuristic_confidence ∈ [0, 1]`) and is **not calibrated**. Crucially, an estimated 1-point VP `(vx, vy)` gives only an *image-space y-proxy* and cannot recover camera roll or true 3D horizon angle without intrinsic matrix `K`.

---

## 3. Temporal Filtering & Fallback Policies (`DynamicHorizonTracker`)

Four policies share one stateful tracker interface (`ema_alpha=0.25`, timestamp-based `max_stale_sec=0.50`, `max_expired_sec=1.20`):
- **`A_fixed`:** Constant configured `fixed_horizon_ratio` (`0.40` sedan baseline).
- **`B_dynamic_raw`:** Raw unsmoothed estimator `horizon_y_ratio` (missing / `None` on `UNAVAILABLE`).
- **`C_dynamic_smoothed`:** Timestamp-aware EMA on `VALID` detections, holds last smoothed state across `STALE` gap (`<= max_expired_sec`), drops to `None` upon `EXPIRED`.
- **`D_dynamic_fallback`:** Uses smoothed dynamic horizon through `VALID` and `STALE`, and seamlessly falls back to profile `fixed_horizon_ratio` on `EXPIRED` or cold `LOW_CONFIDENCE`/`UNAVAILABLE`.
- **Automatic Lifecycle Resets:** `DynamicHorizonTracker` checks `(session_id, camera_profile, (W, H))` on every frame and resets filter memory immediately when any context field changes.

---

## 4. Stage-Level Performance Benchmark (Task 9)

Measured over all `657` real video frames (`427` frames @ `360p` + `230` frames @ `720p`) recorded in [phase7b_horizon_results.json](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_horizon_results.json):

| Variant | Resolution | `preprocess` mean | `edge` mean | `line_detect` mean | `ransac` mean | `qa` mean | Total **Mean** (ms) | Total **Median** (ms) | Total **P95** (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`lsd_320`** | Overall (657f) | `1.966` | `0.057` | `3.259` | `4.004` | `0.145` | **`9.431`** | **`8.530`** | **`16.536`** |
| `lsd_320` | `360p` (427f) | `0.877` | `0.055` | `2.786` | `3.529` | `0.138` | `7.385` | `6.842` | `11.634` |
| `lsd_320` | `720p` (230f) | `3.987` | `0.061` | `4.138` | `4.885` | `0.157` | `13.228` | `12.866` | `19.012` |
| `hough_320` | Overall (657f) | — | — | — | — | — | `8.117` | `6.326` | `22.263` |
| `hybrid_320` | Overall (657f) | — | — | — | — | — | `19.413` | `14.141` | `35.642` |
| `lsd_fullres` | Overall (657f) | — | — | — | — | — | `29.643` | `14.454` | `81.269` |
| `lsd_fullres` | `720p` (230f) | `0.697` | `0.414` | **`61.043`** | `0.485` | `0.038` | **`62.678`** | `62.814` | **`84.022`** |

### Selective Refresh Cadence Amortization (`lsd_320`)
Because horizon pitch evolves far more slowly than frame-level object bounding boxes, running `lsd_320` once every \(k\) frames amortizes CPU cost cleanly:
- `k = 1` (every frame): **`9.431 ms/frame`**
- `k = 2`: **`4.716 ms/frame`**
- `k = 3`: **`3.144 ms/frame`**
- `k = 5`: **`1.886 ms/frame`**
- `k = 10`: **`0.943 ms/frame`**

Full visual stage breakdown is saved in [latency_stage_breakdown.png](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7b_viz/latency_stage_breakdown.png).
