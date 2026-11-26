# Matched-Region Dashcam & Matched-Track Relative Depth Stability Report (Phase 5B)

Executed via [benchmark_optimized_relative_depth.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_optimized_relative_depth.py) and logged to [optimized_relative_depth_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_relative_depth_results.json), [optimized_relative_depth_per_region.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_relative_depth_per_region.csv), [matched_track_temporal_per_track.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/matched_track_temporal_per_track.csv), and [matched_track_temporal_observations.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/matched_track_temporal_observations.csv).

> [!CAUTION]
> **Relative Units & Validity Boundary:** Higher reference-fit stability or lower temporal CV reflects **consistency of relative inverse-depth residuals only — NOT physical metric accuracy**. Manual depth ground truth is unavailable and Pothole440 metric verification remains blocked.

---

## 1. RTX 2050 Backbone vs Extraction Latency & Scaling (Task 8)

All four extractors were benchmarked over 3 un-instrumented CPU passes on cached `640×360` frames (with memory traced separately via `tracemalloc` to prevent allocation hook inflation):

- **GPU Backbones (RTX 2050, `peak GPU allocated = 144.26 MB`, `reserved = 164.00 MB`):**
  - `YOLOv8n` (`imgsz=640`): **`14.09 ms`** mean (`12.48 ms` median, `23.09 ms` p95 -> `70.99 FPS`).
  - `MiDaS_small` active frames: **`27.38 ms`** mean (`25.85 ms` median, `38.76 ms` p95 -> `36.52 FPS`).
- **Extraction-Only Latency per Region & Speedup vs Original RANSAC:**

| Method | Mean (ms) | Median (ms) | P95 (ms) | Mean Speedup vs Orig RANSAC | Stage Breakdown (`ann + fit + res + qa`) | 4+ Proposals Frame (ms) | CPU Peak Traced (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `existing_annular` | `4.14` | `2.66` | `10.54` | `2.48x` | `0.80 + 1.86 + 0.94 + 0.54` | `17.89` | `21,131.89` |
| `ols_plane` | `5.41` | `3.37` | `15.37` | `1.89x` | `0.90 + 2.53 + 1.24 + 0.73` | `24.35` | `20,606.02` |
| `original_ransac` | `10.25` | `7.81` | `20.42` | `1.00x` | `0.83 + 7.62 + 1.13 + 0.67` | `50.38` | `22,919.43` |
| **`optimized_ransac` (Strat 512)** | **`3.02`** | **`2.13`** | **`6.73`** | **`3.39x`** (`3.67x` med) | `0.82 + 1.36 + 0.47 + 0.37` | **`14.10`** | **`7,623.63`** |

**Reference-Sample Scaling & Complete Pipeline Timing:**
- Across large annular rings (`>30k` available px), `original_ransac` averages **`15.98 ms`** (`p95 = 32.53 ms`), while `optimized_ransac` drops to **`5.18 ms`** (`2k–10k px`: **`1.50 ms`** vs `6.19 ms`; `10k–30k px`: **`2.18 ms`** vs `8.10 ms`).
- **3.00× Lower Peak CPU Traced Allocations (`7.62 MB` vs `22.92 MB`)** and `~5.6×` faster reference fitting stage (`7.62 ms -> 1.36 ms`).
- Combining GPU + extraction on typical `4+` proposal frames: `YOLOv8n (14.09 ms) + MiDaS (27.38 ms) + Opt RANSAC (14.10 ms)` = **`~55.6 ms/frame` (`~18.0 FPS`)**, down from `~91.9 ms/frame` (`~10.9 FPS`) with original RANSAC! See [perf_and_budget_comparison.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_depth_viz/perf_and_budget_comparison.png).

---

## 2. Matched-Region Dashcam Comparison — 36 Frames, 98 Regions (Task 5)

Strictly separates each method's **own admitted valid population** from the **matched intersections** (`common_valid_all4 = 21` regions valid under all 4 methods; `common_valid_ransac_pair = 40` regions valid under both Original & Optimized RANSAC):

| Method | Admitted Valid (`n=98`) | Admitted Ref MAD (Med) | Common-All-4 (`n=21`) Ref MAD (Med) | Common-All-4 P75 (Med) | Common-All-4 `|ΔP75|` vs Orig RANSAC (Med / Mean) | `±8%` Box Perturb (`ret`, Med Shift) | `+Curb` Contam (`ret`, Med Err) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `existing_annular` | `39` (`39.80%`) | `2.6520` | `2.6211` | `5.4359` | `1.5295` / `5.1688` | `39/39` (`0.4079` / `6.46%`) | `38/39` (`360.2` / `6,262.1%`) |
| `ols_plane` | `37` (`37.76%`) | `3.5763` | `3.0954` | `8.1259` | `1.5766` / `4.6150` | `37/37` (`0.4389` / **`5.40%`**) | `37/37` (`436.8` / `4,734.8%`) |
| `original_ransac` | `49` (`50.00%`) | `2.5996` | `2.1659` | `6.1874` | `0.0000` / `0.0000` | `45/49` (`1.0336` / `11.21%`) | `25/49` (`2.2872` / **`17.15%`**) |
| **`optimized_ransac`** | **`50` (`51.02%`)** | `2.7359` | **`2.1500`** | `7.6321` | **`0.4692` / `1.5089`** | `41/50` (`1.6382` / `11.25%`) | `28/50` (**`2.0180`** / `21.95%`) |

- **Pairwise Matched RANSAC Intersection (`common_valid_ransac_pair = 40` regions):** Median ref MAD `2.6018` (`optimized`) vs `2.5241` (`original`), median `|ΔP75|` difference **`0.7187` relative units** (`6.78%` guarded).
- **Visual Case Comparisons ([case_compare_valid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_depth_viz/case_compare_valid.png), [case_compare_invalid_water.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_depth_viz/case_compare_invalid_water.png), [case_compare_invalid_polarity.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/optimized_depth_viz/case_compare_invalid_polarity.png)):** Confirm identical semantic verdicts (`neighbor_excluded_from_ring`, `suspected_water_reflection`, `unexpected_residual_polarity`) while extracting in `1.41–2.94 ms` instead of `5.29–10.51 ms`.
- **Invalid Reason Shift Note:** Compared to `original_ransac` (`low_inlier_ratio=22`, `unexpected_residual_polarity=29`), `optimized_ransac` recorded `low_inlier_ratio=15` and `unexpected_residual_polarity=34` (with `water=3`, `unstable=2`) because stratified 512-point planes occasionally yield slightly different consensus margins on borderline non-planar patches; perturbation retention is `41/50` (`82%`) vs `45/49` (`92%`), showing marginal sensitivity trade-off from sampling `512` vs `1500` pixels.

---

## 3. Strictly Matched-Track Temporal Stability (`ruralRoad_potHoles.mp4`, Frames 0–119) (Task 6)

Cached `YOLOv8n` + ByteTrack (`conf=0.25, iou=0.5`) produced **22 unique tracks** and **309 total region observations** evaluated on 100% identical boxes, track IDs, depth maps, and timestamps:
- **Matched Subsets:** `common_valid_all4` = **`5` tracks (`78` obs)**; `common_valid_ransac_pair` = **`14` tracks (`153` obs)**.

| Method | Total Valid Coverage / Dropout (`n=309`) | Own `>=4` Tracks | Unmatched Extra Obs (Med P75) | Common-All-4 (`n=5` tracks, `78` obs) Med `|Δ|` / MAD / CV | Approach `ρ` (`box_area` / `frame`) | Common RANSAC-Pair (`14` tr, `153` obs) Med `|Δ|` / MAD / CV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `existing_annular` | `131` (`42.39%`) / `57.61%` | `5` | `53` (`7.6435`) | `2.7274` / `1.7825` / `0.4635` | `0.2934` / `-0.1038` | — |
| `ols_plane` | `126` (`40.78%`) / `59.22%` | `5` | `48` (`10.1144`) | **`2.4934`** / `1.8027` / **`0.3696`** | `0.4486` / `-0.0210` | — |
| `original_ransac` | **`190` (`61.49%`) / `38.51%`** | **`14`** | `112` (`11.6591`) | `3.3825` / `3.3127` / `0.5793` | `0.1334` / `-0.2513` | `13.5872` / `12.2985` / **`0.5565`** |
| `optimized_ransac` | `180` (`58.25%`) / `41.75%` | **`14`** | `102` (`9.1612`) | **`2.6535`** / `3.1296` / **`0.5074`** | `0.1649` / `-0.2141` | **`12.1544`** / **`10.8110`** / `0.6058` |

**Key Temporal & Normalization Findings:**
1. **Resolving Phase 5A Confounding:** Even on identical `common_valid_all4` tracks (`5` cleaner open-pavement tracks), `ols_plane` (`CV = 0.3696`) remains smoother than RANSAC (`0.5074–0.5793`), confirming that part of RANSAC's Phase 5A CV (`0.59`) came from admitting `14` noisy distant tracks (`matched_ransac_pair`: `MAD = 10.81–12.30`), while another part reflects discrete RANSAC plane switching frame-to-frame (which `optimized_ransac` slightly improved over `original_ransac` on `common_valid_all4`: `2.6535` vs `3.3825` median frame delta, `CV 0.5074` vs `0.5793`).
2. **Local Relative Normalization & Affine Check (`5.0 * depth + 100.0`):**
   - Both `scale_normalized_p75` (`p75 / (1.4826 * ref_mad)`) and `frame_iqr_normalized_p75` (`p75 / frame_IQR`) are **strictly affine invariant on any fixed frame** (`max_abs_diff = 0.0` verified empirically).
   - However, **neither yields physical distance invariance**: `frame_iqr_normalized_p75` still exhibits positive correlation with bounding-box approach area (`ρ = 0.4090` OLS, `0.1110` Opt RANSAC), while its small raw magnitude (`< 0.05`) makes unguarded CV numerically unstable (`None` under the `0.05` mean guard).
