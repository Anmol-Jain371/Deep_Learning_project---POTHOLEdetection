# Phase 6B — Small and Distant Pothole Sensitivity Analysis (Task 4)

> [!NOTE]
> All sensitivity buckets were frozen prior to running [benchmark_dav2_resolution_scaling.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_dav2_resolution_scaling.py) ([phase6b_per_region.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_per_region.csv)). Every category—including poor-performing strata—is reported explicitly below.

---

## 1. Frozen Stratification Criteria

- **Original-Frame Area (`area_frac = bw * bh / (W * H)`):** `tiny_lt_0.5pct`, `small_0.5_to_1.5pct`, `medium_1.5_to_5pct`, `large_ge_5pct`.
- **Effective ViT Token Size (`eff_min_dim = min(bw * net_w/W, bh * net_h/H)` inside network tensor):** `sub1_vit_patch_lt14px` (`<1` DINOv2 `14×14` patch), `patch1to2_14_to_28px`, `patch2to4_28_to_56px`, `patch_ge4_ge56px`.
- **Vertical Road Position (`y2 / H`):** `distant_top_y2_lt_0.50`, `mid_road_0.50_to_0.75`, `near_bot_ge_0.75`.
- **Detector Confidence:** `low_conf_lt_0.40`, `mid_conf_0.40_to_0.65`, `high_conf_ge_0.65`.
- **Resolution & Environment:** `640x360` vs `1280x720`; `rural`, `rangoon`, `urban_rawSurvey`, plus holdout `rural_holdout`, `rangoon_holdout`, `test_clip`.

---

## 2. Sensitivity by Pothole Size & Effective Depth-Network Patch Size

### Exploratory 360f (`n = 575`) & Holdout (`n = 481`) Area-Fraction Validity

| Area Stratum (`Split`) | `n` | `MiDaS 256` | `DAv2 266` | `DAv2 336` | `DAv2 392` | `DAv2 518` | Dominant Failure on Low-Res |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `tiny_lt_0.5pct` (`Expl.`) | `2` | `0.0%` | `0.0%` | `0.0%` | `0.0%` | **`100.0%`** | Polarity lost (`100%` `< 518`) |
| `small_0.5_to_1.5pct` (`Expl.`) | `152` | `51.97%` | `70.39%` | **`80.26%`** | `77.63%` | `78.95%` | Polarity fail drops `48.0% -> 19.1%` |
| `medium_1.5_to_5pct` (`Expl.`) | `236` | `39.41%` | **`35.59%`** | `54.24%` | **`63.98%`** | `61.44%` | Severe polarity loss at `266` (`55.9%`) |
| `large_ge_5pct` (`Expl.`) | `185` | `29.19%` | `43.78%` | **`50.27%`** | `35.68%` | `47.57%` | Foreground clutter & flush patches (`50%–64%` pol fail) |
| `small_0.5_to_1.5pct` (`Holdout`) | `139` | `58.99%` | `79.86%` | `95.68%` | `95.68%` | **`97.84%`** | `266` polarity `20.1%` vs `2.2%–4.3%` |
| `medium_1.5_to_5pct` (`Holdout`) | `202` | `70.79%` | `74.26%` | `62.87%` | `73.27%` | **`87.62%`** | `518` sharply leads on medium cavities |
| `large_ge_5pct` (`Holdout`) | `140` | `65.00%` | `56.43%` | `64.29%` | `60.71%` | **`72.14%`** | `38.6%` pol fail at `266` vs `22.1%` at `518` |

### ViT `14×14` Patch-Resolution Effect (`eff_dim_bucket`, Exploratory 360f)

- When resizing to `266 × 476`, **one box fell below `< 1` ViT patch (`sub1_vit_patch_lt14px`)** and **`218` boxes fell into `1–2` ViT patches (`patch1to2_14_to_28px`: `57.34%` valid, `36.70%` polarity fail, `8.72%` reference inlier fail)**.
- Increasing resolution to `336`/`518` promoted most boxes into `2–4+` patches (`patch2to4`: `58.26% -> 70.77%`; `patch_ge4`: `47.37%` at `266` vs `55.91%` at `518`), cutting `266`'s `low_inlier_ratio` failures inside `1–2` patch buckets from **`8.72%` down to `0.0%`**.
- **Visual Contrast Evidence:** On the distant small pothole in [phase6b_panel_small_distant_rural_f0037.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_viz/phase6b_panel_small_distant_rural_f0037.png) (`area_frac = 0.0108`, `y2 = 0.469`, `conf = 0.71`), all DAv2 sizes detect it validly, **yet normalized local cavity contrast rises steadily with resolution (`scale_norm_p75`: `0.535` [`266`] -> `0.902` [`336`] -> `1.103` [`392`] -> `1.310` [`518`]) while reference MAD drops (`0.0136 -> 0.0055`)**.

---

## 3. Position, Confidence, Environment & Poor-Performing Strata

| Dimension & Stratum (`Expl. 360f`) | `n` | `MiDaS 256` | `DAv2 266` | `DAv2 336` | `DAv2 392` | `DAv2 518` | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Position:** `distant_top_y2_lt_0.50` | `112` | `39.29%` | `58.04%` | **`66.07%`** | `63.39%` | `50.00%` | *(Holdout `n=126`: `77.8%` MiDaS, `95.2%–97.6%` DAv2)* |
| **Position:** `mid_road_0.50_to_0.75` | `217` | `40.55%` | `58.53%` | `47.47%` | `47.47%` | **`70.97%`** | **`336`/`392` regress badly on mid-distance** (`47.5%` vs `71.0%`) |
| **Position:** `near_bot_ge_0.75` | `246` | `38.21%` | `32.52%` | **`67.48%`** | `65.45%` | `58.94%` | `266` collapses (`66.3%` pol fail) on large foreground boxes |
| **Conf (Poor Stratum):** `low_conf_lt_0.40` | `115` | **`26.96%`** | `40.87%` | `40.87%` | **`45.22%`** | `44.35%` | Road-shoulder FPs / `16.5%–22.6%` inlier fails across all models |
| **Conf:** `mid_conf_0.40_to_0.65` | `278` | `38.13%` | `55.76%` | `60.79%` | `56.12%` | **`65.47%`** | Steady DAv2 gain up to `518` |
| **Conf:** `high_conf_ge_0.65` | `182` | `48.90%` | `38.46%` | **`69.78%`** | **`69.78%`** | `67.03%` | **`266` severely regresses (`38.5%`, worse than MiDaS `48.9%`!)** |
| **Env (Poor Stratum):** `rangoon` (`Expl.`) | `112` | **`24.11%`** | `42.86%` | `41.96%` | **`26.79%`** | **`45.54%`** | Non-planar shadows & water; `392` drops to `26.8%` |
| **Env (Poor Stratum):** `test_clip` (`Hold.`) | `25` | `12.00%` | **`4.00%`** | `32.00%` | `12.00%` | **`48.00%`** | Very shallow dry depression fails at low resolutions |

---

## 4. Bounding-Box Perturbation (`±8%`) Sensitivity by Pothole Size (Dev-36)

| Configuration | All Boxes Retained (`n_valid`) | All Med. Rel. Shift | Small Boxes (`< 1.5%` area) Retained | Small Med. Rel. Shift |
| :--- | :---: | :---: | :---: | :---: |
| `midas_256_full` | `41/50` | `11.25%` | `16/21` | `11.81%` |
| `dav2_266_elliptical` | `61/66` | `8.86%` | `24/25` | **`9.16%`** |
| `dav2_336_elliptical` | **`73/73`** | `7.04%` | `26/26` | `6.83%` |
| `dav2_392_elliptical` | `63/70` | `7.22%` | `20/21` | `4.61%` |
| **`dav2_518_elliptical`** | `69/71` | **`5.76%`** | `23/23` | **`2.77%`** |

- **Monotonic Small-Object Stability Drop:** On small potholes (`< 1.5%` frame area), reducing DAv2 resolution from `518 -> 336 -> 266` **more than triples bounding-box perturbation shift (`2.77% -> 6.83% -> 9.16%`)**, demonstrating clear loss of sharp sub-patch cavity localization below `392`/`518`.
