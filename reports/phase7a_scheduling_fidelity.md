# Phase 7A — Scheduling Fidelity & Cache Freshness Report (Tasks 6, 7 & 8)

Sources: [phase7a_scheduler_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_scheduler_results.json), [phase7a_per_frame_scheduling.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_per_frame_scheduling.csv), [phase7a_per_track_measurements.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_per_track_measurements.csv), [phase7a_freshness_and_expiration.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_viz/phase7a_freshness_and_expiration.png).

> [!NOTE]
> **Provenance & Exploratory Labeling (Task 7):** All parameters were frozen before running evaluation (`phase7a_v1_frozen_before_eval`). **Primary (`482f`, `1,051` proposals, `57` track generations):** pre-examined `ruralRoad_202f` (25 FPS), `rawSurvey_160f_10fps` (native 10 FPS, 720p), and `rangoonRoad_f200_319` (25 FPS). **Semi-Independent Contiguous Trajectory (`300f`, `348` proposals):** `rangoonRoad_f350_649` (25 FPS, never evaluated by ByteTrack/depth in Phase 5C/6A/6B, though sparse static frames appeared in early dev catalogs — labeled exploratory).

---

## 1. Scheduling Fidelity vs Same-Model Always-On Baseline (Primary `482f` / `1,051` Proposals)

| Policy (`Model / Mode`) | Depth Calls (`/482`) | Avoided (`%`) | `FRESH_VALID` (`%`) | `CACHED_VALID` | `FRESH_INV` | `EXPIRED` | `NOT_EVAL` | Eff. Valid Avail (`%`) | Never-Meas Tracks (`/57`) | Cache Age Med / P95 (ms) | Agree (`%`) / Matched $\rho$ | Missed $\Delta$ Med / P95 | $\rho$ vs `518` Ref |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MiDaS_AlwaysOn` | `482` (`100%`) | `0.0%` | `487` (`46.3%`) | `0` | `564` | `0` | `0` | `46.3%` | `0` | `0 / 0` | `100%` / `1.000` | `—` | `0.331` |
| `MiDaS_CandGated` | `405` (`84.0%`) | `16.0%` | `487` (`46.3%`) | `0` | `564` | `0` | `0` | `46.3%` | `0` | `0 / 0` | `100%` / `1.000` | `—` | `0.331` |
| `DAv2_336_AlwaysOn` | `482` (`100%`) | `0.0%` | **`671` (`63.8%`)** | `0` | `380` | `0` | `0` | `63.8%` | `0` | `0 / 0` | `100%` / `1.000` | `—` | `0.579` |
| `DAv2_336_CandGated` | `405` (`84.0%`) | `16.0%` | `671` (`63.8%`) | `0` | `380` | `0` | `0` | `63.8%` | `0` | `0 / 0` | `100%` / `1.000` | `—` | `0.579` |
| `DAv2_336_Fixed_k2` | `203` (`42.1%`) | `57.9%` | `336` (`32.0%`) | `365` | `187` | `62` | `101` | `66.7%` | `5` | `40 / 120` | `92.0%` / `0.937` | `0.244 / 1.227` | `0.558` |
| `DAv2_336_Fixed_k3` | `137` (`28.4%`) | `71.6%` | `230` (`21.9%`) | `458` | `129` | `86` | `148` | `65.5%` | `4` | `40 / 200` | `87.7%` / `0.923` | `0.250 / 1.447` | **`0.587`** |
| `DAv2_336_Fixed_k4` | `102` (`21.2%`) | `78.8%` | `171` (`16.3%`) | `500` | `92` | `103` | `185` | `63.8%` | `7` | `80 / 280` | `86.7%` / `0.894` | `0.268 / 1.508` | `0.558` |
| `DAv2_336_Fixed_k5` | `82` (`17.0%`) | `83.0%` | `133` (`12.7%`) | `486` | `82` | `128` | `222` | `58.9%` | `11` | `80 / 300` | `82.9%` / `0.887` | `0.269 / 1.273` | `0.553` |
| **`DAv2_336_TrackAware`** | `240` (`49.8%`) | **`50.2%`** | **`404` (`38.4%`)** | `309` | `273` | **`0`** | `65` | **`67.8%`** | **`0`** | **`0 / 120`** | **`95.2%` / `0.956`** | `0.250 / 1.186` | `0.532` |
| **`DAv2_336_BudgetAware`** | `160` (`33.2%`) | **`66.8%`** | `251` (`23.9%`) | `479` | `206` | **`0`** | `115` | **`69.5%`** | **`0`** | `80 / 320` | `91.2%` / `0.890` | `0.266 / 1.516` | `0.504` |
| `DAv2_266_TrackAware` | `248` (`51.5%`) | `48.5%` | `373` (`35.5%`) | `282` | `327` | `0` | `69` | `62.3%` | `0` | `0 / 120` | `94.8%` / `0.921` | `0.313 / 2.099` | `0.544` |
| `DAv2_266_BudgetAware` | `166` (`34.4%`) | `65.6%` | `251` (`23.9%`) | `450` | `230` | `0` | `120` | `66.7%` | `0` | `80 / 300` | `88.5%` / `0.884` | `0.351 / 1.674` | `0.528` |
| `DAv2_518_AlwaysOn_Ref` | `482` (`100%`) | `0.0%` | `749` (`71.3%`) | `0` | `302` | `0` | `0` | `71.3%` | `0` | `0 / 0` | `—` | `—` | `1.000` |

*Time to First Valid (Median / P95):* `0.00 s / 0.312–0.480 s` across `AlwaysOn`, `TrackAware`, and `BudgetAware` (`0.08–0.12 s / 0.43–0.56 s` for `Fixed k=3..5`). Track-level coverage averages `62.1%` (`TrackAware`) and `62.8%` (`BudgetAware`) vs `60.6%` (`AlwaysOn`) and `39.0%` (`MiDaS`).

---

## 2. Semi-Independent `300f` Trajectory (`rangoonRoad_f350_649`, `348` Proposals)

| Policy | Depth Calls (`/300`) | Avoided (`%`) | `FRESH_VALID` (`%`) | Eff. Valid Avail (`%`) | Never-Meas Tracks | Agree (`%`) / Matched $\rho$ | Missed $\Delta$ Med / P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MiDaS_AlwaysOn` | `300` (`100%`) | `0.0%` | `157` (`45.1%`) | `45.1%` | `0` | `100%` / `1.000` | `—` |
| `DAv2_336_AlwaysOn` | `300` (`100%`) | `0.0%` | **`320` (`92.0%`)** | `92.0%` | `0` | `100%` / `1.000` | `—` |
| `DAv2_336_Fixed_k3` | `94` (`31.3%`) | `68.7%` | `104` (`29.9%`) | `93.4%` | `1` | `93.4%` / `0.714` | `0.560 / 1.950` |
| **`DAv2_336_TrackAware`** | **`76` (`25.3%`)** | **`74.7%`** | `89` (`25.6%`) | **`97.7%`** | **`0`** | **`94.2%`** / `0.644` | `0.622 / 1.931` |
| **`DAv2_336_BudgetAware`** | **`72` (`24.0%`)** | **`76.0%`** | `83` (`23.9%`) | **`97.7%`** | **`0`** | **`94.2%`** / `0.672` | `0.601 / 1.951` |
| `DAv2_518_AlwaysOn_Ref` | `300` (`100%`) | `0.0%` | `159` (`45.7%`) | `45.7%` | `0` | `—` | `—` |

---

## 3. Fidelity Failures & Rapid-Change Refresh Limitations

1. **Starvation & Expiration Under Blind Fixed Intervals:** Fixed intervals `k=2..5` left **`4–11` tracks never measured** on Primary `482f` and accumulated **`62–128` `EXPIRED` steps** (especially on `10 FPS` `raw_survey.mp4`), whereas `TrackAware` & `BudgetAware` achieved **`0` starved tracks** and **`0` expired gaps** (`MEASUREMENT_EXPIRED` triggers proactively!).
2. **Rapid Score Drift Missed Between Refreshes:** Even under `TrackAware` (`95.2%` agreement on `482f`), when depth is skipped for `120–160 ms`, true rapid cavity changes of `0.25–0.62` median (`1.19–1.93` P95 in `scale_norm_p75`) are not observed until the next refresh trigger (`PERIODIC_REFRESH`, `BOX_AREA_CHANGE`, or `LOW_BOX_OVERLAP`). Under `BudgetAware`, **`144` trigger requests (`482f`) were suppressed by `SKIPPED_BY_BUDGET`**, lowering matched Spearman from `0.9561` (`TrackAware`) to `0.8901`.
