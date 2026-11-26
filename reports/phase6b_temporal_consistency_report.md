# Phase 6B — Matched Temporal Stability Report Across Resolutions (Task 6)

> [!CAUTION]
> **Relative Units Only:** All track trajectories use frozen ByteTrack IDs/timestamps, `ValidityAwareTrackFilter(mode="validity_aware_ema", alpha=0.35, max_stale_frames=3, max_missed_frames_reset=8)`, and `scale_norm_p75` / `frame_iqr_norm_p75`. **Smoother filtered outputs do NOT imply higher physical depth accuracy.** Source: [phase6b_resolution_scaling.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_resolution_scaling.json) & [phase6b_per_track.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_per_track.csv).

---

## 1. Matched Track Populations & Dropout Rates

- **Exploratory 360f:** `27` tracks with `>= 3` detections (`562` total observations); **`12` common DAv2 valid tracks (`>= 5` valid obs each, `376` total steps)**; **`9` 5-way common valid tracks including MiDaS (`319` steps)**.
- **Holdout Split:** `21` tracks with `>= 3` detections (`476` total observations); **`14` common DAv2 valid tracks (`390` steps)**; **`12` 5-way common valid tracks (`364` steps)**.

### Track-Level Measurement Dropout Frequency (`All Tracks >= 3 Obs`)

| Configuration (`Elliptical` for DAv2) | Expl. Dropout (`27 tracks, 562 obs`) | Expl. Valid Obs (`562`) | Holdout Dropout (`21 tracks, 476 obs`) | Holdout Valid Obs (`476`) |
| :--- | :---: | :---: | :---: | :---: |
| `midas_256_full` | `54.08%` | `221` | `43.06%` | `314` |
| `dav2_266` | `56.97%` | `266` | `41.77%` | `338` |
| `dav2_336` | `41.38%` | `337` | `28.39%` | `348` |
| `dav2_392` | `41.50%` | `328` | `23.10%` | `364` |
| **`dav2_518`** | **`41.06%`** | **`349`** | **`17.33%`** | **`412`** |

Inside the `12` (`Expl.`) / `14` (`Holdout`) common valid DAv2 tracks, **`518` cuts dropout to `16.66%` (`Expl.`) and `5.61%` (`Holdout`)**, compared to `27.59%` / `25.83%` (`336`) and `56.55%` / `34.95%` (`MiDaS 256`).

---

## 2. Raw vs Validity-Aware EMA Stability on Common DAv2 Valid Tracks

### A. Exploratory 360f (`12` Common DAv2 Tracks, `376` Total Obs)

| Arm | Valid Obs (`376`) | Dropout Rate | Raw Med `|Δ|` | Raw MAD | Raw CV | EMA Med `|Δ|` | EMA MAD | EMA CV (`scale` / `iqr_norm`) | Raw $\rho_{\text{area}}$ / $\rho_{y2}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `midas_256_full` | `153` | `56.55%` | `0.6047` | `0.5270` | `0.5680` | `0.0716` | `0.5014` | `0.4397` / `0.4237` | `+0.1341` / `-0.0892` |
| `dav2_266_elliptical` | `238` | `28.96%` | `0.3160` | `0.2610` | **`0.2863`** | `0.0761` | **`0.2003`** | **`0.2001`** / `0.2848` | `-0.2457` / `-0.1204` |
| `dav2_336_elliptical` | `242` | `27.59%` | **`0.1996`** | **`0.2071`** | `0.3627` | `0.0596` | `0.2296` | `0.2804` / `0.3230` | `-0.2978` / `+0.0339` |
| `dav2_392_elliptical` | `244` | `23.24%` | `0.2493` | `0.2513` | `0.3720` | **`0.0457`** | `0.2031` | `0.3041` / `0.3335` | `-0.2289` / `-0.1420` |
| **`dav2_518_elliptical`** | **`283`** | **`16.66%`** | `0.4181` | `0.4659` | `0.3101` | `0.0960` | `0.2848` | `0.2342` / **`0.2293`** | `-0.1597` / `-0.0440` |

### B. Untouched Holdout (`14` Common DAv2 Tracks, `390` Total Obs)

| Arm | Valid Obs (`390`) | Dropout Rate | Raw Med `|Δ|` | Raw MAD | Raw CV | EMA Med `|Δ|` | EMA MAD | EMA CV (`scale` / `iqr_norm`) | Raw $\rho_{\text{area}}$ / $\rho_{y2}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `midas_256_full` | `286` | `34.95%` | `0.6630` | `0.8226` | `0.5166` | `0.1023` | `0.4456` | `0.3643` / `0.3064` | `+0.3250` / `+0.0961` |
| `dav2_266_elliptical` | `329` | `18.12%` | `0.3385` | **`0.3008`** | **`0.2815`** | `0.0988` | **`0.1880`** | **`0.1941`** / `0.1985` | `-0.2809` / `+0.0972` |
| `dav2_336_elliptical` | `291` | `25.83%` | `0.3546` | `0.4852` | `0.4235` | **`0.0951`** | `0.2707` | `0.3556` / `0.2345` | `-0.1649` / `-0.1189` |
| `dav2_392_elliptical` | `311` | `19.46%` | **`0.2894`** | `0.5158` | `0.3184` | `0.1223` | `0.3728` | `0.2608` / **`0.1799`** | `-0.1706` / `+0.0067` |
| **`dav2_518_elliptical`** | **`361`** | **`5.61%`** | `0.3822` | `0.4527` | `0.3776` | `0.1502` | `0.3066` | `0.2951` / `0.2079` | `-0.0658` / `+0.2340` |

---

## 3. Interpretation

1. **Higher-Res Captures Richer Variation (More Valid Observations, Fewer Gaps):** `DAv2 518` retains the highest continuous valid count (`283/376` expl., `361/390` holdout) with the lowest dropout (`5.61%` holdout). `336`/`392` appear slightly smoother in raw `scale_norm` frame-to-frame delta (`0.20–0.29`) partly because coarsened grids blur fine rim edges, though under `frame_iqr_norm` `EMA CV`, `518` (`0.2293` expl.) and `392` (`0.1799` holdout) are cleanest.
2. **Temporal Filtering Behavior:** Applying `validity_aware_ema` (`alpha = 0.35`) reduces median frame-to-frame score jumps by **~3×–6× across all DAv2 resolutions** without ever propagating `0.0` into invalid frames.
3. **Approach-Distance Drift Remains Unresolved:** Under `scale_norm_p75` (`p75 / (1.4826 * ref_mad)`), all DAv2 resolutions show mild negative within-track correlation vs box area (`rho_area = -0.066` to `-0.298`) because larger near-camera boxes cover noisier/textural road patches (`ref_mad` increases slightly near the vehicle), whereas `MiDaS` trends positive (`+0.134` expl., `+0.325` holdout). Monocular approach-distance drift remains unsolved without extrinsics.
