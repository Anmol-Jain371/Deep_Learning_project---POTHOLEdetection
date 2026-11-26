# Phase 6A — Phase 5C Audit & Urban 720p (`raw_survey.mp4`) Failure Analysis

> [!IMPORTANT]
> Visual failure categories below are **provisional and unverified without manual pixel/physical ground-truth annotations**. Every metric represents relative inverse-depth behavior in [raw_survey_249_audit.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_urban_audit/raw_survey_249_audit.json) — not verified field depth accuracy.

---

## 1. Phase 5C Results Audit & Latency Clarification (Task 1)

Re-inspecting [shadow_depth_per_frame.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/shadow_depth_per_frame.csv) and [shadow_depth_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/shadow_depth_results.json) confirms:

- **Candidate Denominators Verified (`567` frames, `1,178` detections):** `ruralRoad_202f` (`690`), `rangoonRoad_180f` (`214`), `rawSurvey_720p_160f` (`249`), `testClip_25f` (`25`).
- **Resolution of `1.67 ms` vs `3.49 ms` Extraction Timing:**
  - The **`1.67 ms/region`** additive stage sum (`annular 0.40 + ref_fit 0.81 + cavity 0.26 + qa 0.20`) was measured strictly in standalone CPU isolation on `640×360` **`ruralRoad_202f`** frames (excluding `SpecularAquaplaneDetector`, temporal filtering, and `ShadowMeasurementResult` construction).
  - Across full shadow-mode runs including all 4 resolutions, per-sequence `opt_C_ms` averaged **`2.024 ms`** (`ruralRoad_202f`), **`4.196 ms`** (`rangoonRoad_180f`), **`7.021 ms`** (`rawSurvey_720p_160f`, where `1280×720` has `4×` larger annular pixel patches), and **`3.507 ms`** (`testClip_25f`), producing the corpus-weighted mean of **`3.49 ms/region`**.
- **Wall Throughput vs Additive Projections:** Actual observed multi-video shadow-mode wall FPS (`14.76` at `720p` to `16.93–19.88` at `360p`, including CSV logging + demo video encoding) reflects full concurrent `A+B+C+D` execution; standalone Method C estimates (`17.7–21.5 FPS` active, `68.3 FPS` empty-road) reflect additive projection without parallel branch or encoding overhead.

---

## 2. Diagnostic Breakdown of All 249 Urban `raw_survey.mp4` Detections (Task 2)

Evaluating all **249 YOLOv8n detections** (`160` frames, `1280×720`) under `MiDaS_small` (`rgb_corrected`) + Optimized RANSAC (`512` stratified):
- **Valid:** **`19 / 249` (`7.6%`)**
- **Invalid:** **`230 / 249` (`92.4%`)** (`unexpected_residual_polarity`: **180**, `low_inlier_ratio`: **58**, `suspected_water_reflection`: **57**).
- **Pre-Threshold Signed Residual Distribution (all 249 boxes, before any quality gating):**
  - `raw_p75`: `min = -28.271`, `p10 = -6.797`, `p25 = -5.818`, **`median = +0.006`**, `p75 = +1.138`, `p90 = +1.718`, `max = +4.222`.
  - `raw_p50` median: **`-2.465`**; `positive_depression_frac` median: **`0.251`**.
  - Trimming `20%` off bounding-box edges did **not** flip negative boxes positive globally: **`full_p75 > 0 = 125/249`** vs **`core20_p75 > 0 = 97/249`**.

### Provisional Classification Across the 249 Candidates

| Provisional Unverified Category | Count (`%`) | Median Conf | Median `raw_p50` | Median `raw_p75` | Median `pos_frac` | `core20_p75 > 0` | Representative Visual Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `boundary_clipping_or_edge` *(incl. road-shoulder FPs)* | `77` (`30.9%`) | `0.33` | `-19.60` | `-5.67` | `0.057` | `5/77` | [ex_boundary_clipping_or_edge.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_urban_audit/ex_boundary_clipping_or_edge.png) |
| `water_or_specular` | `57` (`22.9%`) | `0.34` | `+0.53` | `+1.06` | `0.595` | `57/57` | [ex_water_or_specular.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_urban_audit/ex_water_or_specular.png) |
| `reference_contamination_or_nonplanar` *(incl. flush gravel fill)* | `40` (`16.1%`) | `0.68` | `-12.95` | `-2.52` | `0.023` | `0/40` | [ex_reference_contamination_or_nonplanar.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_urban_audit/ex_reference_contamination_or_nonplanar.png) |
| `strong_negative_polarity_depth_failure` *(clear dry potholes)* | `34` (`13.7%`) | **`0.77`** | **`-10.01`** | **`-6.76`** | **`0.001`** | **`0/34`** | [ex_strong_negative_polarity_depth_failure.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_urban_audit/ex_strong_negative_polarity_depth_failure.png) |
| `shallow_low_contrast_or_flat` | `22` (`8.8%`) | `0.56` | `-1.01` | `+0.45` | `0.381` | `16/22` | [ex_shallow_low_contrast_or_flat.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_urban_audit/ex_shallow_low_contrast_or_flat.png) |
| `valid` | `19` (`7.6%`) | `0.76` | `+0.27` | `+1.75` | `0.555` | `19/19` | [ex_valid.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_urban_audit/ex_valid.png) |

---

## 3. Root Cause of `unexpected_residual_polarity` Failures

Evaluating the 5 hypotheses against direct quantitative evidence:
1. **Incorrect polarity formula (`Z_ref - Z`)?** **Ruled out.** Both verified `top_row < bot_row` geometry and cross-video valid cavities (`ruralRoad 63.9%`, synthetic tests) show `Z_ref - Z > 0` for depressions.
2. **Model output representation / curvature?** **Major contributor (`1/Z` warp + `MiDaS_small` failure).** On all **`77` high-confidence (`>= 0.60`) non-water urban rejections**, **`0 / 77` (`0%`)** became positive even inside a `20%`-trimmed center core (`core20_p75 > 0 = 0`). Visual inspection of `ex_strong_negative_polarity_depth_failure.png` (`conf = 0.78`, sharp dry rectangular asphalt cavity) shows **`MiDaS_small` predicts inverse depth higher (closer) across the entire darkened pit interior** (`p50 = -7.49`, `pos_frac = 0.006`), whereas Depth Anything V2 Small cleanly resolves that exact same cavity as positive (`VALID`, [panel10_rawSurvey_720p_f0_119_f0000_r0.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_viz/panel10_rawSurvey_720p_f0_119_f0000_r0.png)).
3. **Detector false positives / boundary clutter?** **Second major contributor.** All **`77` low-confidence (`conf < 0.35`) boxes** failed validation (`0/77` valid), exemplified by `ex_boundary_clipping_or_edge.png` (`conf = 0.267, w = 936.7 px`, grassy road shoulder).
4. **Cavity-region bounding-box edge contamination?** Minor factor in `raw_survey` (`0/77` high-conf non-water recovered by trimming); primarily affects only weak/shallow puddles (`shallow_low_contrast_or_flat`: `16/22` positive core, yet near noise floor).
