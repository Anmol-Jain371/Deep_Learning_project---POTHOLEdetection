# Phase 6B — Depth Anything V2 Resolution Scaling & Pareto Report

> [!IMPORTANT]
> **Relative Inverse-Depth Scope Only:** Zero verified metric depth ground truth exists. **`DAv2 518` is treated strictly as a high-resolution model reference, NOT physical ground truth.** All residuals, cross-resolution scores, and provisional acceptance targets in [phase6b_resolution_scaling.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_resolution_scaling.json) measure relative inverse-depth representation behavior and RTX 2050 compute — **never calibrated metric depth in cm**.

---

## 1. Environment, Checkpoint & Frozen Evaluation Verification (Tasks 1 & 2)

- **Environment:** Windows 11, Python `3.12.10`, PyTorch `2.6.0+cu124`, CUDA `12.4`, **NVIDIA GeForce RTX 2050 (`4.0 GB` VRAM)**, OpenCV `5.0.0`, NumPy `2.5.4`.
- **Verified Checkpoints:**
  - `DepthAnythingV2Small` (`depth_anything_v2_vits.pth`): `99,218,434` bytes, SHA256 `715FADE13BE8F229F8A70CC02066F656F2423A59EFFD0579197BBF57860E1378`, `24,785,089` params (`fp16_autocast` baseline).
  - `YOLOv8n` (`runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt`): SHA256 `DDD6A5BE27AC21F3EF17FFE17FBAC5EA901CF2A7F7706956C5E551A669490343`.
- **Actual Observed Tensor Dimensions (`H × W`, never square):** Official DAv2 preprocessing (`Resize(input_size, lower_bound, ensure_multiple_of=14, INTER_CUBIC)`) maps both `640×360` and `1280×720` frames identically to:
  - `input_size=266` -> **`266 × 476`** (`19 × 34 = 646` ViT tokens)
  - `input_size=336` -> **`336 × 602`** (`24 × 43 = 1,032` tokens)
  - `input_size=392` -> **`392 × 700`** (`28 × 50 = 1,400` tokens)
  - `input_size=518` -> **`518 × 924`** (`37 × 66 = 2,442` tokens)
  - `MiDaS_small (256)` baseline -> **`128 × 256`**.
- **Frozen Extraction & Gate Config (`phase6b_v1_frozen_before_corpus`):** Identical YOLO+ByteTrack boxes/tracks/timestamps, cached per-frame depth predictions, `stratified_quadrant` (`512`, `16..128` RANSAC iters), `CoreExtractionConfig()` unchanged.

---

## 2. Controlled Benchmark Results Across Split Stages (Tasks 3 & 5)

### A. Stage A — 36 Development Frames (`98` Regions, With Elliptical vs Full-Box Controls)

| Configuration | Valid (`n = 98`) | Small Valid (`n = 29`) | `±8%` Box Perturb Retained | Median Rel. Shift |
| :--- | :---: | :---: | :---: | :---: |
| `midas_256_full` | `50` (`51.02%`) | `21/29` | `41/50` | `11.25%` |
| `dav2_266_elliptical` / `_full` | `66` (`67.35%`) / `65` (`66.33%`) | `25/29` / `25/29` | `61/66` / `60/65` | `8.86%` / `8.78%` |
| `dav2_336_elliptical` / `_full` | **`73` (`74.49%`)** / `71` (`72.45%`) | **`26/29`** / `26/29` | **`73/73`** / `70/71` | `7.04%` / `7.75%` |
| `dav2_392_elliptical` / `_full` | `70` (`71.43%`) / `69` (`70.41%`) | `21/29` / `21/29` | `63/70` / `65/69` | `7.22%` / `7.98%` |
| `dav2_518_elliptical` / `_full` | `71` (`72.45%`) / `68` (`69.39%`) | `23/29` / `22/29` | `69/71` / `65/68` | **`5.76%`** / `5.86%` |

### B. Stage B (Exploratory `360f` = `575` Regions) & Stage C (Separate Holdout `327f` = `481` Regions)

The holdout split combines genuinely depth-untouched `rangoonRoad_f785_1004` (`74` detections) plus Phase-6A-unseen `ruralRoad_f120_201` (`382` detections) and `testClip_f0_24` (`25` detections), evaluated strictly under frozen settings without tuning.

| Configuration | Expl. `360f` (`n=575`) | Expl. Full-Box Control | Urban `rawSurvey` (`n=154`) | Rural `f0..119` (`n=309`) | Holdout Total (`n=481`) | Holdout `rangoon_f785` (`n=74`) | Holdout `testClip` (`n=25`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `midas_256_full` | `226` (`39.30%`) | `—` | `19` (`12.34%`) | `180` (`58.25%`) | `316` (`65.70%`) | `48` (`64.86%`) | `3` (`12.00%`) |
| `dav2_266_elliptical` | `272` (`47.30%`) | `297` (`51.65%`) | `68` (`44.16%`) | `156` (`50.49%`) | `340` (`70.69%`) | `51` (`68.92%`) | `1` (`4.00%`) |
| `dav2_336_elliptical` | **`343` (`59.65%`)** | `320` (`55.65%`) | `104` (`67.53%`) | `192` (`62.14%`) | `350` (`72.77%`) | **`67` (`90.54%`)** | `8` (`32.00%`) |
| `dav2_392_elliptical` | `335` (`58.26%`) | `338` (`58.78%`) | **`105` (`68.18%`)** | `200` (`64.72%`) | `366` (`76.09%`) | `65` (`87.84%`) | `3` (`12.00%`) |
| **`dav2_518_elliptical`** | **`355` (`61.74%`)** | `333` (`57.91%`) | `80` (`51.95%`) | **`224` (`72.49%`)** | **`414` (`86.07%`)** | `57` (`77.03%`) | **`12` (`48.00%`)** |

*Note:* Higher urban coverage at `336`/`392` (`67.5%`/`68.2%`) than `518` (`51.95%`) reflects `75–90` borderline/low-confidence urban boxes where `518` predicts negative polarity while coarser grids smooth boundary glare/rubble (`native_518_invalid_when_lowres_valid: 90 unexpected_residual_polarity` at `336`); conversely `336` loses `102` regions that were valid at `518`.

---

## 3. Representation Agreement vs Native `DAv2 518` Model Reference (Task 3)

All `MAE`/`RMSE` metrics apply least-squares affine alignment ($d_{518} \approx a \cdot d_r + b$, avoiding raw unscaled comparison).

| Comparator (`Elliptical`) | Within-Image Spearman Med (P10) | Affine-Aligned MAE / nRMSE(IQR) | Expl. Validity Agreement ($\kappa$) | Confusion (`ValidBoth / InvBoth / Lost / Gained`) | Matched Valid Spearman (`scale_norm` / `iqr_norm`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dav2_266` (`expl`) | `0.9994` (`0.9964`) | `0.0766` / `0.0224` | `66.09%` (`0.3302`) | `216 / 164 / 139 / 56` | `0.5270` / `0.6521` |
| `dav2_336` (`expl`) | `0.9995` (`0.9976`) | `0.0596` / `0.0177` | `66.61%` (`0.3005`) | `253 / 130 / 102 / 90` | `0.1954` / `0.7027` |
| `dav2_392` (`expl`) | **`0.9997` (`0.9986`)** | **`0.0441` / `0.0124`** | **`70.43%` (`0.3848`)** | `260 / 145 / 95 / 75` | `0.4633` / **`0.8287`** |
| `dav2_392` (`holdout`) | `0.9998` (`0.9996`) | `0.0415` / `0.0122` | `81.70%` (`0.4132`) | `346 / 47 / 68 / 20` | `0.5542` / `0.7997` |

- **Key Asymmetry Found:** Whole-frame ordinal depth ordering (`within_image_spearman >= 0.9994`) is almost identical across resolutions, yet **localized pothole boundary/plane residuals diverge substantially** (`validity_agreement = 66.1%–70.4%` on exploratory, `75.1%–81.7%` on holdout; `matched scale_norm Spearman = 0.20–0.55`).
- **Native `DAv2 518` Own Failures:** On exploratory `575` regions, `518` itself rejected **`220` (`38.26%`)** (`unexpected_residual_polarity=189`, `water=33`, `low_inlier=19`), including **`60` high-confidence (`>= 0.65`) non-water boxes**. Visual inspection of [phase6b_panel_shadowed_rangoon_f0211.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_viz/phase6b_panel_shadowed_rangoon_f0211.png) confirms even `518` predicts negative cavity residuals across shallow shadowed cracks.

---

## 4. Pareto Comparison & Frozen Acceptance Policy Verdict (Tasks 9 & 10)

Charts: [phase6b_res_vs_coverage_and_latency.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_viz/phase6b_res_vs_coverage_and_latency.png), [phase6b_pareto_and_scaling.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_viz/phase6b_pareto_and_scaling.png), [phase6b_panel_urban_dry_pit_f0000.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_viz/phase6b_panel_urban_dry_pit_f0000.png), [phase6b_panel_water_puddle_f0021.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_viz/phase6b_panel_water_puddle_f0021.png).

| Config | Expl. Valid | Holdout Valid | Cov Ret. (`>=0.90`) | Matched $\rho$ (`>=0.85`) | Agree (`>=88%`) | Urban Ret. (`>=0.85`) | Small Ret. (`>=0.85`) | Depth Mean | Active E2E Mean (FPS) | `<= 45ms` RT? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `midas_256_full` | `39.30%` | `65.70%` | `63.6%` | `—` | `—` | `23.8%` | `64.8%` | `24.85 ms` | **`44.41 ms` (`22.52`)** | **Yes** |
| `dav2_266_elliptical` | `47.30%` | `70.69%` | `76.6%` ❌ | `0.5270` ❌ | `66.1%` ❌ | `85.0%` ✅ | `87.7%` ✅ | `37.45 ms` | `58.10 ms` (`17.21`) | ❌ |
| `dav2_336_elliptical` | `59.65%` | `72.77%` | **`96.6%`** ✅ | `0.1954` ❌ | `66.6%` ❌ | **`130.0%`** ✅ | **`100.0%`** ✅ | `64.08 ms` | `84.68 ms` (`11.81`) | ❌ |
| `dav2_392_elliptical` | `58.26%` | `76.09%` | `94.4%` ✅ | `0.4633` ❌ | `70.4%` ❌ | **`131.2%`** ✅ | `96.7%` ✅ | `88.79 ms` | `109.45 ms` (`9.14`) | ❌ |
| `dav2_518_elliptical` | `61.74%` | `86.07%` | `100%` | `1.000` | `100%` | `100%` | `100%` | `177.76 ms` | `198.50 ms` (`5.04`) | ❌ |

---

## 5. Final Recommendation

### Selected Outcome: **B — DAv2 requires accelerated inference or selective keyframe scheduling before further integration**

- **Why Outcome A is rejected:** No lower resolution meets both coverage fidelity and real-time RTX 2050 throughput. `dav2_266` drops exploratory coverage to `47.30%` (`< 90%` retention) and still runs below real-time (`58.10 ms` / `17.21 FPS`). `dav2_336` preserves coverage (`59.65%` exploratory, `67.53%` urban), yet fails local rank/validity agreement vs `518` (`0.1954` Spearman, `66.6%` agreement), trails `518` on holdout (`72.77%` vs `86.07%`), and only reaches **`11.81 FPS`** (`84.68 ms` E2E; `~13.6 FPS` even with pure `fp16_weights`).
- **Why Outcome B is chosen (while retaining `MiDaS 256` untouched in production for now):** `MiDaS 256` collapses on urban `rawSurvey` (`12.34%` vs `51.95%–68.18%` for DAv2), making simple MiDaS retention unsatisfactory for urban roads. Meanwhile, **`DAv2 336` (best coverage-per-ms knee) or `DAv2 518` (strongest holdout generalization `86.07%`)** can become viable in Phase 7 via **selective keyframe/track-triggered scheduling** (running YOLO+ByteTrack at `65.5 FPS` / `15.26 ms` per frame while running DAv2 on every $K$-th active frame or offline) or **TensorRT FP16 / ONNX GPU** on a CUDA-matched environment.
- **Remaining Limitations:** No physical metric depth ground truth; `518` itself rejects `38.3%` of exploratory candidates; approach-distance drift persists without camera extrinsics; production deployment stays paused pending approval.
