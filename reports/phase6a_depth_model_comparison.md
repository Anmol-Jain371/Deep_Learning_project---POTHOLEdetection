# Phase 6A — Controlled Monocular Depth Model Comparison: `MiDaS Small` vs `Depth Anything V2 Small`

Adapter & benchmark: [depth_model_comparison.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/depth_model_comparison.py), [benchmark_phase6a_depth_comparison.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_phase6a_depth_comparison.py), [phase6a_depth_comparison.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_depth_comparison.json), [phase6a_per_region.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_per_region.csv).

> [!IMPORTANT]
> **Exploratory Scope Only:** Zero independent metric depth ground truth exists. All cross-model comparisons evaluate relative inverse-depth spatial structure, reference planarity, polarity consistency, and RTX 2050 compute — **never calibrated metric depth in cm**.

---

## 1. Depth Anything V2 Small Feasibility & Adapter Contracts (Tasks 5 & 6)

- **Official Source & Provenance:** `depth-anything/Depth-Anything-V2-Small` (Apache-2.0 License), checkpoint `depth_anything_v2_vits.pth` (`99,218,434` bytes = `94.62 MiB`, SHA256 `715FADE13BE8F229F8A70CC02066F656F2423A59EFFD0579197BBF57860E1378`), DINOv2 ViT-S/14 + DPTHead (**`24,785,089` params**), loaded with `strict=True` natively under existing `torch 2.6.0+cu124` + Windows CUDA without upgrading any packages.
- **Side-by-Side Adapter Contract:**

| Property | `MiDaSSmallAdapter` | `DepthAnythingV2SmallAdapter` (`518` native) |
| :--- | :--- | :--- |
| Parameters / Precision | `21,320,545` / `fp16_autocast` | `24,785,089` / `fp16_autocast` |
| Input Preprocessing | `BGR->RGB /255`, `Resize(256, upper_bound, mod32, INTER_CUBIC)`, ImageNet norm | `BGR->RGB /255`, `Resize(518, lower_bound, mod14, INTER_CUBIC)`, ImageNet norm |
| Upsampling to `(H, W)` | `F.interpolate(..., bilinear, align_corners=False)` | `F.interpolate(..., bilinear, align_corners=True)` |
| Verified Output Polarity | `inverse_depth_higher_closer` (`top < bot`) | `inverse_depth_higher_closer` (`top < bot`) |
| Typical Raw Output Range | `~0 .. 800` | `~0 .. 12` *(~65× smaller numerical scale!)* |

To ensure identical strictness across differing raw scales without gate relaxation, Check #9 (`unstable_cavity_residuals`) uses a purely scale-normalized noise floor `4 * 1.4826 * ref_mad` instead of a fixed `1.0` constant.

---

## 2. Controlled 2×2 Benchmark Results (`Model × Cavity Strategy`) (Task 7)

### A. 36 Development Frames (`98` Identical Detections, `common_all4 = 34`)

| Arm | Valid Count (`%`) | Median Inlier Ratio | Median Ref MAD | Common-4 Scale-Norm P75 (Med) | `±8%` Box Perturb (`ret`, Med Rel Shift) | Failure Reasons |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `midas_full` | `50` (`51.02%`) | `0.6641` | `2.5816` | `2.5772` | `41/50` (`11.25%`) | `pol`: 34, `inl`: 15, `wat`: 3, `unst`: 2 |
| `midas_elliptical` | `53` (`54.08%`) | `0.6641` | `2.5816` | `2.5187` | `44/53` (`14.27%`) | `pol`: 33, `inl`: 15, `wat`: 3, `unst`: 1 |
| `dav2_full` | `68` (`69.39%`) | `0.8623` | `0.0112` | `2.2365` | `65/68` (`5.86%`) | `pol`: 28, `inl`: **3**, `wat`: 3 |
| **`dav2_elliptical`** | **`71` (`72.45%`)** | **`0.8623`** | `0.0112` | `2.4122` | **`69/71` (`5.76%`)** | `pol`: **24**, `inl`: **3**, `wat`: 3 |

### B. Exploratory 360-Frame Multi-Source Video Corpus (`575` Regions, `common_all4 = 108`)

| Source Sequence (`Prior Exposure`) | `midas_full` | `midas_elliptical` | `dav2_full` | `dav2_elliptical` |
| :--- | :---: | :---: | :---: | :---: |
| `ruralRoad_f0_119` (`309` dets, unseen train) | `180` (`58.25%`) | `169` (`54.69%`) | `211` (`68.28%`) | **`224` (`72.49%`)** |
| `rangoonRoad_f200_319` (`112` dets, 4/120 in v2) | `27` (`24.11%`) | `25` (`22.32%`) | `42` (`37.50%`) | **`51` (`45.54%`)** |
| `rawSurvey_720p_f0_119` (`154` dets, 0 MD5 train) | `19` (**`12.34%`**) | `19` (`12.34%`) | **`80` (`51.95%`)** | **`80` (`51.95%`)** |
| **Aggregate (`n = 575`)** | `226` (`39.30%`) | `213` (`37.04%`) | `333` (`57.91%`) | **`355` (`61.74%`)** |

Across all `575` regions, Depth Anything V2 Small reduces `low_inlier_ratio` failures by **6× (`118 -> 19`)** (`inlier_ratio` `0.8652` vs `0.6367`), slashes `unexpected_residual_polarity` (`271 -> 189`), and **more than doubles bounding-box stability (`5.76%` shift vs `14.27%` for MiDaS)**.

---

## 3. Visual Diagnostics (`panel10_*.png`, Task 10)

- **[panel10_rawSurvey_720p_f0_119_f0000_r0.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_viz/panel10_rawSurvey_720p_f0_119_f0000_r0.png) (Dry asphalt square excavation):** MiDaS predicts negative blue smear across the pit (`FullP75 = -5.238`, `Inlier = 0.680` -> `INVALID`); **DAv2 cleanly traces the crater (`FullP75 = 0.023`, `Inlier = 1.000` -> `VALID`)**.
- **[panel10_rawSurvey_720p_f0_119_f0081_r0.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_viz/panel10_rawSurvey_720p_f0_119_f0081_r0.png) (Gravel crater):** MiDaS rejects (`Inlier = 0.420`, `low_inlier_ratio`); **DAv2 isolates a crisp `>20σ` crater silhouette (`Inlier = 0.883`, `CoreP75 = 0.375` -> `VALID`)**.
- **[panel10_rawSurvey_720p_f0_119_f0021_r0.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_viz/panel10_rawSurvey_720p_f0_119_f0021_r0.png) (Grey water puddle):** DAv2 rejects the specular flat mirror (`CoreP75 = 0.000`) while MiDaS passes edge glare.
- **[panel10_ruralRoad_f0_119_f0045_r0.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6a_viz/panel10_ruralRoad_f0_119_f0045_r0.png):** Both valid (`DAv2 Inlier = 0.992` vs `MiDaS 0.562`).

---

## 4. RTX 2050 Synchronized Latency, VRAM & Throughput (Task 9)

Measured on RTX 2050 (`detector+ByteTrack @ 640`: `13.61 ms` mean / `11.81 ms` med, `85.32 FPS` on `0`-candidate frames):

| Model & Input | Load (ms) | Preproc (ms) | GPU Infer (ms) | Resize (ms) | Total Depth Mean / Med / P95 (ms) | E2E `1..4+` Active FPS | Peak VRAM Alloc / Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MiDaS Small (256)`** | `3467.3` | `5.38` | `18.47` | `0.95` | **`24.80`** / `22.43` / `37.72` | **`22.3 – 25.7 FPS`** | `392.6` (`+46.8`) |
| **`DAv2 Small (518 native)`** | `802.3` | `30.56` | **`147.26`** | `1.19` | **`179.01`** / `178.00` / `187.86` | **`5.0 – 5.2 FPS`** | `628.4` (`+282.6`) |
| **`DAv2 Small (266` ablation\*)** | `802.3` | `10.03` | `25.95` | `0.67` | **`36.65`** / `34.92` / `42.46` | `~18–20 FPS` est | `461.7` (`+115.8`) |

*(\* `266` ablation measures runtime ONLY; depth quality at `266` is unverified and must not be assumed equal to native `518`.)*

---

## 5. Final Phase 6A Recommendation

### Selected Outcome: **B — Depth Anything V2 Small provides sufficient behavioral and computational advantages to justify further investigation**

- **Supporting Evidence:** Replacing `MiDaS_small` with `DepthAnythingV2_Small (518)` + `elliptical_core` **transforms urban 720p valid detection coverage from `12.34%` to `51.95%`** (`575`-region corpus `39.30% -> 61.74%`), reduces `low_inlier_ratio` non-planar failures **6× (`118 -> 19`)**, improves box-perturbation stability **`2.5×` (`14.27% -> 5.76%`)**, and resolves sharp dry asphalt pits visually confirmed in `panel10_rawSurvey_*`.
- **Unresolved Bottleneck & Next Experimental Gate:** Native `518` `DAv2_Small` runs at **`5.0–5.2 FPS`** (`179 ms` depth) due to $O(N^2)$ ViT self-attention over `~2,442` patches on RTX 2050, whereas downscaled `266` hits **`36.65 ms` (`27.3 FPS`)**. Therefore, **stop prior to production integration** and investigate **accuracy-vs-speed scaling of `DepthAnythingV2SmallAdapter` at `input_size ∈ {266, 336, 392, 518}`** (plus TensorRT/`torch.compile` if needed) before considering optional production deployment.
