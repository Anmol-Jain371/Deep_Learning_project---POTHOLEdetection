# Phase 5C MiDaS Preprocessing & Adaptive Sampling Ablation Report

Executed via [benchmark_shadow_depth_pipeline.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_shadow_depth_pipeline.py) (`Task 3` & `Task 7`) and plotted in [shadow_preprocessing_ablation.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase5c_shadow_viz/shadow_preprocessing_ablation.png).

---

## 1. Authoritative MiDaS Input Contract Verification (Task 3)

Direct inspection of the cached Intel ISL MiDaS repository (`hubconf.py` lines 327–343 & `midas/transforms.py` lines 197–225) confirms:
- `transforms.small_transform` performs `[img / 255.0 -> Resize(256, 256, upper_bound, INTER_CUBIC) -> NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> PrepareForNet()]`.
- `NormalizeImage` expects **RGB channel order** matching ImageNet statistics (`R` mean `0.485`, `G` mean `0.456`, `B` mean `0.406`), and does **not** perform any internal BGR→RGB transpose.
- Production [roadvision_engine.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_engine.py#L656) passes raw OpenCV `BGR` frames (`self.depth_transform(frame)`) directly into MiDaS, swapping the Red and Blue channels prior to ImageNet normalization.

---

## 2. Quantified Preprocessing Divergence & 2×2 Factorial Ablation (`32` Active Dev Frames, `98` Regions)

### Full-Frame Depth Map Shift (`bgr_legacy` vs `rgb_corrected`)

| Metric | Observed Value (`n=32` active frames) |
| :--- | :---: |
| Preprocess Time Mean: `bgr_legacy` vs `rgb_corrected` | `6.3004 ms` vs `6.8005 ms` (`+0.50 ms` for `cvtColor`) |
| GPU Inference Time Mean: `bgr_legacy` vs `rgb_corrected` | `19.4482 ms` vs `19.6366 ms` |
| Pixelwise Inverse-Depth Mean L1 / RMSE | **`32.8659` / `39.6308` relative units** |
| Subsampled Frame Pearson $r$ / Spearman $\rho$ | `0.9953` / `0.9960` |

### 2×2 Factorial Results (Preprocessing × Extraction Method)

| Cell | Valid Count (`n=98`) | Admitted Ref MAD (Med) | Admitted P75 (Med) | Common-All-4 (`n=19`) Ref MAD / P75 (Med) | Mean Ext Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `bgr_legacy` + `legacy_annular` | `41` (`41.84%`) | `1.8842` | `3.1102` | `1.8320` / `3.1500` | `3.8628` |
| `rgb_corrected` + `legacy_annular` | `39` (`39.80%`) | `2.6520` | `6.1245` | `2.6520` / `5.4359` | `3.9030` |
| `bgr_legacy` + `optimized_ransac` | `40` (`40.82%`) | `1.8293` | `6.0277` | `1.1044` / `3.9450` | `2.8332` |
| **`rgb_corrected` + `optimized_ransac`** | **`50` (`51.02%`)** | `2.7359` | `8.6276` | `1.9568` / `7.6321` | `2.8005` |

- **Paired Single-Factor Shifts (Same Extractor, BGR vs RGB):**
  - Holding `legacy_annular` fixed (`n=36` common valid pairs): median `|ΔP75|` shift = **`2.2968` relative units**.
  - Holding `optimized_ransac` fixed (`n=31` common valid pairs): median `|ΔP75|` shift = **`2.8865` relative units** (`10.2%` jump in valid coverage from `40.82%` to `51.02%` under RGB correction!).

> [!IMPORTANT]
> **Scientific Boundary:** Correcting OpenCV `BGR -> RGB` brings MiDaS preprocessing into conformance with its pretrained ImageNet input specification at negligible cost (`+0.50 ms`), substantially shifting relative inverse-depth contrast (`+10.2%` valid RANSAC acceptance). However, without physical ground truth, depth-map shifts **cannot be interpreted as verified metric depth accuracy improvement**.

---

## 3. Optional Adaptive Sampling Investigation: Fixed `512` vs Triggered `512 -> 1024` (Task 7)

Tested escalation rule `should_escalate_budget(out_512)`: re-sample at `1024` when `available_ref_samples > 1024`, non-degenerate plane, and `(inlier_ratio < 0.65 or populated_quadrants < 3)`.

- **Synthetic Multi-Seed Ablation (`27` valid scenarios across seeds `101, 202, 303`):**
  - Adaptive triggers fired: **`0 / 27`** (because synthetic noise $\sigma \le 0.40$ falls inside local finite-difference consensus threshold -> `inlier_ratio >= 0.98`, leaving `high_noise` P75 error unchanged at **`0.00651 -> 0.00651`**).
- **567-Frame Video Stream Ablation (`1,178` candidate detections):**
  - Escalation triggered on **`513 / 1,178` regions (`43.55%`)**.
  - Valid coverage shifted negligibly from **`43.29%` (`510`) to `43.46%` (`512`)**.
  - Matched `n=442` ref MAD median remained flat (`2.6068 -> 2.6101`).
  - Smoothed track CV improved slightly (`0.3423 -> 0.2595`) at the cost of **`+60.5%` mean extraction latency (`3.49 ms -> 5.60 ms`, `p95: 8.67 -> 16.90 ms`)**.
- **Conclusion:** **Reject optional `512 -> 1024` escalation as default**; it doubles outlier-patch work (`43.55%` trigger rate) without resolving the high-noise synthetic variance case or improving real valid coverage enough to justify its latency penalty.
