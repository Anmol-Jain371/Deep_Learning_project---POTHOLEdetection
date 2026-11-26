# Phase 7A — Expanded 36-Frame FP16 Numerical Regression Report (Task 2)

Source: [phase7a_fp16_regression.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_fp16_regression.json), executed via [dav2_inference_optimization.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/dav2_inference_optimization.py) on **NVIDIA RTX 2050 (`4.0 GB` VRAM)**.

---

## 1. Benchmark Coverage (`36` Diverse Dev Frames, `98` Detected Potholes)

Evaluated across all 36 development frames spanning urban `raw_survey` (`1280×720`), unseen `ruralRoad`, shadowed `Rangoon`, multi-pothole scenes, small/distant boxes, wet/specular reflections, and `phone_val` frames (`640×360`), comparing **`FP32` vs `fp16_autocast` vs pure `fp16_weights` (`model.half()`)** across `DAv2 Small` `266`, `336`, and `518`.

---

## 2. Numerical Stability, Cavity Residuals & Local Validity Comparison

| Config | Mode vs `FP32` | Non-Finite Frames | Whole-Image Spearman Mean / Min | Affine MAE | Signed Cavity `p75` Abs Diff Med / P95 | Validity Flips (`Lost/Gained`) | Matched Valid `sn_p75` Spearman (`n`) | Matched `sn_p75` Rel Diff Med / P95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DAv2 266`** | `fp16_autocast` | `0/36` | `0.99989` / `0.99724` | `0.00126` | `0.00092` / `0.01122` | `2/98` (`0/2`) | `0.9941` (`64`) | `2.18%` / `13.41%` |
| `DAv2 266` | `fp16_weights` | `0/36` | `0.99988` / `0.99711` | `0.00131` | `0.00078` / `0.01458` | `4/98` (`2/2`) | `0.9775` (`62`) | `2.72%` / `23.62%` |
| **`DAv2 336`** | `fp16_autocast` | `0/36` | `0.99990` / `0.99775` | `0.00124` | `0.00085` / `0.01333` | `3/98` (`0/3`) | `0.9695` (`70`) | `2.77%` / `19.88%` |
| **`DAv2 336`** | **`fp16_weights`** | `0/36` | `0.99990` / `0.99827` | `0.00127` | **`0.00079`** / `0.01238` | `3/98` (`0/3`) | **`0.9922` (`70`)** | `2.80%` / **`15.55%`** |
| **`DAv2 518`** | `fp16_autocast` | `0/36` | `0.99996` / `0.99963` | `0.00163` | `0.00092` / `0.00704` | `2/98` (`0/2`) | `0.9849` (`69`) | `4.28%` / `23.55%` |
| `DAv2 518` | `fp16_weights` | `0/36` | `0.99996` / `0.99971` | `0.00165` | `0.00082` / `0.00890` | `4/98` (`2/2`) | `0.9906` (`67`) | `3.97%` / `15.91%` |

---

## 3. Latency & Isolated VRAM Scaling (`36` Frames)

| Config | Mode | GPU Infer Mean / P95 (ms) | Total Depth Mean / P95 (ms) | Peak Alloc / Reserved VRAM |
| :--- | :--- | :---: | :---: | :---: |
| `DAv2 266` (`266×476`) | `fp32` / `fp16_autocast` / **`fp16_weights`** | `30.59/31.04` \| `27.67/32.92` \| **`20.89/27.79`** | `51.02` \| `46.80` \| **`39.89` (`52.28` P95)** | `349` \| `410` \| **`335.0 MB` (`356`)** |
| **`DAv2 336`** (`336×602`) | `fp32` / `fp16_autocast` / **`fp16_weights`** | `61.05/62.94` \| `50.43/57.38` \| **`42.25/54.28`** | `92.21` \| `78.25` \| **`72.90` (`125.1` P95)** | `386` \| `444` \| **`364.2 MB` (`404`)** |
| `DAv2 518` (`518×924`) | `fp32` / `fp16_autocast` / **`fp16_weights`** | `179.8/180.8` \| `147.2/148.4` \| **`109.7/110.8`** | `228.1` \| `191.5` \| **`155.2` (`188.8` P95)** | `613` \| `584` \| **`466.3 MB` (`572`)** |

---

## 4. Verification Verdict

- **Why Whole-Frame Spearman Alone Would Mislead:** While whole-image ordinal rank stays `>= 0.997`, expanding verification from 4 frames (`0/4` flips) to `36` frames (`98` regions) uncovered **`3/98–4/98` (`3.1%–4.1%`) boundary validity flips** and **`13.4%–23.6%` P95 local `scale_norm_p75` shifts** on near-threshold planar surfaces (`ref_mad` denominator amplification).
- **Acceptance Decision:** **`fp16_weights` (`model.half()`) at `DAv2 336`** is accepted for experimental Phase 7A scheduling (`0` non-finite frames, `3/98` flips matching `autocast`, `0.9922` matched `sn_p75` Spearman, `15.55%` P95 local diff, **`42.25 ms` GPU infer** vs `61.05 ms` FP32, **`364.2 MB`** VRAM), with boundary-flip variance explicitly flagged.
