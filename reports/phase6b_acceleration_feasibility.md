# Phase 6B — RTX 2050 GPU Performance & Acceleration Feasibility Report (Tasks 7 & 8)

Hardware: **NVIDIA GeForce RTX 2050 (`4.0 GB` VRAM)**, Windows 11, Python `3.12.10`, PyTorch `2.6.0+cu124`, CUDA `12.4`. Sources: [dav2_inference_optimization.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/dav2_inference_optimization.py), [phase6b_resolution_scaling.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase6b_resolution_scaling.json).

---

## 1. Synchronized Latency, Proposal Scaling (`0..4+`), and Isolated VRAM (Task 7)

Baseline detector + ByteTrack (`640`): **`15.26 ms` mean / `12.27 ms` median / `20.88 ms` P95 (`65.53 FPS`)** (`12.53 ms` = `79.80 FPS` on `0`-proposal frames). Memory stats were recorded in a strictly separated pass from runtime loops.

| Model (`fp16_autocast`) | Preproc Mean (ms) | GPU Infer Mean (ms) | Resize Mean (ms) | Depth Total Mean / P95 (ms) | Active Extract Mean (ms) | Active E2E Mean / Med / P95 (ms) | Active E2E FPS | Peak Alloc / Reserved (`720p`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`midas_256`** | `5.30` | `18.59` | `0.96` | `24.85` / `37.87` | `3.58` | **`44.41` / `40.30` / `59.14`** | **`22.52 FPS`** | `592.7 MB` / `620 MB` |
| **`dav2_266`** | `10.89` | `25.50` | `1.07` | `37.45` / `49.19` | `4.67` | `58.10` / `54.39` / `76.77` | `17.21 FPS` | `661.7 MB` / `692 MB` |
| **`dav2_336`** | `14.89` | `48.08` | `1.11` | `64.08` / `72.12` | `4.62` | `84.68` / `80.53` / `103.82` | `11.81 FPS` | `694.4 MB` / `734 MB` |
| **`dav2_392`** | `18.84` | `68.72` | `1.22` | `88.79` / `97.06` | `4.68` | `109.45` / `105.14` / `128.75` | `9.14 FPS` | `735.6 MB` / `790 MB` |
| **`dav2_518`** | `29.76` | **`146.80`** | `1.20` | **`177.76`** / `187.07` | `4.76` | `198.50` / `194.77` / `217.94` | `5.04 FPS` | `828.7 MB` / `950 MB` |

- **Proposal Count Scaling (`0 -> 1 -> 2 -> 3 -> 4+`):** Depth inference cost is invariant to box count (`e.g. dav2_336`: `12.53 ms [0]` -> `84.04 ms [1]` -> `87.99 ms [2]` -> `83.54 ms [3]` -> `79.11 ms [4+]`, because annular extraction adds only `~2.3 ms` median / `4.6 ms` mean). Thus **full-frame ViT depth attention — not RANSAC extraction — bottlenecks active FPS**.

---

## 2. Precision Comparison: `FP32` vs `fp16_autocast` vs Pure `fp16_weights` (`model.half()`)

Measured with `torch.inference_mode()` across `{266, 336, 392, 518}` on 4 matched frames:
- **Numerical Stability:** **`0/4` non-finite frames** and **`0/4` validity flips** vs `FP32` across every resolution (`Spearman >= 0.99977`, affine `MAE <= 0.00167`, `max_raw_abs_diff <= 0.0293`).
- **Pure `fp16_weights` (`model.half()`) Gains Over `fp16_autocast` (Zero Dependencies Required):**

| `DAv2` Size | `FP32` Depth Total (Infer) | `fp16_autocast` Depth Total (Infer) | `fp16_weights` Depth Total (Infer) | `fp16_weights` VRAM Peak Alloc | Est. Active E2E (`fp16_weights`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `266` (`266×476`) | `45.55 ms` (`30.36`) | `35.73 ms` (`24.45`) | **`28.79 ms` (`17.72 ms`)** | `550.2 MB` *(vs `623.8 MB`)* | `~48.7 ms` (`~20.5 FPS`) |
| `336` (`336×602`) | `78.60 ms` (`60.24`) | `63.62 ms` (`48.52`) | **`53.57 ms` (`38.65 ms`)** | `579.9 MB` *(vs `659.0 MB`)* | `~73.5 ms` (`~13.6 FPS`) |
| `392` (`392×700`) | `107.15 ms` (`84.60`) | `86.72 ms` (`68.57`) | **`72.92 ms` (`54.51 ms`)** | `608.4 MB` *(vs `702.5 MB`)* | `~92.9 ms` (`~10.8 FPS`) |
| `518` (`518×924`) | `213.28 ms` (`181.65`) | `174.27 ms` (`146.72`) | **`137.53 ms` (`109.06 ms`)** | `681.2 MB` *(vs `801.2 MB`)* | `~157.5 ms` (`~6.3 FPS`) |

Pure `model.half()` + `inference_mode` cuts GPU inference latency by **~20–27%** and lowers peak VRAM **12–15%** with zero package changes, yet `336` (`~13.6 FPS`) and `266` (`~20.5 FPS`) still fall short of `MiDaS 256` (`22.5 FPS`) on synchronous per-frame inference.

---

## 3. Optional Acceleration Backend Feasibility (Task 8)

1. **`torch.compile` (`inductor` / `cudagraphs` / `aot_eager` at `336`):**
   - `inductor`: **NOT SUPPORTED** on Windows (`RuntimeError: Cannot find a working triton installation`).
   - `cudagraphs`: Compiled in `4,329.9 ms`, numerically sound (`Spearman = 0.99993`), but ran at **`90.84 ms` steady-state** (**`1.89×` SLOWER** than eager `48.03 ms`) due to dynamo wrapper overhead and skipped CUDAGraph fast paths.
   - `aot_eager`: Compiled in `3,703.6 ms`, ran at **`92.15 ms` steady-state** (**`1.92×` SLOWER** than eager).
   - **Verdict:** Reject `torch.compile` on this Windows PyTorch setup.

2. **ONNX Export (`opset=17`) & ONNX Runtime (`1.32.0`) Audit:**
   - **Dynamic Input-Shape Compatibility (`FAILED` for single multi-res graph):** Exporting DAv2 at `336×602` with `dynamic_axes` succeeds (`94.41 MB`, `onnx.checker` passes, `336` ORT matches `MAE = 9.3e-5`, `Spearman = 0.99992`), **but fails at `266`, `392`, and `518`** inside DINOv2 positional-embedding `/Add` (`BroadcastIterator` mismatch `647 vs 1033`, `1401 vs 1033`, `2443 vs 1033`) because traced `interpolate_pos_encoding` freezes token count as a constant.
   - **Per-Resolution Static ONNX (`PASSED` export & parity):** Separate static ONNX graphs at `266×476` (`94.36 MB`, `1,796 ms` export, `MAE = 1.28e-4`, `Spearman = 0.99951`) and `336×602` (`94.36 MB`, `1,961 ms` export, `MAE = 9.3e-5`, `Spearman = 0.99992`) verify cleanly on `CPUExecutionProvider` (`199.7 ms` at `266`, `375.1 ms` at `336` on CPU).
   - **GPU ONNX Runtime / TensorRT Blocker (Stopped per Task 8 Constraints):** `onnxruntime 1.32.0` fails to initialize `CUDAExecutionProvider` (`cublasLt64_13.dll` missing; requires **CUDA 13.x + cuDNN 9.x**) and `TensorrtExecutionProvider` (`cublas64_13.dll` + **TensorRT** missing), silently falling back to `CPUExecutionProvider`. Because the system runs **CUDA 12.4** (`PyTorch 2.6.0+cu124`) and upgrading CUDA/TensorRT requires major environment installation, further TensorRT FP16/ORT-GPU work was halted without altering dependencies.

---

## 4. Path Forward for Phase 7

To deploy DAv2 without regressing real-time responsiveness: (1) use **pure `fp16_weights` (`model.half()`) + `torch.inference_mode()`**, (2) pair **`DAv2 336` or `DAv2 518` with selective keyframe / ByteTrack-triggered scheduling** so detector+tracker runs every frame (`15.3 ms` / `65.5 FPS`) while depth executes asynchronously or every $K$ frames on stable pothole tracks, or (3) optionally deploy CUDA-12-matched TensorRT FP16 in an isolated environment later.
