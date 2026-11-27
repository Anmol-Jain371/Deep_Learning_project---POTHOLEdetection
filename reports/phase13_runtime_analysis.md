# Phase 13 — Runtime & Memory Analysis (RTX 2050 4 GB VRAM)

**Run ID:** `phase13_full_20261127`  
**Environment:** Windows 11, Python 3.12, PyTorch `2.6.0+cu124`, NVIDIA GeForce RTX 2050 (`4,096 MiB` VRAM)  
**Measurement Protocol:** Cold-start `init_warmup_ms` isolated from `torch.cuda.synchronize()`-timed forward passes; `torch.cuda.reset_peak_memory_stats()` per arm.

---

## 1. Component-Level GPU Cost Summary

| Component | Mean Latency (ms) | Median (ms) | P95 (ms) | Throughput (FPS) | Peak Alloc VRAM (MB) | Peak Reserved VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n (`640px`, `N=60`) | 12.383 | 11.021 | 18.955 | 80.75 | 68.49 | 646.0 |
| YOLOv8s (`640px`, `N=60`) | 25.710 | 24.629 | 33.058 | 38.90 | 200.77 | 1,074.0 |
| MiDaS Small (`256`, `fp32`) | 16.451 | 16.120 | 20.903 | — | 127.43 | 152.0 |
| MiDaS Small (`256`, `fp16_autocast`) | 21.410 | 21.019 | 27.770 | — | 163.49 | 184.0 |
| DAv2 Small (`266`, `fp16_autocast`) | 25.801 | 24.551 | 30.503 | — | 243.42 | 280.0 |
| DAv2 Small (`336`, `fp16_autocast`) | 48.175 | 48.051 | 48.681 | — | 274.69 | 312.0 |
| DAv2 Small (`336`, `fp32`) | 60.256 | 60.270 | 60.438 | — | 218.41 | 254.0 |
| DAv2 Small (`392`, `fp16_autocast`) | 68.817 | 68.792 | 69.134 | — | 313.59 | 354.0 |
| DAv2 Small (`518`, `fp16_autocast`) | 147.008 | 146.935 | 147.389 | — | 409.69 | 526.0 |

## 2. Practical Deployment Trade-offs on 4 GB VRAM
- **Real-time `LIVE` profile:** `YOLOv8n` leaves ~`3.4 GB` VRAM free when depth is off (`80.75 FPS` detector-only isolated pass) and ~`3.2 GB` when paired with `MiDaS Small 256` (`16.5 ms` `fp32`).
- **`BALANCED` profile (`YOLOv8n` + `DAv2 336` + `E_budget_aware`):** Cold init is ~`855 ms` for DAv2 336. In the frozen `150`-frame exploratory trace (`frozen_det_mean_ms = 15.004 ms`, `frozen_depth_mean_ms = 65.592 ms` incl. resize/host transfer), `E_budget_aware` cuts depth calls by `46.67%` (`80/150` frames), lifting simulated sustained rate from `11.50 FPS` (`A_always_on`) to `18.63 FPS` (`53.67 ms` mean) within `287.46 MB` peak allocated GPU memory.
- **`SURVEY` offline profile (`DAv2 518`):** Requires `147.0 ms` inference (`184.6 ms` total incl. resize/pre) and `526.0 MB` reserved VRAM, justifying offline-only classification.
