# Phase 13 — Experimental Evaluation Protocol & Statistical Methodology

**Harness Location:** `experiments/phase13/` (`run_evaluation.py`, `evaluation_config.yaml`)  
**Frozen Seed:** `42`  
**Hardware Platform:** Windows 11, NVIDIA GeForce RTX 2050 (4 GB VRAM), PyTorch `2.6.0+cu124`

---

## 1. Benchmark Discipline & Controls
- **Application Freeze:** All Phase 1–12 source code, checkpoints (`feature_freeze_inventory.json`), `roadvision_engine.py`, `config/camera_params.json`, and deployed quality thresholds were frozen prior to running Phase 13 evaluations.
- **Sequential GPU Execution:** Depth models (`MiDaS Small 256`, `DAv2 Small 266/336/392/518`) execute strictly one model at a time with `torch.cuda.empty_cache()` and `gc.collect()` between arms to prevent 4 GB VRAM eviction interference.
- **Synchronized Timing & Cold-Start Separation:** Every timed GPU loop runs warm-up frames first, calls `torch.cuda.synchronize()` around timed forward passes, resets peak CUDA stats (`torch.cuda.reset_peak_memory_stats()`), and separates initialization/warmup latency from per-frame inference latency.
- **Frozen Common Detection Streams:** All five scheduling policies (`A`–`E`) execute over the identical frozen YOLO+ByteTrack + depth sequence so differences reflect pure scheduler behavior rather than tracker stochasticity.

## 2. Correlation-Aware Statistical Policy
- Consecutive video frames and repeated track observations within a road clip are autocorrelated (`N_frames >> N_sessions`).
- `experiments/phase13/statistics.py` reports sample counts, mean, median, std, IQR, and P95 latency, plus paired differences over matched inputs, and explicitly suppresses bootstrap confidence intervals (`ci95_low/high = None`) whenever independent capture-group count is `< 5` to avoid pseudoreplication.
- Missing ground-truth benchmarks are marked **`BLOCKED`** cleanly rather than backfilled from unlabeled clips.
