# Phase 8A — Integration Benchmark Report

> [!IMPORTANT]
> All results in this report are **exploratory**. Videos are from real-world road footage with unknown ground-truth. Relative cavity scores are uncalibrated inverse-depth residuals. No physical metric cm values are reported. Results reflect the specific hardware configuration used (NVIDIA RTX 2050, 4 GB VRAM, CUDA 12.4, PyTorch 2.6.0+cu124, Python 3.12.10, OpenCV 5.0.0).

---

## 1. Hardware Configuration

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 2050 (4 GB VRAM) |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.12.10 |
| OpenCV | 5.0.0 |
| OS | Windows 11 |

---

## 2. Test Videos

| ID | File | Frames | FPS | Resolution | Scene |
|----|------|--------|-----|-----------|-------|
| `ruralRoad_202f` | `upload_1794847200_ruralRoad_potHoles.mp4` | 202 | 25 | 640×360 | Rural road, multiple potholes |
| `urban_rangoon_200f` | `Watch Video_ Terrible potholes in Rangoon Road.mp4` | 200 | 25 | 640×360 | Urban potholes |
| `rawSurvey_160f` | `roadvision-dashboard/public/videos/raw_survey.mp4` | 160 | variable | 1280×720 | Survey footage |

---

## 3. Component Benchmarks (Phase 7A Reference)

Carried over from Phase 7A adaptive scheduler benchmarks:

| Component | Latency |
|-----------|---------|
| YOLOv8n detect+track (640px) | ~15–22 ms/frame |
| DAv2 336 FP16 depth inference | 53.57 ms (median) |
| DAv2 518 FP16 depth inference | 137.53 ms (median) |
| RANSAC 512 samples, stratified_quadrant | ~2–5 ms/track |
| AdaptiveDepthScheduler decide_frame() | <1 ms |
| TrackDepthCache read/update | <0.5 ms |

---

## 4. Live Mode Benchmark (Detection Only)

**Mode: `live` | Scheduler: `B_candidate_gated` | Depth: disabled**

| Video | FPS | Mean frame ms | P95 frame ms | Deadline misses |
|-------|-----|--------------|-------------|-----------------|
| `ruralRoad_202f` | *see results JSON* | *see results JSON* | *see results JSON* | *see results JSON* |
| `urban_rangoon_200f` | *see results JSON* | — | — | — |
| `rawSurvey_160f` | *see results JSON* | — | — | — |

> [!NOTE]
> Live mode does not invoke depth inference. End-to-end FPS is bounded by detection + tracking overhead only. Expected: 25–50+ FPS on RTX 2050.

---

## 5. Balanced Mode Benchmark (DAv2 336 + Budget Scheduler)

**Mode: `balanced` | DAv2 Small 336 FP16 | Scheduler: `E_budget_aware` | budget_target_ema_ms=38ms**

Expected behaviour from Phase 7A:
- Depth invocation rate: ~25–40% of frames (budget scheduler active)
- Budget skips activate when sustained depth latency > 38ms EMA
- End-to-end FPS: ~18–22 fps (depth frames dominate)
- Fresh valid coverage: depends on pothole presence per frame
- Starvation protection: no track goes >0.40s without a measurement attempt

---

## 6. Survey Mode Benchmark (DAv2 518 + Candidate Gated)

**Mode: `survey` | DAv2 Small 518 FP16 | Scheduler: `B_candidate_gated` | budget_target_ema_ms=1000ms (unlimited)**

Expected behaviour from Phase 7A:
- Depth invocation rate: ~100% of frames with any detection
- DAv2 518 latency: ~137ms per depth frame
- End-to-end FPS: ~5–8 fps on RTX 2050
- Maximum measurement coverage and highest holdout validity rate

---

## 7. Measurement State Distribution (Exploratory)

The following state categories are tracked per hazard observation across all modes:

| State | Meaning |
|-------|---------|
| `FRESH_VALID` | Depth ran this frame, extraction succeeded, scores are non-null |
| `FRESH_INVALID` | Depth ran this frame, extraction failed (quality gates not met) |
| `CACHED_VALID` | Depth was skipped, using cached score from recent keyframe |
| `EXPIRED` | Cache age exceeded `max_cache_age_sec=0.45s` |
| `NOT_EVALUATED` | Depth not enabled or no prior measurement |

> [!NOTE]
> `FRESH_INVALID` does not indicate a hardware failure — it reflects RANSAC quality gates correctly rejecting insufficient pavement reference support, water patches, low inlier ratios, or poor spatial coverage. These are intentional quality controls, not bugs.

---

## 8. Phase 7A vs Phase 8A Comparison

| Metric | Phase 7A (component) | Phase 8A (integrated) |
|--------|---------------------|----------------------|
| Depth reduction (E_budget_aware) | Up to 76% | *see results JSON* |
| Live sustained FPS | 20.65 FPS | *see results JSON* |
| DAv2 336 keyframe latency | ~97–111 ms | ~53–70 ms (FP16 weights) |
| Test count | 47 | 135 |

---

## 9. Full Benchmark Results

See: [`runs/benchmark/phase8a_integration_results.json`](../runs/benchmark/phase8a_integration_results.json)

The JSON file contains per-(mode, video) result dicts with:
- `frame_ms`: mean / median / P95 / max frame latency
- `depth_inf_ms`: depth-only timing for keyframes
- `detect_track_ms`: detection + tracking timing
- `deadline_misses`: count of frames exceeding `1000/fps_src` ms
- `measurement_state_histogram`: per-state counts
- `depth_invocation_rate`: fraction of frames where depth ran
- `peak_gpu_mb`: GPU peak allocation (if CUDA available)

---

## 10. Known Limitations

| Limitation | Detail |
|-----------|--------|
| Physical depth unverified | Absolute metric pothole depth requires calibrated camera + ground truth measurements |
| FP16 numerical coverage | Expanded FP16 regression (Phase 7A Task 2) covered 8+ frame types; local cavity stability not fully characterized for all scene types |
| torch.compile incompatible | Windows environment incompatibility persists |
| ONNX GPU providers incompatible | Static ONNX export succeeded but GPU execution providers incompatible with CUDA environment |
| Single-camera only | V2 assumes single forward-facing dashcam; stereo/multi-camera not supported |

---

*Report generated: Phase 8A, RoadVision AI 2.0 Integration*
