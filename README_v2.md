# RoadVision AI 2.0 — Phase 8A README

> [!NOTE]
> This is the Phase 8A prototype documentation. The V2 pipeline produces **uncalibrated relative inverse-depth residuals** only. No physical metric cm/mm measurements are computed or validated. The legacy V1 system (`roadvision_engine.py`, `server.py`) is completely unchanged.

---

## Overview

RoadVision AI 2.0 (`roadvision_v2/`) is a modular pothole detection prototype integrating:

- **YOLOv8n Phase 2 Controlled Detector** — `pothole_unified_yolov8n_100e_sgd`
- **ByteTrack Object Tracker** — `road_tracker.yaml`
- **Depth Anything V2 Small** (336px or 518px, FP16) or **MiDaS Small 256** (FP16 autocast)
- **Optimized Adaptive RANSAC** — stratified_quadrant, 512 samples, 16–128 iterations
- **Elliptical Core Cavity Extraction** — model-specific cavity selection
- **TrackDepthCache** — per-track measurement cache with time-aware EMA (τ=0.16s)
- **AdaptiveDepthScheduler** — 5-strategy scheduling with budget awareness

---

## Quick Start

### Prerequisites

Ensure all Phase 1–7B dependencies are installed (PyTorch 2.6+cu124, ultralytics, OpenCV, etc.).

```bash
cd D:\DeepLearning\Deep_Learning_project---POTHOLEdetection
```

---

## Run Commands

### LIVE Mode — Detection + ByteTrack (no depth)

```bash
python run_roadvision_v2.py \
  --source uploads/upload_1794847200_ruralRoad_potHoles.mp4 \
  --mode live \
  --save-video \
  --save-json \
  --output-dir runs/v2_outputs/
```

**Expected output:**
- `runs/v2_outputs/phase8a_live_upload_1794847200_ruralRoad_potHoles_annotated.mp4`
- `runs/v2_outputs/phase8a_live_upload_1794847200_ruralRoad_potHoles_frames.json`
- `runs/v2_outputs/phase8a_live_upload_1794847200_ruralRoad_potHoles_tracks.json`
- `runs/v2_outputs/phase8a_live_upload_1794847200_ruralRoad_potHoles_session_summary.json`

---

### BALANCED Mode — DAv2 336 + Budget Scheduler

```bash
python run_roadvision_v2.py \
  --source uploads/upload_1794847200_ruralRoad_potHoles.mp4 \
  --mode balanced \
  --enable-depth \
  --save-video \
  --save-json \
  --save-csv \
  --output-dir runs/v2_outputs/
```

**Expected output:**
- Annotated video with relative score overlays (not cm!)
- Per-frame JSON with `FRESH_VALID`, `CACHED_VALID`, and `FRESH_INVALID` measurement states
- Per-frame CSV with timing breakdown
- Session summary with depth invocation rate

---

### SURVEY Mode — DAv2 518 + Full Diagnostics

```bash
python run_roadvision_v2.py \
  --source roadvision-dashboard/public/videos/raw_survey.mp4 \
  --mode survey \
  --enable-depth \
  --diagnostics \
  --save-video \
  --save-json \
  --save-csv \
  --output-dir runs/v2_outputs/
```

**Expected output:**
- Highest measurement coverage (candidate-gated, no budget limit)
- Horizon proxy diagnostics enabled (research-only)
- Full rejection reason annotations

---

### Urban Video — BALANCED Mode

```bash
python run_roadvision_v2.py \
  --source "Watch Video_ Terrible potholes in Rangoon Road.mp4" \
  --mode balanced \
  --enable-depth \
  --save-video \
  --save-json \
  --max-frames 200 \
  --output-dir runs/v2_outputs/
```

---

## Integration Benchmark

```bash
python experiments/benchmark_phase8a_integration.py
```

Runs all 3 modes × 3 videos, outputs to `runs/benchmark/phase8a_integration_results.json`.

---

## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: **135+ tests** passing in ~15–20 seconds (CPU, no GPU for unit tests).

New V2 test files:
- `tests/test_roadvision_v2_config.py` — 20 tests for config loading and validation
- `tests/test_roadvision_v2_schemas.py` — 22 tests for data contracts
- `tests/test_roadvision_v2_pipeline.py` — 22 tests including legacy hash integrity

---

## Configuration

Edit `config/roadvision_v2.yaml` to adjust:

```yaml
modes:
  balanced:
    depth:
      enabled: true
      model_id: "depth_anything_v2_small"
      input_size: 336        # 336 or 518 (must be divisible by 14)
      precision: "fp16_weights"  # or "fp32"
    scheduler:
      strategy: "E_budget_aware"
      budget_target_ema_ms: 38.0   # Soft deadline (25fps = 40ms)
```

> [!WARNING]
> `midas_small` requires `input_size: 256` and does not support `fp16_weights`. Use `fp16_autocast` or `fp32` with MiDaS.

---

## Output Schema

Each hazard observation in the JSON output has:

```json
{
  "frame_context": { "frame_idx": 42, "timestamp_sec": 1.68, "mode": "balanced", ... },
  "detection": { "track_id": 7, "box_xyxy": [100, 150, 250, 300], "detector_confidence": 0.72, ... },
  "measurement": {
    "measurement_state": "FRESH_VALID",
    "raw_relative_cavity_score": 0.123,
    "normalized_relative_score": 1.45,
    "smoothed_relative_score": 0.138,
    "measurement_validity": true,
    "failure_reasons": [],
    "is_water_suspected": false,
    "ref_diagnostics": { "inlier_ratio": 0.78, "populated_quadrants": 4, ... }
  }
}
```

> [!CAUTION]
> `smoothed_relative_score` is an uncalibrated inverse-depth residual. It is **not** in centimetres and must not be labelled as physical depth. Higher values indicate greater relative inverse-depth depression relative to the fitted road plane reference — not a physical pothole depth measurement.

---

## Architecture

```
roadvision_v2/
├── config.py     → load_v2_config(), RoadVisionV2Config, 7 nested config dataclasses
├── schemas.py    → 8 typed dataclasses: FrameContext, DetectionObservation,
│                   RelativeDepthMeasurement, HazardObservation, PipelinePerformance, ...
└── pipeline.py   → RoadVisionV2Pipeline:
                      .process_frame(img_bgr, frame_idx, ts_sec) → (hazards, perf)
                      .process_video(source, output_dir, ...) → summary dict
                      .reset_session(reason)
```

---

## Reports

| Report | Location |
|--------|----------|
| Repository Architecture Audit | `reports/phase8a_repository_architecture_audit.md` |
| V2 Architecture | `reports/phase8a_v2_architecture.md` |
| Integration Benchmark | `reports/phase8a_integration_benchmark.md` |
| Legacy Compatibility | `reports/phase8a_legacy_compatibility.md` |

---

## Limitations

- Physical metric pothole depth, area, and volume are **not computed** (no calibration rig)
- DAv2 518 at ~137ms/keyframe; real-time use requires budget scheduling or hardware upgrade
- `torch.compile` is incompatible with the current Windows/CUDA environment
- ONNX GPU execution providers are incompatible with the current CUDA environment
- Horizon proxy diagnostics are research-only and do not affect any measurements

---

## Backend REST API (Phase 8B)

The modular V2 pipeline is exposed through an isolated Flask Blueprint under `/api/v2` (`roadvision_api_v2/`), backed by SQLite WAL persistence and a bounded single-GPU worker (`max_gpu_concurrency: 1`). Legacy V1 routes (`/api/health`, `/api/detect`, etc.) remain preserved and coordinate GPU access via `gpu_inference_lock()`.

### Feature Flag & Startup

By default, `config/roadvision_api_v2.yaml` commits `enabled: false`. Enable V2 locally either via environment variable (recommended for temporary testing without altering tracked config) or YAML:

```powershell
# PowerShell: enable V2 API on localhost only (debug forced off)
$env:ROADVISION_V2_ENABLED = "true"
python server.py
```

Default V2 listen address: `http://127.0.0.1:5000`.

### PowerShell Examples

Using existing repository road footage (`Watch Video_ Terrible potholes in Rangoon Road.mp4`):

```powershell
# 1. Health & capabilities
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/health"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/capabilities"

# 2. Submit a BALANCED inspection job (multipart upload)
$form = @{
    video        = Get-Item "Watch Video_ Terrible potholes in Rangoon Road.mp4"
    mode         = "balanced"
    enable_depth = "true"
    save_video   = "true"
    save_json    = "true"
    save_csv     = "true"
    max_frames   = "40"
}
$job = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/jobs" -Method Post -Form $form
$jobId = $job.job_id

# 3. Poll status
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/jobs/$jobId"

# 4. Cancel an active or queued job
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/jobs/$jobId/cancel" -Method Post

# 5. Fetch summary, paginated observations, tracks, performance, and artifacts
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/jobs/$jobId/summary"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/jobs/$jobId/observations?page=1&page_size=50"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/jobs/$jobId/tracks?page=1&page_size=50"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/jobs/$jobId/performance?page=1&page_size=50"
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v2/jobs/$jobId/artifacts"

# 6. Download annotated video
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/v2/jobs/$jobId/artifacts/annotated_video?download=1" -OutFile "annotated.mp4"
```

### curl Examples

```bash
curl -s http://127.0.0.1:5000/api/v2/health
curl -s -X POST http://127.0.0.1:5000/api/v2/jobs \
  -F "video=@Watch Video_ Terrible potholes in Rangoon Road.mp4" \
  -F "mode=balanced" -F "enable_depth=true" -F "max_frames=40"
curl -s http://127.0.0.1:5000/api/v2/jobs/<job_id>
curl -s -X POST http://127.0.0.1:5000/api/v2/jobs/<job_id>/cancel
curl -L http://127.0.0.1:5000/api/v2/jobs/<job_id>/artifacts/annotated_video -o annotated.mp4
```

### Python `requests` Example

```python
import time, requests

BASE = "http://127.0.0.1:5000/api/v2"
with open("Watch Video_ Terrible potholes in Rangoon Road.mp4", "rb") as fh:
    r = requests.post(
        f"{BASE}/jobs",
        files={"video": ("rangoon.mp4", fh, "video/mp4")},
        data={"mode": "balanced", "enable_depth": "true", "max_frames": "40"},
    )
job_id = r.json()["job_id"]
while True:
    st = requests.get(f"{BASE}/jobs/{job_id}").json()
    if st["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
        break
    time.sleep(0.5)
print(requests.get(f"{BASE}/jobs/{job_id}/summary").json())
```

### API Configuration Reference (`config/roadvision_api_v2.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Committed default; enable via `ROADVISION_V2_ENABLED=true` or YAML |
| `host` / `port` | `127.0.0.1` / `5000` | Loopback-only bind enforced for local research |
| `database_path` | `runs/v2_api/roadvision_v2_jobs.db` | SQLite WAL database path |
| `upload_directory` / `artifact_directory` | `uploads_v2` / `artifacts_v2` | Isolated per-job media & output roots |
| `maximum_upload_size` | `268435456` (256 MiB) | Streamed upload size cap |
| `allowed_video_formats` | `[.mp4, .avi, .mov, .mkv]` | Permitted extensions (verified via OpenCV decode) |
| `maximum_video_resolution` | `[1920, 1080]` | Maximum width × height |
| `maximum_video_duration` / `maximum_processed_frames` | `300.0`s / `9000` | Conservative caps for RTX 2050 |
| `max_queued_jobs` / `max_gpu_concurrency` | `5` / `1` | Bounded queue & single active GPU job |
| `progress_update_interval` / `worker_shutdown_timeout` | `0.5`s / `15.0`s | Throttled DB writes & cooperative stop timeout |
| `allowed_cors_origins` | Localhost Vite/React origins | Wildcard `*` rejected |

### Known API Limitations

- **No authentication or user authorization**: Anyone able to reach `127.0.0.1:5000` can list/poll jobs and download artifacts. UUID job IDs are identifiers, not authorization. Not safe for public deployment.
- **Single-process worker boundary**: Enforced via `SingleProcessWorkerGuard` file lock; multi-worker WSGI scaling is unsupported for local GPU execution.
- **OpenAPI specification**: Full machine-readable specification is in [roadvision_api_v2/openapi.yaml](roadvision_api_v2/openapi.yaml).

---

*Phase 8A + Phase 8B — RoadVision AI 2.0 Backend Complete*
