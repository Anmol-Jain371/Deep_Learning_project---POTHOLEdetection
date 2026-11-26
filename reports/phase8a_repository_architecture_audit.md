# Phase 8A — Repository Architecture Audit

> [!NOTE]
> This report covers RoadVision AI Phase 8A. All evaluations are **exploratory**. Relative cavity scores are uncalibrated inverse-depth residuals. No physical metric cm values are computed or validated.

---

## 1. Production Entry Points (V1 — Unchanged)

| File | Role | Lines |
|------|------|-------|
| `roadvision_engine.py` | Legacy CLI pipeline + Flask backend engine | 976 |
| `server.py` | Flask HTTP API (4 routes) | 193 |

### 1.1 `server.py` Routes

| Route | Method | Function |
|-------|--------|----------|
| `/api/health` | GET | Health check |
| `/api/benchmark` | GET | Benchmark info |
| `/api/camera-config` | GET | Return camera profiles |
| `/api/detect` | POST | Frame-level pothole detection |

**Default detector weight priority:** `pothole_rdd1666_50e` → `combined_v2_50e` → `combined_v2` → `pothole_best.pt`  
**Per-request state:** `pipeline.smoother.records` and `pipeline.display_id_map` are reset.

### 1.2 `roadvision_engine.py` Core Classes

| Class | Description |
|-------|-------------|
| `RoadVisionPipeline` | Main V1 processing pipeline |
| `MetricDepthCalibrator` | Fixed k_disp=0.082, depth_exponent=1.35 |
| `AnnularDonutDepthExtractor` | Legacy annular ring extractor |
| `TemporalEMASmoother` | Fixed-alpha EMA display smoother |
| `DashcamKinematicGPS` | GPS track interpolator |
| `SpecularAquaplaneDetector` | Water/reflection detection |

> [!IMPORTANT]
> `SpecularAquaplaneDetector` is imported by both `experiments/optimized_relative_depth.py` and `experiments/core_aware_cavity_extraction.py`. The V2 pipeline inherits this dependency implicitly through the extraction layer.

---

## 2. Detector Checkpoints

| Checkpoint | File | Purpose |
|-----------|------|---------|
| V1 production | `pothole_rdd1666_50e` (dynamic path) | Legacy server.py pipeline |
| V2 provisional | `runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt` | Phase 2 controlled YOLOv8n |

**V2 detector SHA256:** `ddd6a5be27ac21f3ef17ffe17fbac5ea901cf2a7f7706956c5e551a669490343` (6,512,419 bytes)

---

## 3. Tracking Configuration

**ByteTrack config:** `road_tracker.yaml`
- `track_high_thresh: 0.14`
- `fuse_score: false`
- `track_buffer: 75`

**SHA256:** `28122be9b3cda03fca65df5ecc76a95568dc70e5fe50ff7991feff85fdada621`

---

## 4. Depth Models

| Model | Source | Location |
|-------|--------|----------|
| MiDaS Small | TorchHub `intel-isl/MiDaS` | Torch Hub cache |
| Depth Anything V2 Small | HuggingFace `depth-anything/Depth-Anything-V2-Small` | `~/.cache/huggingface/hub/` |

---

## 5. Experimental Modules Inventory

| Module | Purpose | V2 Reuse |
|--------|---------|----------|
| `experiments/depth_model_comparison.py` | MiDaS + DAv2 adapters | ✅ Via pipeline |
| `experiments/optimized_relative_depth.py` | RANSAC reference extraction | ✅ Direct call |
| `experiments/core_aware_cavity_extraction.py` | Cavity strategy extraction | ✅ Direct call |
| `experiments/track_depth_cache.py` | Per-track measurement cache + EMA | ✅ Direct use |
| `experiments/adaptive_depth_scheduler.py` | 5-strategy depth scheduler | ✅ Direct use |
| `experiments/validity_aware_temporal_filter.py` | Fixed-step EMA + rolling median | ✅ Via cache |
| `experiments/dynamic_horizon_estimator.py` | Vanishing-point proxy (optional) | ✅ Optional diag |
| `experiments/roi_guided_segmentation.py` | ROI-guided MobileSAM | ❌ Offline only |
| `experiments/shadow_depth_pipeline.py` | Phase 7A integration pilot | ❌ Reference only |
| `experiments/dav2_inference_optimization.py` | FP16/ONNX benchmarks | ❌ Completed |

---

## 6. Dashboard Files

Located in `roadvision-dashboard/` (React build). **No V2 modifications planned in Phase 8A.**

| File | SHA256 (prefix) |
|------|-----------------|
| `roadvision-dashboard/public/index.html` | (recorded in Phase 8A Task 2) |
| `roadvision-dashboard/src/App.js` | (recorded in Phase 8A Task 2) |

---

## 7. Duplicate Functionality Identified

| Functionality | V1 Location | Experimental Location | V2 Resolution |
|--------------|-------------|----------------------|---------------|
| Annular ring sampling | `roadvision_engine.py:AnnularDonutDepthExtractor` | `optimized_relative_depth.py` | Use experimental (stratified_quadrant, 512 samples) |
| Temporal smoothing | `roadvision_engine.py:TemporalEMASmoother` | `validity_aware_temporal_filter.py` + `track_depth_cache.py` | Use `TrackDepthCache` (time-aware EMA + cache separation) |
| Depth loading | `roadvision_engine.py` (MiDaS BGR→BGR) | `depth_model_comparison.py` (RGB corrected) | Use corrected adapter |
| Camera distance formula | `roadvision_engine.py:MetricDepthCalibrator` | None | **NOT used in V2** |

---

## 8. Compatibility Risks

| Risk | Mitigation |
|------|-----------|
| `SpecularAquaplaneDetector` import side-effect | V2 lazily imports experimental modules; engine import is isolated |
| ByteTrack `persist=True` state across sessions | `reset_session()` resets `TrackDepthCache` and `AdaptiveDepthScheduler` |
| torch.hub network calls on first MiDaS load | Warm-up call or first-frame latency spike; documented |
| HuggingFace download on first DAv2 load | Cached after first run; air-gap environments need pre-download |
| OpenCV 5.0 LSD array shape `(N,4)` vs `(N,1,4)` | Fixed in Phase 7B with `np.asarray().reshape(-1,4)` |

---

## 9. Migration Plan

```
Phase 8A (Current):
  roadvision_v2/          ← New standalone package
  run_roadvision_v2.py    ← New CLI entry point
  config/roadvision_v2.yaml

  Legacy unchanged:
  roadvision_engine.py    ← V1 unchanged
  server.py               ← V1 unchanged

Phase 8B (Future):
  server_v2.py            ← Flask API wrapping RoadVisionV2Pipeline
  Dashboard update        ← New V2 endpoint integration

Phase 9+ (Future):
  Metric depth calibration (requires calibration rig)
  Physical area/volume (requires stereo or LiDAR ground truth)
```

> [!WARNING]
> Do NOT replace `server.py` or `roadvision_engine.py` until Flask API migration is separately planned and tested (Phase 8B+).

---

## 10. Git Status at Phase 8A Start

- Modified: `experiments/dav2_inference_optimization.py`
- Untracked: All Phase 7A/7B experiment files and Phase 8A new files
- All 71 existing tests passing at Phase 8A start

---

*Report generated: Phase 8A, RoadVision AI 2.0 Integration*
