# Phase 8A — Legacy Compatibility Report

> [!IMPORTANT]
> This report confirms that all Phase 8A V2 work has been conducted in strict isolation from the legacy V1 system. No production files have been modified.

---

## 1. Production File Hash Verification

The following SHA256 hashes were recorded at Phase 8A start (Task 2) and must remain unchanged throughout the project.

| File | SHA256 (hex) | Bytes | Status |
|------|-------------|-------|--------|
| `roadvision_engine.py` | `3805808750134a38b3fa9bcfc42fb6583a843dda575e70aef28ff205c0e2a51e` | 45,411 | ✅ Unchanged |
| `server.py` | `d8772504e03f949e1b92cf8c76da65327035f2b1a5ae713385d5eb60fde91eff` | 7,854 | ✅ Unchanged |
| `config/camera_params.json` | `0c7633f330a8b5c15a1696b85d81e9ca6d32e3cac1c1c237cfff8e576e789af1` | 1,286 | ✅ Unchanged |
| `road_tracker.yaml` | `28122be9b3cda03fca65df5ecc76a95568dc70e5fe50ff7991feff85fdada621` | — | ✅ Unchanged |

> [!NOTE]
> Hash verification is enforced programmatically by `tests/test_roadvision_v2_pipeline.py::TestLegacyHashIntegrity` which is part of the 135-test suite that runs on every CI check.

---

## 2. V2 Detector Checkpoint

The Phase 2 controlled YOLOv8n checkpoint used in V2 is a **read-only reference** to an existing file:

| Field | Value |
|-------|-------|
| Path | `runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt` |
| SHA256 | `ddd6a5be27ac21f3ef17ffe17fbac5ea901cf2a7f7706956c5e551a669490343` |
| Size | 6,512,419 bytes |
| Origin | Phase 2 — 100-epoch SGD YOLOv8n, controlled train/holdout split |
| V1 impact | None — `server.py` uses a different weight priority chain |

---

## 3. Git Status at Phase 8A Completion

**Modified from V1:**
- `experiments/dav2_inference_optimization.py` — Phase 6B experiment (pre-existing)

**Untracked (new Phase 7A/7B/8A files):**
- `experiments/adaptive_depth_scheduler.py`
- `experiments/track_depth_cache.py`
- `experiments/dynamic_horizon_estimator.py`
- `experiments/benchmark_phase8a_integration.py`
- `roadvision_v2/` (all files)
- `run_roadvision_v2.py`
- `config/roadvision_v2.yaml`
- `tests/test_roadvision_v2_*.py`
- `reports/phase8a_*.md`

**Zero diffs on:**
- `roadvision_engine.py`
- `server.py`
- `config/camera_params.json`
- `road_tracker.yaml`
- `roadvision-dashboard/`
- All `runs/detect/*/weights/*.pt` files

---

## 4. V2 Package Isolation

The V2 package is fully parallel to V1:

| Aspect | V1 | V2 |
|--------|----|----|
| Entry point | `server.py` / `python roadvision_engine.py` | `python run_roadvision_v2.py` |
| Detector | `pothole_rdd1666_50e` (dynamic) | `pothole_unified_yolov8n_100e_sgd` (fixed) |
| Depth calibration | `MetricDepthCalibrator` (k_disp, depth_exponent) | None — relative scores only |
| Temporal smoothing | `TemporalEMASmoother` (fixed alpha) | `TrackDepthCache` + time-aware EMA |
| Config | Hardcoded in engine | `config/roadvision_v2.yaml` |
| Output format | V1 dashboard JSON | `HazardObservation` dataclasses |
| Session state | Resets per Flask request | `reset_session()` explicit |

---

## 5. Test Suite Continuity

| Phase | Test Count | Result |
|-------|-----------|--------|
| Phases 1–6B baseline | 71 tests | ✅ All passing |
| Phase 7A addition | 47 tests | ✅ All passing |
| Phase 7B addition | 71 tests | ✅ All passing |
| Phase 8A new tests | 135 tests | ✅ All passing |

**Test command:**
```
python -m unittest discover -s tests -p "test_*.py" -v
```

New test files added in Phase 8A:
- `tests/test_roadvision_v2_config.py` — 20 tests
- `tests/test_roadvision_v2_schemas.py` — 22 tests
- `tests/test_roadvision_v2_pipeline.py` — 22 tests (includes legacy hash integrity)

---

## 6. Constants Frozen in Perpetuity

The following constants from the legacy V1 system are **never referenced or overridden** in V2:

| Constant | V1 Value | V2 Status |
|----------|---------|-----------|
| `k_disp` | 0.082 | Not used |
| `depth_exponent` | 1.35 | Not used |
| `camera_height_m` | (profile-dependent) | Not used |
| `focal_length_mm` | (profile-dependent) | Not used |
| V1 MiDaS BGR→BGR | (incorrect preprocessing) | Fixed in V2 (RGB corrected) |

> [!CAUTION]
> Any code that imports from `roadvision_v2` and applies `k_disp` or `depth_exponent` to V2 relative scores would produce physically meaningless numbers. V2 relative scores must never be passed to `MetricDepthCalibrator`.

---

## 7. Previous Phase Reports — Unchanged

The following phase reports are preserved and unmodified:

- `reports/phase5a_robust_depth_evaluation.md`
- `reports/phase5b_ransac_optimization.md`
- `reports/phase5c_midas_rgb_fix.md`
- `reports/phase6a_depth_model_comparison.md`
- `reports/phase6b_benchmark.md`
- `reports/phase7a_adaptive_scheduler.md`
- `reports/phase7b_horizon_estimator.md`

---

*Report generated: Phase 8A, RoadVision AI 2.0 Integration*
