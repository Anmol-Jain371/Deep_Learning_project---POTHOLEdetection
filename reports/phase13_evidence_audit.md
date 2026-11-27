# Phase 13 — Scientific Evidence Audit

**Audit Date:** 2026-11-27  
**Audited Repository Commit / State:** Pre-Phase 13 frozen application (`feature_freeze_inventory.json`)  
**Policy:** Existing narrative reports are NEVER treated as sufficient scientific proof without verifiable code, checkpoint identity, raw logs, and traceable metric definitions.

---

## Evidence Classification Registry

### Category A — Directly Reproducible Experimental Results (Deterministic / Code + Seed Traceable)
1. **Optimized Adaptive RANSAC Reference-Plane Ablation (`experiments/optimized_relative_depth.py`, `experiments/robust_relative_depth.py`):**
   - Synthetic suite with known planar slope, Gaussian/structured noise, injected outliers, and boundary truncation (`seed=42`). Fully reproducible in CPU tests (`runs/benchmark/optimized_relative_depth_results.json`, `synthetic_relative_depth_validation.json`).
2. **Core-Aware Cavity Extraction (`experiments/core_aware_cavity_extraction.py`):**
   - Deterministic core masks (`full_box`, `rect_trim_*`, `elliptical_core`, `multi_scale_agreement`) runnable over identical saved depth maps (`runs/benchmark/phase6a_depth_comparison.json`).
3. **Validity-Aware Temporal Filtering (`experiments/validity_aware_temporal_filter.py`):**
   - Synthetic invalid-gap & stale-reset sequences (`tests/test_phase9_quality_assessment.py`, `runs/benchmark/optimized_relative_depth_results.json`).
4. **Phase 9 Quality Estimator & Phase 10 Geometry Snapshot Resampling (`roadvision_v2/quality/assessment.py`, `roadvision_v2/geometry/surface.py`):**
   - Verified via unit tests (`tests/test_phase9_quality_assessment.py`, `tests/test_phase10_geometry_visualization.py`). *Exception:* the **42 historical snapshots** written under `artifacts_v2/` prior to the `valid_ring_mask` lookup fix contain empty `reference_mask_grid` (`0` cells) and cannot be retroactively treated as corrected snapshots.
5. **Phase 11 Geospatial Synchronization & Phase 12 Maintenance Determinism (`roadvision_v2/geospatial/`, `roadvision_v2/maintenance/`):**
   - Deterministic under controlled synthetic GPS fixtures and stored SQLite/artifact records (`tests/test_phase11_geospatial_intelligence.py`, `tests/test_phase12_maintenance_intelligence.py`).

### Category B — Historical Results Supported by Saved Artifacts (Non-Empty Checkpoints & Metrics Logs)
1. **YOLOv8n vs YOLOv8s Controlled Training (`runs/detect/pothole_unified_yolov8n_100e_sgd/`, `runs/detect/pothole_unified_yolov8s_100e/`, `runs/benchmark/phase2_yolov8n_vs_yolov8s_controlled.json`):**
   - Checkpoints, `args.yaml`, `results.csv`, and PR/confusion outputs exist on disk; reproducible inference metrics on `dataset_unified` splits (subject to Category C/D independence caveats).
2. **Pothole440 Regrouped Segmentation Split & Training (`dataset_pothole440_seg_grouped/`, `runs/segment/pothole440_yolov8n_seg_v1/`, `runs/benchmark/pothole440_segmentation_test.json`):**
   - Base-ID grouped split (`68` train / `15` val / `15` test unaugmented originals) fixed raw cross-split leakage, with full SHA-256 manifests.
3. **Phase 6B Precision Regression (`runs/benchmark/phase7a_fp16_regression.json`) & Phase 8A/8B Pipeline Benchmarks (`runs/benchmark/phase8a_integration_results.json`, `phase8b_api_results.json`):**
   - Supported by saved JSON/CSV logs and RTX 2050 timings.

### Category C — Exploratory Results from Previously Used Data (Not Independent External Proof)
1. **Road Video Evaluations (`ruralRoad_202f`, `rawSurvey_160f`, `Watch Video_ Terrible potholes in Rangoon Road.mp4`):**
   - Repeatedly inspected during Phases 2–10 tuning. Useful only for exploratory systems, latency, scheduling, and failure-case profiling.
2. **Detector Evaluation on `dataset_unified/val`, `dataset_unified/test`, and `dataset_unified/external_rangoon`:**
   - Inspected in Phase 2 (`dHash/pHash <= 6` cross-split pairs reported in `reports/depth_ground_truth_audit.md`; incomplete annotations on 35 continuous `external_rangoon` frames). Metrics measure exploratory fit, NOT untouched external generalization.
3. **Cross-Resolution Relative Depth Comparison (`MiDaS 256` vs `DAv2 266/336/392/518`):**
   - Higher-resolution DAv2 (`518`) serves strictly as an internal relative-consistency reference, NOT physical depth ground truth.

### Category D — Unsupported, Provenance-Incomplete, or Methodologically Compromised Historical Claims
1. **Physical Depth Validation (`depth_ground_truth_manifest.csv` `P01–P15`, `runs/depth_validation_report.json`):**
   - Audited in `reports/depth_ground_truth_audit.md`: failed live-pipeline reproduction (`0` detections at `P13`, `P07` mismatch), `P01/P02` matched pre-existing synthetic test triplets (`T01`/`T02`), lacked straightedge measurement logs/calibration provenance, and `suggested_k_disp = 0.07664` was fitted in-sample. **Classified as uncertified; physical cm/volume claims remain rejected in V2.**
2. **Legacy Zero-Clamped Production Depth & Calibration Constants (`roadvision_engine.py`, `k_disp=0.082`, `depth_exponent=1.35`):**
   - Hardcoded heuristic constants not calibrated against verified ground truth.
3. **Raw Unregrouped Pothole440 `test` Split Metrics:**
   - Contaminated (`34/40` test images shared base pothole IDs with `train`). Superseded by grouped split (`dataset_pothole440_seg_grouped`).
4. **Heuristic `HIGH_QUALITY` Edge Cases (Phase 9):**
   - Confirmed via snapshot audit (`0f286074`): renormalizing over `available_weight` when `temporal_consistency` is `None` (`0.85 >= 0.50`) plus low weight on `detection_quality` (`0.15`) and `environmental_risk` (`0.10`) allows `HIGH_QUALITY` despite `detection_quality < 0.50`, boundary truncation, or absent temporal history.

### Category E — Blocked Evaluations Requiring Unavailable Ground Truth
1. **Independent Annotated Detection Generalization:** Blocked (`evaluation/ground_truth_labels/*.txt` has `40` empty `0`-byte files; `dataset_unified` lacks untouched held-out test data).
2. **Physical Depth / Volume Accuracy Validation:** Blocked (no straightedge/LiDAR/stereo ruler-verified road dataset paired with dashcam telemetry).
3. **Real-World Cross-Trip GPS Matching Validation:** Blocked (no genuine multi-pass synchronized dashcam GPS logs with verified physical candidate pair labels).
4. **Human Civil-Engineering Maintenance Prioritization Accuracy:** Blocked (no expert civil-engineering severity/priority labels).
