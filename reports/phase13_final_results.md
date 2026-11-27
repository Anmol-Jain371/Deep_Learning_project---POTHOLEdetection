# Phase 13 — Final Results & Research Contribution Assessment

**Run ID:** `phase13_full_20261127`  
**Feature Freeze Inventory:** `runs/evaluation/phase13/feature_freeze_inventory.json` (`git HEAD: 50d37f37f24f28976ca8124ba2027d020f3b0ef3`)  
**Manifest Status:** `dataset_audit`, `detection`, `geometry`, `depth_cavity`, `scheduler_temporal`, `quality_geo_maint` → all `COMPLETED` (`independent_detection_generalization_status = BLOCKED`; `physical_depth_accuracy = BLOCKED`; `field_geospatial_validation = BLOCKED`; `expert_severity_labels = BLOCKED`).

---

## 1. Consolidated Evidence Summary by Evaluation Type

- **Implemented & Regression-Tested (Software Systems):** V2 modular pipeline, Phase 9 quality diagnostics, Phase 10 geometry serialization (`valid_ring_mask` fixed), Phase 11 GPS ingestion/matching, Phase 12 SQLite migrations + persistent human reviews + maintenance API + multi-format reports (`HTML`, `JSON`, `CSV`, `GeoJSON`).
- **Synthetic Validation (`COMPLETED`, controlled fixed seeds):**
  - Optimized MSAC/RANSAC reduces outlier reference-plane RMSE from `0.001065` (OLS) to `0.000131` and cuts latency `2.0x` vs baseline RANSAC (`4.514 ms` vs `9.014 ms`), with `100%` synthetic invalid rejection vs `0%` for legacy annular extraction.
  - Phase 10 grid downsampling preserves signed residual mean within `0.000396` (`raw=0.010664` vs `ds=0.011059`), preserves `9` missing cells & cross-sectional gaps, populates `1,141` reference-ring cells, and incurs `0` extra depth calls.
  - Validity-aware temporal EMA reduces step-to-step jitter `4.1x` (`0.00862` vs `0.03500` raw) without hiding expired frames (`legacy_ema_zero_clamp` hides `100%` of expired gaps as numeric values).
- **Exploratory Experimental Evidence (`COMPLETED`, previously used data — NOT independent test data):**
  - `YOLOv8n` (`80.75 FPS`, `68.49 MB` VRAM, `0.3643 mAP50` on `dataset_unified/test`) vs `YOLOv8s` (`38.90 FPS`, `200.77 MB` VRAM, `0.3581 mAP50`).
  - `DAv2 Small 336 fp16` (`48.18 ms`, `59.52%` fresh valid coverage, `0.9992` rank corr. vs `DAv2 518`) vs `MiDaS 256` (`28.57%` coverage) vs `DAv2 518` (`147.01 ms`).
  - `E_budget_aware` scheduler (`46.67%` depth-call reduction, `71.34%` effective availability, `0` expired observations vs `39` expired under `C_fixed_interval`).
- **Blocked Evaluations (No Fabricated Claims):**
  - Independent annotated detection generalization (`0/40` annotated labels in `evaluation/`; `328` `dHash<=6` cross-split pairs in `dataset_unified`).
  - Physical metric depth (`cm/mm`) & volume accuracy.
  - Real-world cross-trip GPS matching field accuracy & civil-engineering maintenance priority ground truth.

---

## 2. Research Contribution Evidence Matrix

| Component | Implemented Method | Comparison Baseline | Actual Experimental Evidence (`phase13_full_20261127`) | Observed Benefit / Trade-off | Known Limitations | Remaining Validation Required |
|---|---|---|---|---|---|---|
| **Optimized Reference Plane** | Stratified-quadrant MSAC/RANSAC + validity gates | Legacy Annular / OLS / Baseline RANSAC | `result_geometry.json` (`45` synthetic cases) | `100%` synthetic invalid rejection (vs `0%` legacy); `8.1x` lower outlier RMSE vs OLS; `2.0x` faster than baseline RANSAC | Synthetic plane RMSE ≠ physical pothole depth | Straightedge/LiDAR field comparison |
| **Core-Aware Cavity Extraction** | Elliptical core (`0.90`) inside fitted plane + `p75` | `full_box`, `rect_trim_20`, P440 `yolov8n-seg` | `result_depth_cavity.json` (`42` boxes; `15` P440 test imgs) | Similar validity (`59.5%` vs `61.9%`), avoids corner debris; P440 seg reaches `0.5977` mIoU | Close-up P440 domain shift from dashcams; `N=15` test imgs | Dashcam segmentation annotations |
| **Budget-Aware Depth Scheduling** | `E_budget_aware` causal track-aware keyframe gating | `A_always_on`, `B_cand_gated`, `C_fixed_int`, `D_track_aware` | `result_scheduler_temporal.json` (`150f`, `464` obs) | Cuts depth calls `46.67%` (`18.63` vs `11.50` sim FPS) with `0` expired frames | Evaluated on frozen `150f` exploratory stream | Independent long multi-weather video |
| **Validity-Aware Temporal Cache** | Explicit 5-state lifecycle + time-aware EMA expiry | `raw`, `legacy_ema_zero_clamp`, `rolling_median` | `result_scheduler_temporal.json` (`25`-step synthetic) | Preserves null gaps on `2/2` expired frames (`0/2` legacy); cuts jitter `4.1x` | Lag increases slightly across abrupt cavity steps | Multi-frame real field depth stability |
| **Explainable Quality Assessment** | 6-component weighted evidence index | Det-only, Geom-only, `D_missing_pen`, `E_cons_gated` | `result_quality_geo_maint.json` (`300` real obs + `4` probes) | Separates `INVALID`/`EXPIRED` (`73` obs); exposes diagnostic codes | Renormalizes over missing temporal history (`13.3%` `HIGH` → `MOD` under E) | Human quality rating ground truth |
| **Relative 2.5D Visualization** | Downsampled residual grids + ref/cavity masks | Unmasked legacy snapshots (`42` old JSONs) | `result_geometry.json`, `test_phase10_*.py` | `0.000396` mean resampling diff, `0` extra depth calls, fixed `valid_ring_mask` | Older `42` saved JSONs lack ref masks; strictly uncalibrated | Optional UX study with engineers |
| **Geospatial Observation Assoc.** | Timestamp interp + gap reject + Haversine cluster | Synthetic pair fixtures | `result_quality_geo_maint.json`, `test_phase11_*.py` | Clean gap rejection (`>2s`), duplicate merge, heading checks | Synthetic fixtures only; camera ≠ pothole center | Real multi-pass dashcam + RTK GPS |
| **Maintenance Intelligence** | Heuristic triage + SQLite review audit + segments | Stateless job tables (`Phase 8B`) | `test_phase12_*.py`, real `14`-job smoke check | Persistent review history (`UNIQUE` guarded), HTML/CSV/JSON/GeoJSON exports | Heuristic priority ≠ PCI or repair urgency | Civil-engineer ground-truth triage |
