# Dummy, Mock, Synthetic & Hardcoded Data Authenticity Audit

This document classifies every mock, dummy, fallback, synthetic, simulated, or hardcoded construct discovered across source code, UI components, databases, and evaluation scripts into Categories **A–G**:
- **A:** Genuine production functionality
- **B:** Legitimate synthetic scientific experimentation
- **C:** Legitimate unit-test mocking
- **D:** Clearly labeled demonstration data
- **E:** Unused legacy code / dead files
- **F:** Placeholder or unfinished functionality
- **G:** Misleading or unisolated artificial data reachable from normal application or scientific evaluation

---

## 1. Summary Table of Findings

| ID | Category | File & Line | Construct / Keyword | Reachable in Production App? | Affects Scientific Results? | Status / Corrective Action |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **DM-01** | **G** | `server.py:L45`, `BenchmarkComparison.jsx:L10-62, L244-246, L307-309` | Hardcoded V1 claims (`84.7% mAP50`, `92.4% precision`), YOLOv8s 11.13M vs 5.96 MB size contradiction, `261.5` vs `256.5 ms` prose discrepancy | Yes (Legacy V1 tab & `/api/health`) | Critical if cited in manuscript | Mark Legacy V1 UI cards explicitly as historical/unreconciled; NEVER cite in Phase 14 paper. |
| **DM-02** | **G** | `server.py:L162-167` | Legacy `/api/detect` dict fallbacks (`smooth_cm=6.5`, `width=30.0`, `length=25.0`, `volume=1.0`, `asphalt=2.3`, `conf=0.85`) | Yes (Legacy `/api/detect` if smoother keys missing) | No (V1 legacy only) | Replace fallback defaults with `None` in legacy endpoint. |
| **DM-03** | **G** | `runs/v2_api/roadvision_v2_jobs.db` + `Overview.jsx` | DB stores `blank_negative_control.mp4`, `cancel_test.mp4`, `upload_1794546659_test_clip.mp4`, `synthetic_trip_a/b.csv` mixed with real videos | **Yes** (`Overview` & `Maintenance` aggregate all 16 jobs) | Yes if DB KPI totals cited | Isolate test DB / add `is_test_fixture` column; clean demo DB before presentation. |
| **DM-04** | **G** | `roadvision_api_v2/storage.py:L86-110`, `geospatial_store.py:L250-486` | `synthetic_trip_a.csv` & `synthetic_trip_b.csv` persisted in `geospatial_clusters` with **`contains_synthetic=0`** | **Yes** (Geospatial & Maintenance UI) | Yes (synthetic cluster omits badge & could get cross-trip boost) | Add `is_synthetic` column/flag for GPS uploads; auto-flag synthetic fixtures. |
| **DM-05** | **G** | `experiments/phase13/quality_geo_maint_eval.py:L205, L216-230, L248-254` | `NC_blank_video` creates `blank.mp4` but skips detector/pipeline execution (hardcodes outcome); `FC_p9` hardcodes `snap_t15`; `synth_tracks` sets wrong key `contains_synthetic` instead of `is_synthetic` | No | Yes (`failure_cases.csv`) | Fix Phase 13 harness to actually execute pipeline on `blank.mp4` and use `is_synthetic`. |
| **DM-06** | **F / G** | `depth_ground_truth_manifest.csv` + `validate_physical_depth.py:L404-417` | `P01`/`P02` numeric values identical to `generate_synthetic_test_data()` `T01`/`T02`; `k_disp=0.082` uncertified | No (V2 ignores V1 calibration) | Blocked if metric cm claimed | Explicitly excluded from V2; documented as uncertified historical artifact. |
| **DM-07** | **E** | `roadvision-dashboard/src/data/sample_demo_potholes.json` (L1-68) | Static Bengaluru pothole array (`P-001`..`P-006`, lat `12.9718`) | No (0 imports in `src/`) | No | Unused leftover demo JSON; safe to delete in cleanup. |
| **DM-08** | **A** | `roadvision_engine.py:L308-365` | `DashcamKinematicGPS` returns `(None, None, t)` when `has_real_gps=False` (dead-reckoning only fires when `.srt` is loaded) | Yes (V1 engine) | Safe | Properly guards against fabricated GPS when `.srt` is absent. |
| **DM-09** | **A** | `roadvision_v2/geometry/surface.py:L212-214`, `roadvision_v2/pipeline.py:L505-511` | `None` / `NaN` returned when measurements invalid or depth unavailable; fallback warning explicit | Yes (V2 runtime) | Safe | Genuine zero vs unavailable properly separated via `None` and `measurement_state`. |
| **DM-10** | **B** | `experiments/phase13/geometry_ablation.py`, `experiments/phase13/scheduler_temporal_ablation.py:L221-253`, `experiments/test_relative_depth_synthetic.py` | Synthetic mathematical 3D road planes, synthetic step/gap signal, synthetic geospatial matching coordinates | No | Legitimate synthetic validation | Clearly labeled `synthetic` in JSON/CSV/tables. |
| **DM-11** | **C** | `tests/test_v2_*.py`, `tests/test_phase*.py` | Temp dirs, synthetic black MP4s, dummy pipeline factories for error/OOM/lock/race testing | No | None | Legitimate unit/integration test isolation alongside real GPU pipeline tests. |
| **DM-12** | **B** | `experiments/phase13/scheduler_temporal_ablation.py:L171-217` | Frozen-stream simulated latency (`simulated_frame_ms_mean`, `simulated_sustained_fps`) | No | Clearly labeled simulated | Distinguished from measured E2E worker FPS (`14.67`–`30.6` FPS). |

---

## 2. Detailed Reachability Conclusions
1. **Do artificial numbers reach V2 normal user outputs?**
   - **Relative Depth, Quality, Geometry Grids:** **NO** — verified across all 16 existing jobs and Tests A–D that V2 relative cavity scores, quality indices, and 2.5D signed residual grids come strictly from genuine PyTorch YOLOv8n + MiDaS / Depth Anything V2 CUDA execution, returning `None` (`null`) whenever gates fail.
   - **Overview & Maintenance Aggregates (YES — Hygiene Issue):** **YES**, because `runs/v2_api/roadvision_v2_jobs.db` stores prior test clips (`blank_negative_control.mp4`, `cancel_test.mp4`, `clip.mp4`, `upload_1794546659_test_clip.mp4`) and synthetic GPS uploads (`synthetic_trip_a/b.csv`), causing `Overview` KPIs (`totalJobsCount=16`, `totalRecentObservations`) and `Geospatial` clusters (`648c2b99` with `contains_synthetic=0`) to mix test fixtures with genuine video runs.
   - **Legacy V1 Suite Tab (YES — Unreconciled Legacy UI):** `BenchmarkComparison.jsx` shows hardcoded `84.7% mAP50` and `92.4% Precision` that do not reconcile with `runs/detect/pothole_rdd1666_50e/results.csv` (`0.45765` val mAP50).
