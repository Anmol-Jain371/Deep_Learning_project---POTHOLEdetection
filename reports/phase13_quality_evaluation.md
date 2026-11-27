# Phase 13 — Measurement Quality, Geospatial & Maintenance Evaluation

**Run ID:** `phase13_full_20261127`  
**Artifacts:** `result_quality_geo_maint.json`, `figures/fig5_quality_estimator_ablation.png`

---

## 1. Phase 9 Quality Estimator Ablation (Deployed Estimator Kept Unchanged)

Evaluated across controlled synthetic edge-case probes and `N = 300` real exploratory observations from `0f286074-9655-4d5a-abb8-fab934954e84`:

### Synthetic Probes (`QualityConfig(v1_deployed)`)
| Probe Scenario | `A_det_only` | `B_geom_only` | `C_deployed` (idx) | `D_missing_penalized` (idx) | `E_conservative_gated` (idx) |
|---|---|---|---|---|---|
| `ideal_full_evidence` | `HIGH_QUALITY` | `HIGH_QUALITY` | `HIGH_QUALITY` (`0.972`) | `HIGH_QUALITY` (`0.972`) | `HIGH_QUALITY` (`0.972`) |
| `low_det_conf_boundary_clipped_missing_temporal` | `LOW_QUALITY` | `HIGH_QUALITY` | **`HIGH_QUALITY` (`0.8161`)** | `MODERATE_QUALITY` (`0.6937`) | **`MODERATE_QUALITY` (`0.6937`)** |
| `invalid_high_det_confidence` | `INVALID` | `INVALID` | `INVALID` (`null`) | `INVALID` (`null`) | `INVALID` (`null`) |
| `expired_cached_observation` | `INVALID` | `INVALID` | `INVALID` (`null`) | `INVALID` (`null`) | `INVALID` (`null`) |

### Exploratory Real-Session Distribution (`N = 300` observations from job `0f286074`)
- **`C_deployed`:** `HIGH_QUALITY = 145`, `MODERATE_QUALITY = 51`, `INVALID = 73`, `INSUFFICIENT_EVIDENCE = 31`
- **`D_missing_penalized`:** `HIGH_QUALITY = 158`, `MODERATE_QUALITY = 38`, `INVALID = 73`, `INSUFFICIENT_EVIDENCE = 31`
- **`E_conservative_gated`:** **`HIGH_QUALITY = 105`**, **`MODERATE_QUALITY = 91`**, `INVALID = 73`, `INSUFFICIENT_EVIDENCE = 31`

**Finding:** `40` observations (`13.3%` of `300`) downgraded from `HIGH_QUALITY` to `MODERATE_QUALITY` under variant `E` (`conservative_gated`) because `C_deployed` renormalizes missing `temporal_consistency` weight (`0.15`) proportionally across remaining components and imposes no category cap when `detection_quality < 0.50`, boundary truncation occurs, or temporal evidence is missing. Independent human quality labels do not exist (`calibration/accuracy` NOT claimed).

## 2. Phase 11 Synthetic Geospatial Verification (`field_validation_status = BLOCKED`)
- Duplicate timestamps merged (`1`), linear interpolation at `0.5s` → `INTERPOLATED`, `>2.0s` sync gap at `3.0s` → `UNAVAILABLE`, geodesic Haversine distance verified (`9.969 m` on test offset), synthetic clustering produced `[POTENTIAL_MATCH, UNMATCHED, UNMATCHED]`. Genuine multi-trip field validation remains **`BLOCKED`**.

## 3. Phase 12 Maintenance Evaluation (`expert_severity_validation_status = BLOCKED`)
- Deterministic priority rules verified: `PRIORITY_REVIEW` vs `MANUAL_VERIFICATION_REQUIRED` for invalid/expired/missing-depth persistent detections, SQLite `UNIQUE` review duplicate rejection (`409`), immutable `automated_evidence_json` audit preservation across `PATCH`, restart persistence verified on `14` real completed jobs (`189` tracks, `2,839` observations), and zero fictional GPS segments when telemetry is absent.
