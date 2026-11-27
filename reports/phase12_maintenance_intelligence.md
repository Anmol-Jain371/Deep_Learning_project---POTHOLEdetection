# Phase 12 — Road Maintenance Intelligence Report

**Date:** 2026-11-27  
**Pipeline Scoring Version:** `heuristic_review_priority_v1`  
**Status:** Implemented, unit-tested, and verified against live completed inspection artifacts.

---

## 1. Summary of Implemented Architecture

Phase 12 extends RoadVision AI 2.0 with evidence-aware maintenance triage, persistent human-in-the-loop review workflows, road-segment analytics, historical inspection comparison, and downloadable multi-format reports without modifying the perception pipeline or running additional inference.

### Core Modules Created & Updated
- **Analytics & Reports (`roadvision_v2/maintenance/`):**
  - `roadvision_v2/maintenance/analytics.py`: `compute_review_priority`, `aggregate_job_maintenance`, `build_road_segment_analytics`, `compare_inspections`.
  - `roadvision_v2/maintenance/reports.py`: `build_inspection_report_payload`, `render_inspection_report_html`, `render_inspection_report_csv`.
  - `roadvision_v2/config.py`: `MaintenancePriorityConfig` dataclass recording weights, thresholds, and scoring version.
- **Database Migration & Persistence (`roadvision_api_v2/`):**
  - `roadvision_api_v2/storage.py`: Additive, idempotent migration `001_phase12_manual_reviews` tracked via `schema_migrations`, creating `manual_reviews` (`UNIQUE(job_id, track_id, track_generation_id)`) and `manual_review_history`.
  - `roadvision_api_v2/maintenance_store.py`: `MaintenanceStore` orchestrating parameterized queries, immutable `automated_evidence_json` snapshots on review creation, history tracking on `PATCH`, safe artifact reads, and report compilation.
  - `roadvision_api_v2/routes.py` & `roadvision_api_v2/openapi.yaml`: Exposed 9 maintenance/report endpoints and added `PATCH` to V2 CORS allowed methods.
- **Dashboard Integration (`roadvision-dashboard/src/`):**
  - `roadvision-dashboard/src/pages/MaintenanceIntelligence.jsx`: Overview, Review Queue (with filters, geometry thumbnails, quality breakdowns, and review workflow), Road Segment Analytics, and Historical Comparison.
  - `roadvision-dashboard/src/pages/Reports.jsx`: Extended with HTML, JSON, CSV, GeoJSON, and print-ready PDF exports plus Maintenance Priority & Manual Review summary cards.

---

## 2. REST API Endpoints Added

1. `GET /api/v2/maintenance/overview`
2. `GET /api/v2/maintenance/review-queue` (paginated; filters `job_id`, `priority_category`, `review_status`, `quality_category`)
3. `GET /api/v2/maintenance/reviews` / `POST /api/v2/maintenance/reviews` (enforces `UNIQUE(job_id, track_id, track_generation_id)` returning `409 DUPLICATE_REVIEW`)
4. `GET /api/v2/maintenance/reviews/<review_id>` / `PATCH /api/v2/maintenance/reviews/<review_id>` (preserves immutable automated evidence & appends audit history)
5. `GET /api/v2/jobs/<job_id>/maintenance`
6. `GET /api/v2/jobs/<job_id>/segments` (configurable `segment_length_m` in `[10, 1000]`)
7. `GET /api/v2/maintenance/compare`
8. `GET /api/v2/jobs/<job_id>/report` (`format=json|html|csv|geojson`, optional `download=1`)

---

## 3. Scientific & Engineering Safeguards

- **Priority vs. Sufficiency Separation:** Persistent detections with missing, invalid, or expired depth measurements are flagged `MANUAL_VERIFICATION_REQUIRED` rather than downgraded as harmless.
- **Non-Metric Semantics:** Categories (`PRIORITY_REVIEW`, `STANDARD_REVIEW`, `INSUFFICIENT_EVIDENCE`, `MANUAL_VERIFICATION_REQUIRED`, `REVIEWED`) reflect inspection-review urgency only—NOT structural pavement damage severity or Pavement Condition Index (PCI).
- **Human Review Traceability:** Manual review states (`PENDING_REVIEW`, `IN_REVIEW`, `CONFIRMED_VISUALLY`, `REJECTED_FALSE_POSITIVE`, `NEEDS_FIELD_INSPECTION`) never overwrite `automated_evidence_json`; `CONFIRMED_VISUALLY` records human visual confirmation only. No fake reviewer identities are invented.
- **Segment & Comparison Guardrails:** Missing GPS produces `geographical_analysis_unavailable: true` with zero fabricated coordinates. Zero-length GPS trajectories suppress route-length division. Cross-inspection comparisons block raw relative-depth comparison across incompatible depth models/resolutions/precisions/strategies.

---

## 4. Phase 12 Verification & Real-Artifact Smoke Check

- **Unit & Integration Tests:** `python -m unittest tests/test_phase12_maintenance_intelligence.py -v` → **5/5 tests passed (`OK` in 2.004s)**.
- **Real Completed Database Verification (14 real completed jobs, isolated temp DB copy):**
  - `14` completed inspections, `189` session tracks, `2,839` observations.
  - Priority distribution: `PRIORITY_REVIEW: 27`, `MANUAL_VERIFICATION_REQUIRED: 68`, `STANDARD_REVIEW: 86`, `INSUFFICIENT_EVIDENCE: 8`, `REVIEWED: 0`.
  - Quality distribution: `HIGH_QUALITY: 54`, `MODERATE_QUALITY: 1`, `INVALID: 19`, `INSUFFICIENT_EVIDENCE: 113`, `NOT_EVALUATED: 2`.
  - Verified review creation, `CONFIRMED_VISUALLY` persistence across Flask app re-initialization, historical comparison compatibility warnings (`relative_depth_compatible: false`), and HTML/JSON/CSV report generation (`36,577 B` HTML, `464,564 B` JSON, `8,404 B` CSV).
- **Frontend Verification:** `oxlint` → **0 errors** (`52` pre-existing warnings); `vite build` → succeeded in **676ms**.
