# Phase 8B.4 — REST API Contracts, Progress Polling, and Artifact Management

## 1. Implemented Endpoints (`roadvision_api_v2/routes.py`)

- **System**: `GET /api/v2/health`, `GET /api/v2/modes`, `GET /api/v2/capabilities`
- **Jobs**: `POST /api/v2/jobs` (`202`), `GET /api/v2/jobs` (`200`), `GET /api/v2/jobs/<job_id>` (`200`), `POST /api/v2/jobs/<job_id>/cancel` (`202`)
- **Results** (`200` when `COMPLETED`, `409` while active/non-completed):
  - `GET /api/v2/jobs/<job_id>/summary`
  - `GET /api/v2/jobs/<job_id>/observations`
  - `GET /api/v2/jobs/<job_id>/tracks`
  - `GET /api/v2/jobs/<job_id>/performance`
- **Artifacts**:
  - `GET /api/v2/jobs/<job_id>/artifacts` (`200`)
  - `GET /api/v2/jobs/<job_id>/artifacts/<artifact_name>` (`200` / `206 Partial Content` / `416`)

## 2. Progress Monitoring & Pagination

- Polling (`GET /api/v2/jobs/<job_id>`) excludes large per-frame observation arrays and throttles DB writes (`progress_update_interval=0.5s`).
- `processed_frames` is strictly monotonic; `progress_percentage` returns `null` if `total_frames` is unknown/unreliable.
- Observations, tracks, and performance rows support bounded `page` & `page_size` (`default=100`, `max=500`).
- Observation filters: `track_id`, `min_frame`, `max_frame`, `measurement_state` (`FRESH_VALID`, `FRESH_INVALID`, `CACHED_VALID`, `EXPIRED`, `NOT_EVALUATED`), `min_conf`, `max_conf`.

## 3. Scientific Safeguards & Artifact Security

- Every observation, summary, and capabilities response preserves uncalibrated relative inverse-depth semantics; no cm/mm/volume or physical severity claims are emitted.
- `compute_measurement_coverage()` reports `fresh_valid_rate_over_observations` and `fresh_valid_rate_over_keyframes` separately from `cached_valid_rate_over_observations` with explicit denominators.
- Unavailable metrics are serialized as JSON `null` (`allow_nan=False`).
- `roadvision_api_v2/artifacts.py` inspects actual `phase8a_<mode>_<stem>_*` files, builds a verified manifest, marks `partial: true` on cancelled/failed jobs, blocks symlinks and directory traversal (`realpath` containment), and supports HTTP byte-range streaming (`206`) for H.264 annotated MP4 video.
