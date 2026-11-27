# Phase 8B.2 — Persistent Job Management and Bounded GPU Worker

## 1. SQLite Persistence (`roadvision_api_v2/storage.py`)

- Uses Python's standard-library `sqlite3` module without external database servers or ORMs.
- Configured in **WAL mode** (`PRAGMA journal_mode=WAL;`, `PRAGMA synchronous=NORMAL;`, `PRAGMA busy_timeout=5000;`).
- Thread-safe access protected by a `threading.RLock()` and atomic `BEGIN IMMEDIATE` write transactions for job creation, queue-bound checks, job claiming, and terminal state transitions.
- Parameterized SQL queries across all operations.
- Indexes:
  - `idx_jobs_status` on `inspection_jobs(status)`
  - `idx_jobs_created_at` on `inspection_jobs(created_at DESC)`
  - `idx_jobs_status_created` on `inspection_jobs(status, created_at ASC)`

### `inspection_jobs` Schema Columns
- Identity & input: `job_id` (UUID TEXT PK), `original_filename`, `validated_input_path`, `mode`, `config_snapshot_json`, `output_dir`, `pipeline_version`
- Lifecycle & UTC timestamps: `status`, `created_at`, `started_at`, `completed_at`, `updated_at`, `cancel_requested`
- Progress & counts: `total_frames`, `processed_frames`, `detection_observation_count`, `unique_candidate_track_count`, `depth_invocation_count`, `scheduler_request_count`, `failed_depth_count`, `measurement_state_counts_json`, `progress_percentage`, `throughput_fps`
- Results & error safety: `error_code`, `error_message`, `latest_warning`, `artifact_manifest_json`, `artifact_availability_json`, `summary_json`, `telemetry_json`

## 2. Job Lifecycle (`roadvision_api_v2/job_manager.py`)

States: `QUEUED`, `INITIALIZING`, `RUNNING`, `CANCELLING`, `COMPLETED`, `FAILED`, `CANCELLED`.

Allowed transitions:
- `QUEUED` → `INITIALIZING` | `CANCELLED` | `FAILED`
- `INITIALIZING` → `RUNNING` | `CANCELLING` | `FAILED`
- `RUNNING` → `COMPLETED` | `FAILED` | `CANCELLING`
- `CANCELLING` → `CANCELLED` | `FAILED`
- Terminal states (`COMPLETED`, `FAILED`, `CANCELLED`) cannot be overwritten.

Race protection:
- `create_job()` atomically checks `COUNT(*) WHERE status = 'QUEUED' < max_queued_jobs` inside `BEGIN IMMEDIATE`, raising `QueueFullError` (HTTP 429) on saturation.
- `claim_next_job()` prevents duplicate execution by ensuring zero active jobs (`INITIALIZING`, `RUNNING`, `CANCELLING`) before claiming the oldest `QUEUED` job.
- `complete_job()` resolves completion-vs-cancellation races atomically: if `cancel_requested == 1` or `status == CANCELLING`, finalizes as `CANCELLED` instead of `COMPLETED`.

## 3. Bounded GPU Worker (`roadvision_api_v2/worker.py`) & GPU Lock (`roadvision_api_v2/gpu_lock.py`)

- Enforces `max_gpu_concurrency = 1` for 4 GB RTX 2050 VRAM safety.
- Never auto-started on package import; explicitly started in `server.py` `if __name__ == "__main__":` guarded against Werkzeug reloader duplicates (`not debug or WERKZEUG_RUN_MAIN == "true"`), with Flask debug disabled when V2 is enabled.
- `SingleProcessWorkerGuard` acquires an OS file lock (`.roadvision_v2_worker.lock` via `msvcrt.locking` on Windows) so duplicate local worker processes cannot dispatch jobs concurrently.
- `gpu_inference_lock()` serializes legacy `/api/detect` (`RoadVisionPipeline.process_video`) and V2 GPU jobs without modifying legacy response contracts.
- Instantiates a fresh `RoadVisionV2Pipeline` per job (only loading the depth model required by that mode), clears model references on exit, runs `gc.collect()`, and calls `torch.cuda.empty_cache()` (noting `empty_cache()` returns unreferenced cached blocks to CUDA and does not free live references).

## 4. Restart Recovery Policy

On worker startup (`recover_interrupted_jobs()`):
- `COMPLETED`, `FAILED`, and `CANCELLED` jobs are preserved untouched.
- Interrupted `INITIALIZING`, `RUNNING`, and `CANCELLING` jobs are transitioned to `FAILED` with `error_code = 'WORKER_INTERRUPTED'` (never silently resumed or marked completed).
- `QUEUED` jobs with `cancel_requested = 1` become `CANCELLED`; remaining `QUEUED` jobs are preserved for dispatch.
