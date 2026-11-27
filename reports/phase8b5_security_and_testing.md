# Phase 8B.5 — Security Boundaries and Automated Verification

## 1. Local Research Security Boundaries

- **Localhost binding**: `config/roadvision_api_v2.yaml` enforces loopback-only `host` (`127.0.0.1`); `APIConfig.validate()` rejects non-loopback bind hosts such as `0.0.0.0`.
- **Debug disabled**: `server.py` forces `debug=False` when V2 API is enabled.
- **CORS restriction**: `/api/v2/*` `after_request` hook removes any wildcard `Access-Control-Allow-Origin` and echoes `Origin` only when present in `allowed_cors_origins` (explicitly rejecting `*` in config validation).
- **Path & upload isolation**: Server-generated UUID job directories, streamed upload size cap, OpenCV real-decode validation, manifest-only artifact serving, symlink rejection, and `os.path.realpath` containment checks.
- **Error sanitization**: `APIError` and `sanitize_error_message()` strip directory paths and never return stack traces or environment variables.
- **Authentication limitation**: **No user authentication or authorization is implemented.** Any local client able to reach `127.0.0.1:5000` can list/poll jobs and download artifacts. UUID job IDs do not provide authorization. This backend is a local research prototype and must not be deployed publicly.

## 2. Automated Test Coverage

Added/updated V2 backend test modules:
- `tests/test_v2_api_foundation.py` — config defaults, env override, validation bounds, system endpoints, CLI `--help`, feature-flag toggle.
- `tests/test_v2_job_manager.py` — SQLite WAL mode, indexes, persistence, atomic claiming, transition guards, terminal protection, monotonic progress, queue saturation, restart recovery.
- `tests/test_v2_worker.py` — single-GPU concurrency, worker start/stop, duplicate worker lock, init/running cancellation, completion-vs-cancel race, GPU OOM & missing-model isolation, cleanup, legacy GPU lock coordination.
- `tests/test_v2_api_uploads.py` — multipart uploads, missing/empty/unsupported/corrupt/oversized files, invalid mode/params, forbidden overrides, unsafe filenames, cleanup on 429/errors.
- `tests/test_v2_api.py` — CRUD/polling endpoints, 404/409/400 contracts, observation/track/performance pagination & filters, separate fresh/cached coverage denominators, `NaN`/`Inf` -> `null`.
- `tests/test_v2_api_security.py` — path traversal, symlink escape, cross-job isolation, CORS origin enforcement, error sanitization.
- `tests/test_v2_api_artifacts.py` — verified manifest, HTTP 206 Range streaming, 416 invalid range, `partial: true` cancelled artifacts.
