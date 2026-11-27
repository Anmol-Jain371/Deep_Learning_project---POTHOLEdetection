# Phase 8B.3 — Video Uploads, Validation, and V2 Pipeline Integration

## 1. Upload & Option Validation (`roadvision_api_v2/validation.py`)

- Multipart upload requires `video` file field (`400` if missing or empty filename/stream).
- Extension checked against `allowed_video_formats` (`.mp4`, `.avi`, `.mov`, `.mkv`; `415` on unsupported format).
- Original filename is sanitized for display metadata only (`sanitize_display_filename`); disk storage uses server-generated `<job_id>_input<ext>` inside an isolated `uploads_v2/<job_id>/` directory.
- Streaming byte counter aborts and deletes partial uploads immediately if `maximum_upload_size` (256 MiB default) is exceeded (`413`).
- Real `cv2.VideoCapture` decode verification reads the first frame (`415` if corrupt/unreadable), enforces `maximum_video_resolution` (`1920x1080` default), and checks `maximum_video_duration` (`300.0`s default) when FPS/frame-count metadata are valid.
- Multipart options validated: `mode` (`live | balanced | survey`), `enable_depth`, `diagnostics`, `save_video`, `save_json`, `save_csv`, `max_frames` (`1..9000`). Unknown or sensitive overrides (`checkpoint_path`, `source`, `output_dir`, `device`, `weights`, `url`) are rejected with `400`.
- Cleanup guarantees remove temporary upload/output directories on validation failure or queue saturation (`429`).

## 2. Pipeline Execution & Backward-Compatible Hooks (`roadvision_v2/pipeline.py`)

- Added optional keyword callbacks to `RoadVisionV2Pipeline.process_video(...)` while keeping all existing Phase 8A / CLI signatures 100% compatible:
  - `on_init(meta)`
  - `on_progress(progress_dict)`
  - `should_cancel()`
  - `on_warning(msg)`
  - `on_complete(summary)`
- Cancellation checked at frame boundaries; `cap.release()` and `vid_writer.release()` always run in `try...finally:`.
- Distinguishes `scheduler_depth_requests` from `depth_calls` (actual successful depth executions) and `failed_depth_inferences`.
- Explicit fallback recorded (`fallback_used`, `fallback_reason`, warning emitted) if depth is disabled or fails to load.
- JSON artifacts written atomically via temp file + `os.replace()` with `allow_nan=False` (`NaN`/`Infinity` converted to `None` in `roadvision_v2/schemas.py`).
- Annotated MP4 transcoding (`transcode_h264=True`) uses `imageio-ffmpeg` (`libx264`, `yuv420p`, `+faststart`) and verifies OpenCV readability before replacing output.
