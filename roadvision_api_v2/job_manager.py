"""
Persistent inspection job manager and lifecycle state machine for RoadVision AI 2.0.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .errors import (
    BadRequestError,
    InvalidJobStateError,
    NotFoundError,
    QueueFullError,
)
from .schemas import (
    PIPELINE_VERSION,
    empty_measurement_state_counts,
    sanitize_json_value,
)
from .storage import JobDatabase

JOB_STATUS_QUEUED = "QUEUED"
JOB_STATUS_INITIALIZING = "INITIALIZING"
JOB_STATUS_RUNNING = "RUNNING"
JOB_STATUS_CANCELLING = "CANCELLING"
JOB_STATUS_COMPLETED = "COMPLETED"
JOB_STATUS_FAILED = "FAILED"
JOB_STATUS_CANCELLED = "CANCELLED"

ALL_JOB_STATUSES = (
    JOB_STATUS_QUEUED,
    JOB_STATUS_INITIALIZING,
    JOB_STATUS_RUNNING,
    JOB_STATUS_CANCELLING,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_CANCELLED,
)

ACTIVE_JOB_STATUSES = (
    JOB_STATUS_INITIALIZING,
    JOB_STATUS_RUNNING,
    JOB_STATUS_CANCELLING,
)

TERMINAL_JOB_STATUSES = (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_CANCELLED,
)

VALID_TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    JOB_STATUS_QUEUED: (JOB_STATUS_INITIALIZING, JOB_STATUS_CANCELLED, JOB_STATUS_FAILED),
    JOB_STATUS_INITIALIZING: (JOB_STATUS_RUNNING, JOB_STATUS_CANCELLING, JOB_STATUS_FAILED),
    JOB_STATUS_RUNNING: (JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLING),
    JOB_STATUS_CANCELLING: (JOB_STATUS_CANCELLED, JOB_STATUS_FAILED),
    JOB_STATUS_COMPLETED: (),
    JOB_STATUS_FAILED: (),
    JOB_STATUS_CANCELLED: (),
}


def utc_now_iso() -> str:
    """Return timezone-aware UTC ISO-8601 timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def validate_uuid(job_id: str) -> str:
    """Validate that job_id is a canonical UUID string; raise BadRequestError otherwise."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise BadRequestError("Missing job_id", error_code="INVALID_JOB_ID")
    try:
        parsed = uuid.UUID(job_id.strip())
    except ValueError as exc:
        raise BadRequestError("Invalid job_id format (expected UUID)", error_code="INVALID_JOB_ID") from exc
    return str(parsed)


def can_transition(current_status: str, next_status: str) -> bool:
    """Return True if transitioning current_status -> next_status is permitted."""
    return next_status in VALID_TRANSITIONS.get(current_status, ())


def _default_artifact_availability() -> Dict[str, bool]:
    return {
        "annotated_video": False,
        "frames_json": False,
        "tracks_json": False,
        "performance_csv": False,
        "session_summary_json": False,
    }


class JobManager:
    """Coordinates SQLite persistence and strict atomic lifecycle state transitions."""

    def __init__(
        self,
        db: JobDatabase,
        max_queued_jobs: int = 5,
        progress_update_interval: float = 0.5,
    ) -> None:
        self.db = db
        self.max_queued_jobs = int(max_queued_jobs)
        self.progress_update_interval = float(progress_update_interval)
        self._last_progress_write: Dict[str, float] = {}

    @staticmethod
    def _row_to_dict(row: Any, include_internal_paths: bool = False) -> Dict[str, Any]:
        d = dict(row)
        state_counts = json.loads(d.pop("measurement_state_counts_json") or "{}")
        config_snapshot = json.loads(d.pop("config_snapshot_json") or "{}")
        artifact_availability = json.loads(d.pop("artifact_availability_json") or "{}")
        manifest_raw = d.pop("artifact_manifest_json", None)
        artifact_manifest = json.loads(manifest_raw) if manifest_raw else None
        summary_raw = d.pop("summary_json", None)
        summary = json.loads(summary_raw) if summary_raw else None
        telemetry_raw = d.pop("telemetry_json", None)
        telemetry = json.loads(telemetry_raw) if telemetry_raw else None

        out = {
            "job_id": d["job_id"],
            "original_filename": d["original_filename"],
            "mode": d["mode"],
            "config_snapshot": config_snapshot,
            "status": d["status"],
            "created_at": d["created_at"],
            "started_at": d["started_at"],
            "completed_at": d["completed_at"],
            "updated_at": d["updated_at"],
            "total_frames": d["total_frames"],
            "processed_frames": d["processed_frames"],
            "detection_observation_count": d["detection_observation_count"],
            "unique_candidate_track_count": d["unique_candidate_track_count"],
            "depth_invocation_count": d["depth_invocation_count"],
            "scheduler_request_count": d["scheduler_request_count"],
            "failed_depth_count": d["failed_depth_count"],
            "measurement_state_counts": state_counts,
            "progress_percentage": d["progress_percentage"],
            "throughput_fps": d["throughput_fps"],
            "cancel_requested": bool(d["cancel_requested"]),
            "error_code": d["error_code"],
            "error_message": d["error_message"],
            "latest_warning": d["latest_warning"],
            "artifact_availability": artifact_availability,
            "artifact_manifest": artifact_manifest,
            "summary": summary,
            "telemetry": telemetry,
            "pipeline_version": d["pipeline_version"],
        }
        if include_internal_paths:
            out["validated_input_path"] = d["validated_input_path"]
            out["output_dir"] = d["output_dir"]
        return sanitize_json_value(out)

    def create_job(
        self,
        *,
        job_id: Optional[str] = None,
        original_filename: str,
        validated_input_path: str,
        mode: str,
        config_snapshot: Dict[str, Any],
        output_dir: str,
        total_frames: Optional[int] = None,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> Dict[str, Any]:
        """Atomically enforce queue limit and persist a new QUEUED job."""
        jid = validate_uuid(job_id) if job_id else str(uuid.uuid4())
        now = utc_now_iso()
        state_counts_json = json.dumps(empty_measurement_state_counts(), allow_nan=False)
        config_json = json.dumps(sanitize_json_value(config_snapshot), allow_nan=False)
        avail_json = json.dumps(_default_artifact_availability(), allow_nan=False)

        with self.db.transaction(immediate=True) as conn:
            q_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM inspection_jobs WHERE status = ?",
                (JOB_STATUS_QUEUED,),
            ).fetchone()
            queued_count = int(q_row["cnt"]) if q_row else 0
            if queued_count >= self.max_queued_jobs:
                raise QueueFullError(
                    f"Job queue capacity reached (max_queued_jobs={self.max_queued_jobs}). Try again later.",
                    details={"max_queued_jobs": self.max_queued_jobs, "queued_jobs": queued_count},
                )

            conn.execute(
                """
                INSERT INTO inspection_jobs (
                    job_id, original_filename, validated_input_path, mode, config_snapshot_json,
                    status, created_at, started_at, completed_at, updated_at,
                    total_frames, processed_frames, detection_observation_count,
                    unique_candidate_track_count, depth_invocation_count,
                    scheduler_request_count, failed_depth_count,
                    measurement_state_counts_json, progress_percentage, throughput_fps,
                    output_dir, cancel_requested, error_code, error_message, latest_warning,
                    artifact_manifest_json, artifact_availability_json, summary_json,
                    telemetry_json, pipeline_version
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, NULL, NULL, ?,
                    ?, 0, 0,
                    0, 0,
                    0, 0,
                    ?, NULL, NULL,
                    ?, 0, NULL, NULL, NULL,
                    NULL, ?, NULL,
                    NULL, ?
                )
                """,
                (
                    jid,
                    original_filename,
                    validated_input_path,
                    mode,
                    config_json,
                    JOB_STATUS_QUEUED,
                    now,
                    now,
                    total_frames,
                    state_counts_json,
                    output_dir,
                    avail_json,
                    pipeline_version,
                ),
            )
            row = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            return self._row_to_dict(row, include_internal_paths=True)

    def get_job(self, job_id: str, include_internal_paths: bool = False) -> Dict[str, Any]:
        """Fetch a single job by UUID or raise NotFoundError."""
        jid = validate_uuid(job_id)
        with self.db.read_connection() as conn:
            row = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Job '{jid}' not found", error_code="JOB_NOT_FOUND")
        return self._row_to_dict(row, include_internal_paths=include_internal_paths)

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List jobs ordered newest first with optional status filter and pagination."""
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        if status is not None:
            status = status.upper().strip()
            if status not in ALL_JOB_STATUSES:
                raise BadRequestError(
                    f"Unsupported status filter '{status}'",
                    error_code="INVALID_STATUS_FILTER",
                )

        with self.db.read_connection() as conn:
            if status is not None:
                total_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM inspection_jobs WHERE status = ?", (status,)
                ).fetchone()
                rows = conn.execute(
                    "SELECT * FROM inspection_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status, limit, offset),
                ).fetchall()
            else:
                total_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM inspection_jobs"
                ).fetchone()
                rows = conn.execute(
                    "SELECT * FROM inspection_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()

        total = int(total_row["cnt"]) if total_row else 0
        return {
            "jobs": [self._row_to_dict(r, include_internal_paths=False) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def count_jobs_by_status(self) -> Dict[str, int]:
        """Return counts grouped by lifecycle status."""
        counts = {s: 0 for s in ALL_JOB_STATUSES}
        with self.db.read_connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM inspection_jobs GROUP BY status"
            ).fetchall()
        for r in rows:
            counts[r["status"]] = int(r["cnt"])
        return counts

    def claim_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Atomically claim the oldest QUEUED job for execution if no active job is running.
        Transitions QUEUED -> INITIALIZING (or QUEUED -> CANCELLED if cancel_requested=1).
        """
        with self.db.transaction(immediate=True) as conn:
            active_row = conn.execute(
                "SELECT job_id FROM inspection_jobs WHERE status IN (?, ?, ?) LIMIT 1",
                ACTIVE_JOB_STATUSES,
            ).fetchone()
            if active_row is not None:
                return None

            while True:
                row = conn.execute(
                    "SELECT * FROM inspection_jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
                    (JOB_STATUS_QUEUED,),
                ).fetchone()
                if row is None:
                    return None

                now = utc_now_iso()
                jid = row["job_id"]
                if int(row["cancel_requested"]) == 1:
                    conn.execute(
                        """
                        UPDATE inspection_jobs
                        SET status = ?, completed_at = ?, updated_at = ?
                        WHERE job_id = ? AND status = ?
                        """,
                        (JOB_STATUS_CANCELLED, now, now, jid, JOB_STATUS_QUEUED),
                    )
                    continue

                cur = conn.execute(
                    """
                    UPDATE inspection_jobs
                    SET status = ?, started_at = ?, updated_at = ?
                    WHERE job_id = ? AND status = ? AND cancel_requested = 0
                    """,
                    (JOB_STATUS_INITIALIZING, now, now, jid, JOB_STATUS_QUEUED),
                )
                if cur.rowcount == 1:
                    updated = conn.execute(
                        "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
                    ).fetchone()
                    return self._row_to_dict(updated, include_internal_paths=True)
                return None

    def mark_running(
        self,
        job_id: str,
        total_frames: Optional[int] = None,
        latest_warning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transition INITIALIZING -> RUNNING (or preserve CANCELLING if cancelled during init)."""
        jid = validate_uuid(job_id)
        now = utc_now_iso()
        with self.db.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Job '{jid}' not found", error_code="JOB_NOT_FOUND")
            status = row["status"]
            if status in TERMINAL_JOB_STATUSES:
                raise InvalidJobStateError(f"Cannot mark terminal job '{jid}' as RUNNING")
            if status == JOB_STATUS_CANCELLING or int(row["cancel_requested"]) == 1:
                # Stay in CANCELLING so worker cooperates cleanly
                return self._row_to_dict(row, include_internal_paths=True)
            if not can_transition(status, JOB_STATUS_RUNNING):
                raise InvalidJobStateError(
                    f"Invalid transition {status} -> {JOB_STATUS_RUNNING} for job '{jid}'"
                )

            tf = total_frames if (total_frames is not None and total_frames > 0) else row["total_frames"]
            conn.execute(
                """
                UPDATE inspection_jobs
                SET status = ?, total_frames = COALESCE(?, total_frames),
                    latest_warning = COALESCE(?, latest_warning), updated_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (JOB_STATUS_RUNNING, tf, latest_warning, now, jid, JOB_STATUS_INITIALIZING),
            )
            updated = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            return self._row_to_dict(updated, include_internal_paths=True)

    def update_progress(
        self,
        job_id: str,
        *,
        processed_frames: int,
        total_frames: Optional[int] = None,
        detection_observation_count: int = 0,
        unique_candidate_track_count: int = 0,
        depth_invocation_count: int = 0,
        scheduler_request_count: int = 0,
        failed_depth_count: int = 0,
        measurement_state_counts: Optional[Dict[str, int]] = None,
        throughput_fps: Optional[float] = None,
        latest_warning: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Update job progress monotonically with DB write throttling.
        Returns True if a DB write was performed, False if throttled or skipped.
        """
        jid = validate_uuid(job_id)
        now_mono = time.monotonic()
        if not force:
            last = self._last_progress_write.get(jid, 0.0)
            if (now_mono - last) < self.progress_update_interval:
                return False

        now = utc_now_iso()
        with self.db.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT status, processed_frames, total_frames FROM inspection_jobs WHERE job_id = ?",
                (jid,),
            ).fetchone()
            if row is None or row["status"] in TERMINAL_JOB_STATUSES or row["status"] == JOB_STATUS_QUEUED:
                return False

            prev_processed = int(row["processed_frames"] or 0)
            mono_processed = max(prev_processed, int(processed_frames))
            effective_total = total_frames if (total_frames is not None and total_frames > 0) else row["total_frames"]
            if effective_total is not None and int(effective_total) > 0:
                pct: Optional[float] = round(min(100.0, (mono_processed / int(effective_total)) * 100.0), 2)
            else:
                pct = None

            counts_json = (
                json.dumps(sanitize_json_value(measurement_state_counts), allow_nan=False)
                if measurement_state_counts is not None
                else None
            )
            fps_val = sanitize_json_value(round(float(throughput_fps), 2)) if throughput_fps is not None else None

            conn.execute(
                """
                UPDATE inspection_jobs
                SET processed_frames = ?,
                    total_frames = COALESCE(?, total_frames),
                    detection_observation_count = MAX(detection_observation_count, ?),
                    unique_candidate_track_count = MAX(unique_candidate_track_count, ?),
                    depth_invocation_count = MAX(depth_invocation_count, ?),
                    scheduler_request_count = MAX(scheduler_request_count, ?),
                    failed_depth_count = MAX(failed_depth_count, ?),
                    measurement_state_counts_json = COALESCE(?, measurement_state_counts_json),
                    progress_percentage = ?,
                    throughput_fps = COALESCE(?, throughput_fps),
                    latest_warning = COALESCE(?, latest_warning),
                    updated_at = ?
                WHERE job_id = ? AND status IN (?, ?, ?)
                """,
                (
                    mono_processed,
                    effective_total,
                    int(detection_observation_count),
                    int(unique_candidate_track_count),
                    int(depth_invocation_count),
                    int(scheduler_request_count),
                    int(failed_depth_count),
                    counts_json,
                    pct,
                    fps_val,
                    latest_warning,
                    now,
                    jid,
                    JOB_STATUS_INITIALIZING,
                    JOB_STATUS_RUNNING,
                    JOB_STATUS_CANCELLING,
                ),
            )
            self._last_progress_write[jid] = now_mono
            return True

    def record_warning(self, job_id: str, warning_message: str) -> None:
        """Persist latest processing warning for a non-terminal job."""
        jid = validate_uuid(job_id)
        now = utc_now_iso()
        with self.db.transaction(immediate=True) as conn:
            conn.execute(
                """
                UPDATE inspection_jobs
                SET latest_warning = ?, updated_at = ?
                WHERE job_id = ? AND status NOT IN (?, ?, ?)
                """,
                (str(warning_message)[:300], now, jid, *TERMINAL_JOB_STATUSES),
            )

    def is_cancellation_requested(self, job_id: str) -> bool:
        """Check whether cooperative cancellation has been requested for a job."""
        jid = validate_uuid(job_id)
        with self.db.read_connection() as conn:
            row = conn.execute(
                "SELECT status, cancel_requested FROM inspection_jobs WHERE job_id = ?",
                (jid,),
            ).fetchone()
        if row is None:
            return False
        return bool(row["cancel_requested"]) or row["status"] in (
            JOB_STATUS_CANCELLING,
            JOB_STATUS_CANCELLED,
        )

    def request_cancellation(self, job_id: str) -> Dict[str, Any]:
        """
        Atomically request job cancellation.
        - QUEUED -> CANCELLED immediately.
        - INITIALIZING / RUNNING -> CANCELLING (worker cancels cooperatively).
        - CANCELLING -> idempotent return.
        - COMPLETED / FAILED / CANCELLED -> raises InvalidJobStateError (409).
        """
        jid = validate_uuid(job_id)
        now = utc_now_iso()
        with self.db.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Job '{jid}' not found", error_code="JOB_NOT_FOUND")
            status = row["status"]

            if status in TERMINAL_JOB_STATUSES:
                raise InvalidJobStateError(
                    f"Cannot cancel job '{jid}' because it is already in terminal state '{status}'",
                    details={"status": status},
                )

            if status == JOB_STATUS_QUEUED:
                conn.execute(
                    """
                    UPDATE inspection_jobs
                    SET status = ?, cancel_requested = 1, completed_at = ?, updated_at = ?
                    WHERE job_id = ? AND status = ?
                    """,
                    (JOB_STATUS_CANCELLED, now, now, jid, JOB_STATUS_QUEUED),
                )
            elif status in (JOB_STATUS_INITIALIZING, JOB_STATUS_RUNNING):
                conn.execute(
                    """
                    UPDATE inspection_jobs
                    SET status = ?, cancel_requested = 1, updated_at = ?
                    WHERE job_id = ? AND status = ?
                    """,
                    (JOB_STATUS_CANCELLING, now, jid, status),
                )
            elif status == JOB_STATUS_CANCELLING:
                conn.execute(
                    "UPDATE inspection_jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
                    (now, jid),
                )

            updated = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            return self._row_to_dict(updated, include_internal_paths=False)

    def complete_job(
        self,
        job_id: str,
        *,
        processed_frames: int,
        total_frames: Optional[int],
        detection_observation_count: int,
        unique_candidate_track_count: int,
        depth_invocation_count: int,
        scheduler_request_count: int,
        failed_depth_count: int,
        measurement_state_counts: Dict[str, int],
        throughput_fps: Optional[float],
        artifact_manifest: Dict[str, Any],
        artifact_availability: Dict[str, bool],
        summary: Dict[str, Any],
        telemetry: Optional[Dict[str, Any]] = None,
        latest_warning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Atomically finalize a job.
        Resolves completion-vs-cancellation races atomically:
        if cancel_requested == 1 or status == CANCELLING, finalizes as CANCELLED instead of COMPLETED.
        Never overwrites an already-terminal job.
        """
        jid = validate_uuid(job_id)
        now = utc_now_iso()
        with self.db.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Job '{jid}' not found", error_code="JOB_NOT_FOUND")
            status = row["status"]
            if status in TERMINAL_JOB_STATUSES:
                return self._row_to_dict(row, include_internal_paths=False)

            was_cancelled = (status == JOB_STATUS_CANCELLING) or (int(row["cancel_requested"]) == 1)
            target_status = JOB_STATUS_CANCELLED if was_cancelled else JOB_STATUS_COMPLETED
            if not can_transition(status, target_status):
                raise InvalidJobStateError(
                    f"Cannot finalize job '{jid}' from state '{status}' to '{target_status}'"
                )

            mono_processed = max(int(row["processed_frames"] or 0), int(processed_frames))
            eff_total = total_frames if (total_frames is not None and total_frames > 0) else row["total_frames"]
            if target_status == JOB_STATUS_COMPLETED and (eff_total is None or eff_total <= 0):
                eff_total = mono_processed
            pct: Optional[float]
            if target_status == JOB_STATUS_COMPLETED and eff_total and eff_total > 0:
                pct = 100.0
            elif eff_total and eff_total > 0:
                pct = round(min(100.0, (mono_processed / eff_total) * 100.0), 2)
            else:
                pct = None

            conn.execute(
                """
                UPDATE inspection_jobs
                SET status = ?,
                    completed_at = ?,
                    updated_at = ?,
                    processed_frames = ?,
                    total_frames = COALESCE(?, total_frames),
                    detection_observation_count = MAX(detection_observation_count, ?),
                    unique_candidate_track_count = MAX(unique_candidate_track_count, ?),
                    depth_invocation_count = MAX(depth_invocation_count, ?),
                    scheduler_request_count = MAX(scheduler_request_count, ?),
                    failed_depth_count = MAX(failed_depth_count, ?),
                    measurement_state_counts_json = ?,
                    progress_percentage = ?,
                    throughput_fps = ?,
                    artifact_manifest_json = ?,
                    artifact_availability_json = ?,
                    summary_json = ?,
                    telemetry_json = ?,
                    latest_warning = COALESCE(?, latest_warning),
                    error_code = CASE WHEN ? = 'CANCELLED' THEN 'JOB_CANCELLED' ELSE error_code END,
                    error_message = CASE WHEN ? = 'CANCELLED' THEN 'Job cancelled cooperatively' ELSE error_message END
                WHERE job_id = ? AND status = ?
                """,
                (
                    target_status,
                    now,
                    now,
                    mono_processed,
                    eff_total,
                    int(detection_observation_count),
                    int(unique_candidate_track_count),
                    int(depth_invocation_count),
                    int(scheduler_request_count),
                    int(failed_depth_count),
                    json.dumps(sanitize_json_value(measurement_state_counts), allow_nan=False),
                    pct,
                    sanitize_json_value(round(float(throughput_fps), 2)) if throughput_fps is not None else None,
                    json.dumps(sanitize_json_value(artifact_manifest), allow_nan=False),
                    json.dumps(sanitize_json_value(artifact_availability), allow_nan=False),
                    json.dumps(sanitize_json_value(summary), allow_nan=False),
                    json.dumps(sanitize_json_value(telemetry), allow_nan=False) if telemetry is not None else None,
                    latest_warning,
                    target_status,
                    target_status,
                    jid,
                    status,
                ),
            )
            self._last_progress_write.pop(jid, None)
            updated = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            return self._row_to_dict(updated, include_internal_paths=False)

    def finalize_cancelled_job(
        self,
        job_id: str,
        *,
        processed_frames: int = 0,
        total_frames: Optional[int] = None,
        detection_observation_count: int = 0,
        unique_candidate_track_count: int = 0,
        depth_invocation_count: int = 0,
        scheduler_request_count: int = 0,
        failed_depth_count: int = 0,
        measurement_state_counts: Optional[Dict[str, int]] = None,
        throughput_fps: Optional[float] = None,
        artifact_manifest: Optional[Dict[str, Any]] = None,
        artifact_availability: Optional[Dict[str, bool]] = None,
        summary: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transition an active/cancelling job to CANCELLED cleanly without overwriting COMPLETED/FAILED."""
        jid = validate_uuid(job_id)
        now = utc_now_iso()
        with self.db.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Job '{jid}' not found", error_code="JOB_NOT_FOUND")
            status = row["status"]
            if status in TERMINAL_JOB_STATUSES:
                return self._row_to_dict(row, include_internal_paths=False)

            mono_processed = max(int(row["processed_frames"] or 0), int(processed_frames))
            eff_total = total_frames if (total_frames is not None and total_frames > 0) else row["total_frames"]
            pct = (
                round(min(100.0, (mono_processed / eff_total) * 100.0), 2)
                if (eff_total and eff_total > 0)
                else None
            )

            conn.execute(
                """
                UPDATE inspection_jobs
                SET status = ?,
                    cancel_requested = 1,
                    completed_at = ?,
                    updated_at = ?,
                    processed_frames = ?,
                    total_frames = COALESCE(?, total_frames),
                    detection_observation_count = MAX(detection_observation_count, ?),
                    unique_candidate_track_count = MAX(unique_candidate_track_count, ?),
                    depth_invocation_count = MAX(depth_invocation_count, ?),
                    scheduler_request_count = MAX(scheduler_request_count, ?),
                    failed_depth_count = MAX(failed_depth_count, ?),
                    measurement_state_counts_json = COALESCE(?, measurement_state_counts_json),
                    progress_percentage = ?,
                    throughput_fps = COALESCE(?, throughput_fps),
                    artifact_manifest_json = COALESCE(?, artifact_manifest_json),
                    artifact_availability_json = COALESCE(?, artifact_availability_json),
                    summary_json = COALESCE(?, summary_json),
                    telemetry_json = COALESCE(?, telemetry_json),
                    error_code = 'JOB_CANCELLED',
                    error_message = 'Job cancelled cooperatively'
                WHERE job_id = ? AND status = ?
                """,
                (
                    JOB_STATUS_CANCELLED,
                    now,
                    now,
                    mono_processed,
                    eff_total,
                    int(detection_observation_count),
                    int(unique_candidate_track_count),
                    int(depth_invocation_count),
                    int(scheduler_request_count),
                    int(failed_depth_count),
                    json.dumps(sanitize_json_value(measurement_state_counts), allow_nan=False)
                    if measurement_state_counts is not None
                    else None,
                    pct,
                    sanitize_json_value(round(float(throughput_fps), 2)) if throughput_fps is not None else None,
                    json.dumps(sanitize_json_value(artifact_manifest), allow_nan=False)
                    if artifact_manifest is not None
                    else None,
                    json.dumps(sanitize_json_value(artifact_availability), allow_nan=False)
                    if artifact_availability is not None
                    else None,
                    json.dumps(sanitize_json_value(summary), allow_nan=False)
                    if summary is not None
                    else None,
                    json.dumps(sanitize_json_value(telemetry), allow_nan=False)
                    if telemetry is not None
                    else None,
                    jid,
                    status,
                ),
            )
            self._last_progress_write.pop(jid, None)
            updated = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            return self._row_to_dict(updated, include_internal_paths=False)

    def fail_job(
        self,
        job_id: str,
        *,
        error_code: str,
        error_message: str,
        telemetry: Optional[Dict[str, Any]] = None,
        artifact_manifest: Optional[Dict[str, Any]] = None,
        artifact_availability: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Transition a non-terminal job to FAILED without touching terminal jobs."""
        jid = validate_uuid(job_id)
        now = utc_now_iso()
        with self.db.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Job '{jid}' not found", error_code="JOB_NOT_FOUND")
            status = row["status"]
            if status in TERMINAL_JOB_STATUSES:
                return self._row_to_dict(row, include_internal_paths=False)
            if not can_transition(status, JOB_STATUS_FAILED):
                raise InvalidJobStateError(f"Cannot fail job '{jid}' from state '{status}'")

            conn.execute(
                """
                UPDATE inspection_jobs
                SET status = ?,
                    completed_at = ?,
                    updated_at = ?,
                    error_code = ?,
                    error_message = ?,
                    telemetry_json = COALESCE(?, telemetry_json),
                    artifact_manifest_json = COALESCE(?, artifact_manifest_json),
                    artifact_availability_json = COALESCE(?, artifact_availability_json)
                WHERE job_id = ? AND status = ?
                """,
                (
                    JOB_STATUS_FAILED,
                    now,
                    now,
                    str(error_code)[:64],
                    str(error_message)[:300],
                    json.dumps(sanitize_json_value(telemetry), allow_nan=False)
                    if telemetry is not None
                    else None,
                    json.dumps(sanitize_json_value(artifact_manifest), allow_nan=False)
                    if artifact_manifest is not None
                    else None,
                    json.dumps(sanitize_json_value(artifact_availability), allow_nan=False)
                    if artifact_availability is not None
                    else None,
                    jid,
                    status,
                ),
            )
            self._last_progress_write.pop(jid, None)
            updated = conn.execute(
                "SELECT * FROM inspection_jobs WHERE job_id = ?", (jid,)
            ).fetchone()
            return self._row_to_dict(updated, include_internal_paths=False)

    def recover_interrupted_jobs(self) -> Dict[str, int]:
        """
        Restart recovery policy:
        - COMPLETED, FAILED, CANCELLED remain untouched.
        - QUEUED jobs with cancel_requested=1 become CANCELLED; remaining QUEUED jobs
          stay QUEUED so the restarted worker can pick them up cleanly.
        - Interrupted INITIALIZING, RUNNING, and CANCELLING jobs are marked FAILED with
          error_code='WORKER_INTERRUPTED' (never silently resumed or marked COMPLETED).
        """
        now = utc_now_iso()
        with self.db.transaction(immediate=True) as conn:
            interrupted = conn.execute(
                """
                UPDATE inspection_jobs
                SET status = ?,
                    completed_at = ?,
                    updated_at = ?,
                    error_code = 'WORKER_INTERRUPTED',
                    error_message = 'Inspection job interrupted by backend restart before completion.'
                WHERE status IN (?, ?, ?)
                """,
                (
                    JOB_STATUS_FAILED,
                    now,
                    now,
                    JOB_STATUS_INITIALIZING,
                    JOB_STATUS_RUNNING,
                    JOB_STATUS_CANCELLING,
                ),
            )
            queued_cancelled = conn.execute(
                """
                UPDATE inspection_jobs
                SET status = ?,
                    completed_at = ?,
                    updated_at = ?,
                    error_code = 'JOB_CANCELLED',
                    error_message = 'Queued job cancelled before restart dispatch.'
                WHERE status = ? AND cancel_requested = 1
                """,
                (JOB_STATUS_CANCELLED, now, now, JOB_STATUS_QUEUED),
            )
            queued_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM inspection_jobs WHERE status = ?",
                (JOB_STATUS_QUEUED,),
            ).fetchone()

        return {
            "interrupted_marked_failed": int(interrupted.rowcount),
            "queued_cancelled_on_recovery": int(queued_cancelled.rowcount),
            "queued_preserved_for_dispatch": int(queued_row["cnt"]) if queued_row else 0,
        }
