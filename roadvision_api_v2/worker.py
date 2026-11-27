"""
Bounded single-GPU background worker for RoadVision AI 2.0 inspection jobs.

Guarantees:
- Explicit start() / stop() lifecycle (never auto-started at package import).
- Max 1 concurrent GPU inspection job, coordinated with legacy /api/detect via gpu_inference_lock().
- Per-job isolated RoadVisionV2Pipeline instance (no tracker or cache leakage across jobs).
- Cooperative cancellation checked at init and frame boundaries.
- Restart recovery called explicitly before dispatching queued work.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import torch

from roadvision_v2.config import RoadVisionV2Config, load_v2_config
from roadvision_v2.pipeline import RoadVisionV2Pipeline

from .artifacts import build_artifact_manifest
from .configuration import APIConfig
from .errors import sanitize_error_message
from .gpu_lock import SingleProcessWorkerGuard, gpu_inference_lock
from .job_manager import JOB_STATUS_CANCELLED, JobManager

logger = logging.getLogger("roadvision_api_v2.worker")


def _parse_iso_seconds_diff(start_iso: Optional[str], end_iso: Optional[str]) -> Optional[float]:
    if not start_iso or not end_iso:
        return None
    try:
        t0 = datetime.fromisoformat(start_iso)
        t1 = datetime.fromisoformat(end_iso)
        return round(max(0.0, (t1 - t0).total_seconds()), 4)
    except Exception:
        return None


class BoundedGPUWorker:
    """Controlled sequential GPU worker processing one inspection job at a time."""

    def __init__(
        self,
        job_manager: JobManager,
        api_cfg: APIConfig,
        pipeline_factory: Optional[Callable[[RoadVisionV2Config, str, Optional[str]], Any]] = None,
        device: Optional[str] = None,
        poll_interval_sec: float = 0.25,
        geospatial_store: Optional[Any] = None,
    ) -> None:
        self.job_manager = job_manager
        self.api_cfg = api_cfg
        self.geospatial_store = geospatial_store
        self._pipeline_factory = pipeline_factory or (
            lambda cfg, sid, dev: RoadVisionV2Pipeline(cfg=cfg, session_id=sid, device=dev)
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.poll_interval_sec = float(poll_interval_sec)

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active_job_id: Optional[str] = None
        self._recovery_summary: Optional[Dict[str, int]] = None

        lock_path = os.path.join(
            os.path.dirname(api_cfg.resolved_database_path),
            ".roadvision_v2_worker.lock",
        )
        self._process_guard = SingleProcessWorkerGuard(lock_path)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active_job_id(self) -> Optional[str]:
        return self._active_job_id

    @property
    def recovery_summary(self) -> Optional[Dict[str, int]]:
        return self._recovery_summary

    def start(self, enforce_process_lock: bool = True) -> bool:
        """Recover interrupted jobs and start the bounded worker thread idempotently."""
        with self._lifecycle_lock:
            if self.is_running:
                return True
            if enforce_process_lock and not self._process_guard.acquire():
                logger.warning("Another V2 worker process already holds the worker lock; refusing duplicate start.")
                return False
            self._recovery_summary = self.job_manager.recover_interrupted_jobs()
            self._stop_event.clear()
            self._wake_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="roadvision-v2-gpu-worker",
                daemon=False,
            )
            self._thread.start()
            logger.info(
                "BoundedGPUWorker started | device=%s recovery=%s",
                self.device,
                self._recovery_summary,
            )
            return True

    def notify(self) -> None:
        """Wake worker immediately when a new job is enqueued."""
        self._wake_event.set()

    def stop(self, timeout: Optional[float] = None) -> bool:
        """Signal cooperative shutdown and wait cleanly for any active job boundary."""
        with self._lifecycle_lock:
            t_wait = timeout if timeout is not None else self.api_cfg.worker_shutdown_timeout
            self._stop_event.set()
            self._wake_event.set()
            th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=t_wait)
        stopped = th is None or not th.is_alive()
        if stopped:
            with self._lifecycle_lock:
                self._thread = None
                self._process_guard.release()
        return stopped

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            job = None
            try:
                job = self.job_manager.claim_next_job()
            except Exception as exc:
                logger.error("Error claiming next job: %s", exc)
                self._wake_event.wait(timeout=self.poll_interval_sec)
                self._wake_event.clear()
                continue

            if job is None:
                self._wake_event.wait(timeout=self.poll_interval_sec)
                self._wake_event.clear()
                continue

            self._active_job_id = job["job_id"]
            try:
                self._execute_job(job)
            finally:
                self._active_job_id = None

    def _build_v2_config_from_job(self, job: Dict[str, Any]) -> Tuple[RoadVisionV2Config, Dict[str, Any]]:
        snap = job.get("config_snapshot") or {}
        depth_snap = snap.get("depth") or {}
        exec_opts = snap.get("execution_options") or {}
        enable_depth = depth_snap.get("enabled")
        diagnostics = bool(exec_opts.get("diagnostics", False))
        enable_geometry = bool(exec_opts.get("enable_geometry", False))
        overrides = None
        if diagnostics or enable_geometry:
            overrides = {}
            if diagnostics:
                overrides["diagnostics"] = {
                    "enable_horizon_proxy": True,
                    "show_rejection_reasons": True,
                    "visualize_annular": True,
                }
            if enable_geometry:
                overrides["geometry_snapshot"] = {"enabled": True}
        cfg = load_v2_config(
            mode=job["mode"],
            enable_depth=enable_depth,
            overrides=overrides,
            check_assets=True,
        )
        return cfg, exec_opts

    def _execute_job(self, job: Dict[str, Any]) -> None:
        job_id = job["job_id"]
        input_path = job["validated_input_path"]
        output_dir = job["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        queue_wait_sec = _parse_iso_seconds_diff(job.get("created_at"), job.get("started_at"))

        pipeline = None
        t_job_start = time.perf_counter()
        try:
            with gpu_inference_lock():
                if self.job_manager.is_cancellation_requested(job_id):
                    self.job_manager.finalize_cancelled_job(job_id)
                    return

                cfg, exec_opts = self._build_v2_config_from_job(job)
                t_init0 = time.perf_counter()
                pipeline = self._pipeline_factory(cfg, job_id, self.device)
                model_init_sec = round(time.perf_counter() - t_init0, 4)

                if self.job_manager.is_cancellation_requested(job_id):
                    self.job_manager.finalize_cancelled_job(
                        job_id,
                        telemetry={"queue_wait_sec": queue_wait_sec, "model_init_sec": model_init_sec},
                    )
                    return

                latest_warning_holder: Dict[str, Optional[str]] = {"msg": None}
                last_progress_snapshot: Dict[str, Any] = {}

                def _on_init(meta: Dict[str, Any]) -> None:
                    self.job_manager.mark_running(
                        job_id,
                        total_frames=meta.get("total_frames"),
                        latest_warning=latest_warning_holder["msg"],
                    )

                def _on_warning(msg: str) -> None:
                    latest_warning_holder["msg"] = str(msg)[:300]
                    try:
                        self.job_manager.record_warning(job_id, latest_warning_holder["msg"])
                    except Exception:
                        pass

                def _should_cancel() -> bool:
                    return self._stop_event.is_set() or self.job_manager.is_cancellation_requested(job_id)

                def _on_progress(prog: Dict[str, Any]) -> None:
                    last_progress_snapshot.clear()
                    last_progress_snapshot.update(prog)
                    self.job_manager.update_progress(
                        job_id,
                        processed_frames=int(prog.get("processed_frames", 0)),
                        total_frames=prog.get("total_frames"),
                        detection_observation_count=int(prog.get("detection_observation_count", 0)),
                        unique_candidate_track_count=int(prog.get("unique_candidate_track_count", 0)),
                        depth_invocation_count=int(prog.get("depth_invocation_count", 0)),
                        scheduler_request_count=int(prog.get("scheduler_request_count", 0)),
                        failed_depth_count=int(prog.get("failed_depth_count", 0)),
                        measurement_state_counts=prog.get("measurement_state_counts"),
                        throughput_fps=prog.get("throughput_fps"),
                        latest_warning=latest_warning_holder["msg"],
                        force=False,
                    )

                summary = pipeline.process_video(
                    source=input_path,
                    output_dir=output_dir,
                    save_video=bool(exec_opts.get("save_video", True)),
                    save_json=bool(exec_opts.get("save_json", True)),
                    save_csv=bool(exec_opts.get("save_csv", True)),
                    diagnostics=bool(exec_opts.get("diagnostics", False)),
                    max_frames=exec_opts.get("max_frames"),
                    stem_override=stem,
                    on_init=_on_init,
                    on_progress=_on_progress,
                    should_cancel=_should_cancel,
                    on_warning=_on_warning,
                    transcode_h264=bool(self.api_cfg.enable_h264_transcode),
                    write_empty_frames_json=True,
                )

                total_job_sec = round(time.perf_counter() - t_job_start, 4)
                telemetry = {
                    "queue_wait_sec": queue_wait_sec,
                    "model_init_sec": summary.get("model_init_sec", model_init_sec),
                    "video_processing_sec": summary.get("elapsed_sec"),
                    "artifact_generation_sec": summary.get("artifact_generation_sec"),
                    "total_worker_execution_sec": total_job_sec,
                    "processed_frames": summary.get("total_frames", 0),
                    "sustained_processing_fps": summary.get("end_to_end_fps"),
                    "actual_depth_executions": summary.get("depth_calls", 0),
                    "scheduler_depth_requests": summary.get("scheduler_depth_requests", 0),
                    "failed_depth_inferences": summary.get("failed_depth_inferences", 0),
                    "peak_allocated_gpu_mb": summary.get("peak_allocated_gpu_mb"),
                    "peak_reserved_gpu_mb": summary.get("peak_reserved_gpu_mb"),
                }

                was_cancelled = bool(summary.get("cancelled", False)) or self.job_manager.is_cancellation_requested(job_id)
                target_manifest_status = JOB_STATUS_CANCELLED if was_cancelled else "COMPLETED"
                manifest, availability = build_artifact_manifest(
                    job_id=job_id,
                    mode=job["mode"],
                    stem=stem,
                    output_dir=output_dir,
                    job_status=target_manifest_status,
                    video_codec=summary.get("video_codec"),
                    video_browser_compatible=summary.get("video_browser_compatible"),
                )

                if was_cancelled:
                    self.job_manager.finalize_cancelled_job(
                        job_id,
                        processed_frames=int(summary.get("total_frames", 0)),
                        total_frames=job.get("total_frames"),
                        detection_observation_count=int(summary.get("total_hazard_observations", 0)),
                        unique_candidate_track_count=int(summary.get("unique_tracks", 0)),
                        depth_invocation_count=int(summary.get("depth_calls", 0)),
                        scheduler_request_count=int(summary.get("scheduler_depth_requests", 0)),
                        failed_depth_count=int(summary.get("failed_depth_inferences", 0)),
                        measurement_state_counts=summary.get("measurement_state_counts"),
                        throughput_fps=summary.get("end_to_end_fps"),
                        artifact_manifest=manifest,
                        artifact_availability=availability,
                        summary=summary,
                        telemetry=telemetry,
                    )
                else:
                    completed_job = self.job_manager.complete_job(
                        job_id,
                        processed_frames=int(summary.get("total_frames", 0)),
                        total_frames=int(summary.get("total_frames", 0)) or job.get("total_frames"),
                        detection_observation_count=int(summary.get("total_hazard_observations", 0)),
                        unique_candidate_track_count=int(summary.get("unique_tracks", 0)),
                        depth_invocation_count=int(summary.get("depth_calls", 0)),
                        scheduler_request_count=int(summary.get("scheduler_depth_requests", 0)),
                        failed_depth_count=int(summary.get("failed_depth_inferences", 0)),
                        measurement_state_counts=summary.get("measurement_state_counts") or {},
                        throughput_fps=summary.get("end_to_end_fps"),
                        artifact_manifest=manifest,
                        artifact_availability=availability,
                        summary=summary,
                        telemetry=telemetry,
                        latest_warning=latest_warning_holder["msg"],
                    )
                    if self.geospatial_store is not None:
                        try:
                            full_completed = self.job_manager.get_job(job_id, include_internal_paths=True)
                            self.geospatial_store.sync_job_tracks_with_gps(full_completed)
                        except Exception as geo_exc:
                            logger.warning("Post-completion GPS sync failed for %s: %s", job_id, geo_exc)
        except Exception as exc:
            err_str = str(exc).lower()
            is_oom = isinstance(exc, RuntimeError) and "out of memory" in err_str
            err_code = "GPU_OUT_OF_MEMORY" if is_oom else "PROCESSING_FAILED"
            safe_msg = (
                "GPU out of memory during V2 inspection job execution."
                if is_oom
                else sanitize_error_message(exc)
            )
            logger.exception("V2 worker job %s failed (%s)", job_id, err_code)
            manifest, availability = build_artifact_manifest(
                job_id=job_id,
                mode=job["mode"],
                stem=stem,
                output_dir=output_dir,
                job_status="FAILED",
            )
            try:
                self.job_manager.fail_job(
                    job_id,
                    error_code=err_code,
                    error_message=safe_msg,
                    telemetry={"queue_wait_sec": queue_wait_sec, "total_worker_execution_sec": round(time.perf_counter() - t_job_start, 4)},
                    artifact_manifest=manifest,
                    artifact_availability=availability,
                )
            except Exception as fail_exc:
                logger.error("Failed to record job failure for %s: %s", job_id, fail_exc)
        finally:
            # Drop model references before releasing cached CUDA allocator blocks.
            # Note: torch.cuda.empty_cache() returns unreferenced cached blocks to the CUDA
            # driver/allocator; it does NOT release live tensor/model references elsewhere.
            if pipeline is not None:
                try:
                    if hasattr(pipeline, "_depth_model"):
                        pipeline._depth_model = None
                    if hasattr(pipeline, "_yolo"):
                        pipeline._yolo = None
                except Exception:
                    pass
                del pipeline
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
