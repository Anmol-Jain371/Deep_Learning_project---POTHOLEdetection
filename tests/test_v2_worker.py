"""Automated tests for BoundedGPUWorker concurrency, cancellation, isolation, and OOM (Phase 8B.2/8B.3)."""

import os
import sys
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_api_v2.configuration import APIConfig
from roadvision_api_v2.gpu_lock import gpu_inference_lock
from roadvision_api_v2.job_manager import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JobManager,
)
from roadvision_api_v2.storage import JobDatabase
from roadvision_api_v2.worker import BoundedGPUWorker


def _write_dummy_mp4(path: str, n_frames: int = 12) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    for _ in range(n_frames):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()


class _FakePipeline:
    instances_created = []
    active_count = 0
    max_observed_active = 0
    _lock = threading.Lock()

    def __init__(self, cfg, session_id, device, *, sleep_per_frame=0.01, mode_behavior="ok"):
        self.cfg = cfg
        self.session_id = session_id
        self.device = device
        self.sleep_per_frame = sleep_per_frame
        self.mode_behavior = mode_behavior
        self._depth_model = object()
        self._yolo = object()
        self.local_track_counter = 0
        _FakePipeline.instances_created.append(session_id)

    def process_video(
        self,
        source,
        output_dir,
        save_video=True,
        save_json=True,
        save_csv=True,
        diagnostics=False,
        max_frames=None,
        stem_override=None,
        on_init=None,
        on_progress=None,
        should_cancel=None,
        on_warning=None,
        transcode_h264=False,
        write_empty_frames_json=True,
    ):
        with _FakePipeline._lock:
            _FakePipeline.active_count += 1
            _FakePipeline.max_observed_active = max(
                _FakePipeline.max_observed_active, _FakePipeline.active_count
            )
        try:
            if self.mode_behavior == "oom":
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            if self.mode_behavior == "crash":
                raise ValueError("Unexpected internal tensor mismatch")

            os.makedirs(output_dir, exist_ok=True)
            stem = stem_override or "clip"
            total = max_frames or 10
            if on_init:
                on_init({"total_frames": total, "source_fps": 25.0, "width": 160, "height": 120})

            processed = 0
            cancelled = False
            for idx in range(total):
                if should_cancel and should_cancel():
                    cancelled = True
                    break
                time.sleep(self.sleep_per_frame)
                processed += 1
                self.local_track_counter += 1
                if on_progress:
                    on_progress({
                        "processed_frames": processed,
                        "total_frames": total,
                        "detection_observation_count": processed,
                        "unique_candidate_track_count": 1,
                        "depth_invocation_count": 1 if self.cfg.depth.enabled else 0,
                        "scheduler_request_count": 1 if self.cfg.depth.enabled else 0,
                        "failed_depth_count": 0,
                        "measurement_state_counts": {
                            "FRESH_VALID": 0, "FRESH_INVALID": 0, "CACHED_VALID": 0,
                            "EXPIRED": 0, "NOT_EVALUATED": processed,
                        },
                        "throughput_fps": 100.0,
                    })

            import json
            prefix = f"phase8a_{self.cfg.mode}_{stem}"
            summary = {
                "session_id": self.session_id,
                "mode": self.cfg.mode,
                "total_frames": processed,
                "end_to_end_fps": 100.0,
                "depth_enabled": self.cfg.depth.enabled,
                "depth_calls": 1 if self.cfg.depth.enabled else 0,
                "scheduler_depth_requests": 1 if self.cfg.depth.enabled else 0,
                "failed_depth_inferences": 0,
                "unique_tracks": 1,
                "total_hazard_observations": processed,
                "measurement_state_counts": {
                    "FRESH_VALID": 0, "FRESH_INVALID": 0, "CACHED_VALID": 0,
                    "EXPIRED": 0, "NOT_EVALUATED": processed,
                },
                "cancelled": cancelled,
            }
            if save_json:
                with open(os.path.join(output_dir, f"{prefix}_frames.json"), "w", encoding="utf-8") as fh:
                    json.dump([], fh)
                with open(os.path.join(output_dir, f"{prefix}_tracks.json"), "w", encoding="utf-8") as fh:
                    json.dump({}, fh)
            with open(os.path.join(output_dir, f"{prefix}_session_summary.json"), "w", encoding="utf-8") as fh:
                json.dump(summary, fh)
            return summary
        finally:
            with _FakePipeline._lock:
                _FakePipeline.active_count -= 1


class TestV2Worker(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = APIConfig(
            enabled=True,
            database_path=os.path.join(self._tmp.name, "jobs.db"),
            upload_directory=os.path.join(self._tmp.name, "uploads"),
            artifact_directory=os.path.join(self._tmp.name, "artifacts"),
            progress_update_interval=0.05,
            worker_shutdown_timeout=5.0,
            project_root=self._tmp.name,
        )
        self.cfg.ensure_runtime_dirs()
        self.db = JobDatabase(self.cfg.resolved_database_path)
        self.jm = JobManager(self.db, max_queued_jobs=5, progress_update_interval=0.05)
        _FakePipeline.instances_created = []
        _FakePipeline.active_count = 0
        _FakePipeline.max_observed_active = 0

    def tearDown(self):
        self._tmp.cleanup()

    def _submit(self, name="vid.mp4", mode="live", max_frames=8):
        in_path = os.path.join(self.cfg.resolved_upload_dir, name)
        _write_dummy_mp4(in_path, n_frames=max_frames)
        out_dir = os.path.join(self.cfg.resolved_artifact_dir, name)
        return self.jm.create_job(
            original_filename=name,
            validated_input_path=in_path,
            mode=mode,
            config_snapshot={
                "mode": mode,
                "depth": {"enabled": mode != "live"},
                "execution_options": {"save_video": False, "save_json": True, "save_csv": False, "max_frames": max_frames},
            },
            output_dir=out_dir,
            total_frames=max_frames,
        )

    def _wait_terminal(self, job_id, timeout=8.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.jm.get_job(job_id)
            if j["status"] in (JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED):
                return j
            time.sleep(0.05)
        self.fail(f"Job {job_id} did not reach terminal state within {timeout}s")

    def test_single_concurrency_and_sequential_isolation(self):
        worker = BoundedGPUWorker(
            self.jm,
            self.cfg,
            pipeline_factory=lambda cfg, sid, dev: _FakePipeline(cfg, sid, dev, sleep_per_frame=0.015),
            device="cpu",
            poll_interval_sec=0.05,
        )
        j1 = self._submit("a.mp4", mode="live", max_frames=6)
        j2 = self._submit("b.mp4", mode="live", max_frames=6)

        self.assertTrue(worker.start(enforce_process_lock=True))
        # Second worker on same DB lock must refuse duplicate start
        worker2 = BoundedGPUWorker(self.jm, self.cfg, device="cpu")
        self.assertFalse(worker2.start(enforce_process_lock=True))

        try:
            r1 = self._wait_terminal(j1["job_id"])
            r2 = self._wait_terminal(j2["job_id"])
            self.assertEqual(r1["status"], JOB_STATUS_COMPLETED)
            self.assertEqual(r2["status"], JOB_STATUS_COMPLETED)
            self.assertEqual(_FakePipeline.max_observed_active, 1)
            self.assertEqual(_FakePipeline.instances_created, [j1["job_id"], j2["job_id"]])
        finally:
            self.assertTrue(worker.stop())

    def test_running_and_init_cancellation(self):
        init_entered = threading.Event()
        release_init = threading.Event()

        def _slow_init_factory(cfg, sid, dev):
            init_entered.set()
            release_init.wait(timeout=3.0)
            return _FakePipeline(cfg, sid, dev, sleep_per_frame=0.03)

        worker = BoundedGPUWorker(
            self.jm, self.cfg, pipeline_factory=_slow_init_factory, device="cpu", poll_interval_sec=0.05
        )
        j_init = self._submit("init_cancel.mp4", mode="live", max_frames=20)
        worker.start(enforce_process_lock=False)
        try:
            self.assertTrue(init_entered.wait(timeout=3.0))
            self.jm.request_cancellation(j_init["job_id"])
            release_init.set()
            res = self._wait_terminal(j_init["job_id"])
            self.assertEqual(res["status"], JOB_STATUS_CANCELLED)
        finally:
            worker.stop()

        # Now test running cancellation mid-frame
        worker_run = BoundedGPUWorker(
            self.jm,
            self.cfg,
            pipeline_factory=lambda cfg, sid, dev: _FakePipeline(cfg, sid, dev, sleep_per_frame=0.05),
            device="cpu",
            poll_interval_sec=0.05,
        )
        j_run = self._submit("run_cancel.mp4", mode="live", max_frames=40)
        worker_run.start(enforce_process_lock=False)
        try:
            time.sleep(0.15)
            self.jm.request_cancellation(j_run["job_id"])
            res_run = self._wait_terminal(j_run["job_id"])
            self.assertEqual(res_run["status"], JOB_STATUS_CANCELLED)
            self.assertLess(res_run["processed_frames"], 40)
        finally:
            worker_run.stop()

    def test_oom_isolation_and_recovery_to_next_job(self):
        behaviors = iter(["oom", "ok"])

        def _factory(cfg, sid, dev):
            return _FakePipeline(cfg, sid, dev, sleep_per_frame=0.01, mode_behavior=next(behaviors))

        worker = BoundedGPUWorker(self.jm, self.cfg, pipeline_factory=_factory, device="cpu", poll_interval_sec=0.05)
        j_oom = self._submit("oom.mp4", mode="live", max_frames=4)
        j_ok = self._submit("ok.mp4", mode="live", max_frames=4)
        worker.start(enforce_process_lock=False)
        try:
            r_oom = self._wait_terminal(j_oom["job_id"])
            r_ok = self._wait_terminal(j_ok["job_id"])
            self.assertEqual(r_oom["status"], JOB_STATUS_FAILED)
            self.assertEqual(r_oom["error_code"], "GPU_OUT_OF_MEMORY")
            self.assertEqual(r_ok["status"], JOB_STATUS_COMPLETED)
        finally:
            worker.stop()

    def test_completion_vs_cancellation_race_atomic(self):
        j = self._submit("race.mp4", mode="live", max_frames=5)
        self.jm.claim_next_job()
        self.jm.mark_running(j["job_id"], total_frames=5)
        self.jm.request_cancellation(j["job_id"])
        final = self.jm.complete_job(
            j["job_id"],
            processed_frames=5,
            total_frames=5,
            detection_observation_count=0,
            unique_candidate_track_count=0,
            depth_invocation_count=0,
            scheduler_request_count=0,
            failed_depth_count=0,
            measurement_state_counts={},
            throughput_fps=10.0,
            artifact_manifest={"job_id": j["job_id"], "partial": True, "artifacts": []},
            artifact_availability={},
            summary={},
        )
        self.assertEqual(final["status"], JOB_STATUS_CANCELLED)

    def test_missing_model_and_resource_cleanup(self):
        created_refs = []

        def _missing_factory(cfg, sid, dev):
            raise FileNotFoundError("Missing checkpoint runs/detect/missing.pt")

        worker = BoundedGPUWorker(self.jm, self.cfg, pipeline_factory=_missing_factory, device="cpu", poll_interval_sec=0.05)
        j_miss = self._submit("missing_ckpt.mp4", mode="live", max_frames=3)
        worker.start(enforce_process_lock=False)
        try:
            r_miss = self._wait_terminal(j_miss["job_id"])
            self.assertEqual(r_miss["status"], JOB_STATUS_FAILED)
            self.assertEqual(r_miss["error_code"], "PROCESSING_FAILED")
        finally:
            worker.stop()

        def _track_ref_factory(cfg, sid, dev):
            p = _FakePipeline(cfg, sid, dev, sleep_per_frame=0.005)
            created_refs.append(p)
            return p

        worker2 = BoundedGPUWorker(self.jm, self.cfg, pipeline_factory=_track_ref_factory, device="cpu", poll_interval_sec=0.05)
        j_ok = self._submit("cleanup.mp4", mode="live", max_frames=3)
        worker2.start(enforce_process_lock=False)
        try:
            r_ok = self._wait_terminal(j_ok["job_id"])
            self.assertEqual(r_ok["status"], JOB_STATUS_COMPLETED)
            self.assertIsNone(created_refs[0]._depth_model)
            self.assertIsNone(created_refs[0]._yolo)
        finally:
            worker2.stop()

    def test_legacy_gpu_lock_coordinates_with_v2_worker(self):
        # Verify gpu_inference_lock blocks V2 worker while legacy holds GPU
        started = threading.Event()

        def _factory(cfg, sid, dev):
            started.set()
            return _FakePipeline(cfg, sid, dev, sleep_per_frame=0.01)

        worker = BoundedGPUWorker(self.jm, self.cfg, pipeline_factory=_factory, device="cpu", poll_interval_sec=0.05)
        j = self._submit("locked.mp4", mode="live", max_frames=4)
        with gpu_inference_lock():
            worker.start(enforce_process_lock=False)
            time.sleep(0.25)
            self.assertFalse(started.is_set(), "Worker must wait for gpu_inference_lock")
        try:
            res = self._wait_terminal(j["job_id"])
            self.assertEqual(res["status"], JOB_STATUS_COMPLETED)
            self.assertTrue(started.is_set())
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
