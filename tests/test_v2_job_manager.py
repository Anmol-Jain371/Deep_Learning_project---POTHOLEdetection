"""Automated tests for SQLite JobDatabase and JobManager lifecycle (Phase 8B.2)."""

import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_api_v2.errors import InvalidJobStateError, NotFoundError, QueueFullError
from roadvision_api_v2.job_manager import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_CANCELLING,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_INITIALIZING,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JobManager,
)
from roadvision_api_v2.storage import JobDatabase


class TestV2JobManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "test_jobs.db")
        self.db = JobDatabase(self.db_path)
        self.jm = JobManager(self.db, max_queued_jobs=5, progress_update_interval=0.05)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_job(self, name="clip.mp4", mode="balanced", total_frames=100):
        out_dir = os.path.join(self._tmp.name, "out", name)
        in_path = os.path.join(self._tmp.name, "in", name)
        return self.jm.create_job(
            original_filename=name,
            validated_input_path=in_path,
            mode=mode,
            config_snapshot={"mode": mode, "depth": {"enabled": mode != "live"}},
            output_dir=out_dir,
            total_frames=total_frames,
        )

    def test_database_wal_and_indexes_created(self):
        self.assertTrue(os.path.exists(self.db_path))
        self.assertEqual(self.db.verify_wal_mode(), "wal")
        with self.db.read_connection() as conn:
            idxs = [r["name"] for r in conn.execute("PRAGMA index_list('inspection_jobs')").fetchall()]
        self.assertIn("idx_jobs_status", idxs)
        self.assertIn("idx_jobs_created_at", idxs)

    def test_job_persistence_across_instances(self):
        j = self._make_job("road.mp4", "survey", 50)
        jm2 = JobManager(JobDatabase(self.db_path))
        loaded = jm2.get_job(j["job_id"])
        self.assertEqual(loaded["original_filename"], "road.mp4")
        self.assertEqual(loaded["mode"], "survey")
        self.assertEqual(loaded["status"], JOB_STATUS_QUEUED)
        self.assertEqual(loaded["total_frames"], 50)

    def test_atomic_claim_and_duplicate_execution_prevention(self):
        j1 = self._make_job("first.mp4")
        j2 = self._make_job("second.mp4")

        claimed = []
        barrier = threading.Barrier(4)

        def _claimer():
            barrier.wait()
            c = self.jm.claim_next_job()
            if c is not None:
                claimed.append(c["job_id"])

        threads = [threading.Thread(target=_claimer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(claimed, [j1["job_id"]])
        # Cannot claim j2 while j1 is INITIALIZING
        self.assertIsNone(self.jm.claim_next_job())
        self.assertEqual(self.jm.get_job(j2["job_id"])["status"], JOB_STATUS_QUEUED)

    def test_valid_and_invalid_transitions_and_terminal_protection(self):
        j = self._make_job("t.mp4")
        jid = j["job_id"]
        with self.assertRaises(InvalidJobStateError):
            self.jm.mark_running(jid)  # Still QUEUED

        self.jm.claim_next_job()
        self.jm.mark_running(jid, total_frames=40)
        self.jm.complete_job(
            jid,
            processed_frames=40,
            total_frames=40,
            detection_observation_count=5,
            unique_candidate_track_count=2,
            depth_invocation_count=4,
            scheduler_request_count=4,
            failed_depth_count=0,
            measurement_state_counts={"FRESH_VALID": 4, "CACHED_VALID": 1, "FRESH_INVALID": 0, "EXPIRED": 0, "NOT_EVALUATED": 0},
            throughput_fps=20.0,
            artifact_manifest={"job_id": jid, "partial": False, "artifacts": []},
            artifact_availability={},
            summary={"total_frames": 40},
        )
        self.assertEqual(self.jm.get_job(jid)["status"], JOB_STATUS_COMPLETED)

        # Terminal state protections
        with self.assertRaises(InvalidJobStateError):
            self.jm.request_cancellation(jid)
        with self.assertRaises(InvalidJobStateError):
            self.jm.mark_running(jid)
        res_fail = self.jm.fail_job(jid, error_code="ERR", error_message="Should not overwrite")
        self.assertEqual(res_fail["status"], JOB_STATUS_COMPLETED)

    def test_monotonic_progress_and_unknown_total_frames(self):
        j = self._make_job("nofps.mp4", total_frames=None)
        jid = j["job_id"]
        self.jm.claim_next_job()
        self.jm.mark_running(jid, total_frames=None)
        self.jm.update_progress(jid, processed_frames=10, total_frames=None, force=True)
        d1 = self.jm.get_job(jid)
        self.assertEqual(d1["processed_frames"], 10)
        self.assertIsNone(d1["progress_percentage"])

        # Out-of-order lower frame count must not regress progress
        self.jm.update_progress(jid, processed_frames=6, total_frames=None, force=True)
        d2 = self.jm.get_job(jid)
        self.assertEqual(d2["processed_frames"], 10)

    def test_queue_saturation_and_concurrent_submissions(self):
        errors = []
        created = []
        barrier = threading.Barrier(8)

        def _submit(i):
            barrier.wait()
            try:
                j = self._make_job(f"v{i}.mp4")
                created.append(j["job_id"])
            except QueueFullError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_submit, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(created), 5)
        self.assertEqual(len(errors), 3)

    def test_restart_recovery_policy(self):
        # 1 completed, 1 cancelled, 1 running (interrupted), 1 initializing (interrupted), 1 queued
        j_done = self._make_job("done.mp4")
        self.jm.claim_next_job()
        self.jm.mark_running(j_done["job_id"], total_frames=5)
        self.jm.complete_job(
            j_done["job_id"],
            processed_frames=5,
            total_frames=5,
            detection_observation_count=0,
            unique_candidate_track_count=0,
            depth_invocation_count=0,
            scheduler_request_count=0,
            failed_depth_count=0,
            measurement_state_counts={},
            throughput_fps=10.0,
            artifact_manifest={"job_id": j_done["job_id"], "partial": False, "artifacts": []},
            artifact_availability={},
            summary={},
        )

        j_canc = self._make_job("canc.mp4")
        self.jm.request_cancellation(j_canc["job_id"])

        j_run = self._make_job("run.mp4")
        self.jm.claim_next_job()
        self.jm.mark_running(j_run["job_id"], total_frames=20)

        j_q = self._make_job("queued.mp4")

        # Simulate direct DB state for another interrupted INITIALIZING job
        rec = self.jm.recover_interrupted_jobs()
        self.assertEqual(rec["interrupted_marked_failed"], 1)
        self.assertEqual(rec["queued_preserved_for_dispatch"], 1)

        self.assertEqual(self.jm.get_job(j_done["job_id"])["status"], JOB_STATUS_COMPLETED)
        self.assertEqual(self.jm.get_job(j_canc["job_id"])["status"], JOB_STATUS_CANCELLED)
        run_after = self.jm.get_job(j_run["job_id"])
        self.assertEqual(run_after["status"], JOB_STATUS_FAILED)
        self.assertEqual(run_after["error_code"], "WORKER_INTERRUPTED")
        self.assertEqual(self.jm.get_job(j_q["job_id"])["status"], JOB_STATUS_QUEUED)

    def test_unknown_job_raises_not_found(self):
        with self.assertRaises(NotFoundError):
            self.jm.get_job("11111111-1111-4111-8111-111111111111")


if __name__ == "__main__":
    unittest.main(verbosity=2)
