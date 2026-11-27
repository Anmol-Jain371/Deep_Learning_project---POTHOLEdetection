"""Security and boundary tests for RoadVision AI 2.0 API (Phase 8B.5)."""

import io
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from flask import Flask

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_api_v2 import register_v2_api
from roadvision_api_v2.configuration import APIConfig
from roadvision_api_v2.job_manager import JOB_STATUS_COMPLETED, JobManager


def _make_mp4_bytes() -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        path = tf.name
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
        for _ in range(3):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


class TestV2APISecurity(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = APIConfig(
            enabled=True,
            host="127.0.0.1",
            database_path=os.path.join(self._tmp.name, "jobs.db"),
            upload_directory=os.path.join(self._tmp.name, "uploads"),
            artifact_directory=os.path.join(self._tmp.name, "artifacts"),
            allowed_cors_origins=["http://localhost:5173"],
            max_queued_jobs=2,
            project_root=self._tmp.name,
        )
        self.cfg.ensure_runtime_dirs()
        self.app = Flask(__name__)
        self.rt = register_v2_api(self.app, self.cfg)
        self.client = self.app.test_client()
        self.jm: JobManager = self.rt["job_manager"]

    def tearDown(self):
        self._tmp.cleanup()

    def _create_completed_job_with_artifact(self):
        out_dir = os.path.join(self.cfg.resolved_artifact_dir, "jobA")
        os.makedirs(out_dir, exist_ok=True)
        secret_file = os.path.join(self._tmp.name, "secret.txt")
        with open(secret_file, "w", encoding="utf-8") as fh:
            fh.write("TOP_SECRET")

        j = self.jm.create_job(
            original_filename="clip.mp4",
            validated_input_path=os.path.join(self.cfg.resolved_upload_dir, "clip.mp4"),
            mode="live",
            config_snapshot={"mode": "live"},
            output_dir=out_dir,
            total_frames=3,
        )
        jid = j["job_id"]
        real_out = os.path.join(self.cfg.resolved_artifact_dir, jid)
        os.makedirs(real_out, exist_ok=True)
        # Update output_dir inside DB cleanly via complete_job after placing file
        summary_file = os.path.join(out_dir, f"phase8a_live_{jid}_session_summary.json")
        with open(summary_file, "w", encoding="utf-8") as fh:
            fh.write('{"ok": true}')

        self.jm.claim_next_job()
        self.jm.mark_running(jid, total_frames=3)
        self.jm.complete_job(
            jid,
            processed_frames=3,
            total_frames=3,
            detection_observation_count=0,
            unique_candidate_track_count=0,
            depth_invocation_count=0,
            scheduler_request_count=0,
            failed_depth_count=0,
            measurement_state_counts={},
            throughput_fps=10.0,
            artifact_manifest={
                "job_id": jid,
                "partial": False,
                "artifacts": [{
                    "artifact_id": "session_summary_json",
                    "filename": os.path.basename(summary_file),
                    "artifact_type": "session_summary_json",
                    "mime_type": "application/json",
                    "size_bytes": os.path.getsize(summary_file),
                    "created_at": "2026-11-27T00:00:00+00:00",
                    "download_url": f"/api/v2/jobs/{jid}/artifacts/session_summary_json",
                    "available": True,
                    "partial": False,
                }],
            },
            artifact_availability={"session_summary_json": True},
            summary={"ok": True},
        )
        return jid, out_dir, secret_file

    def test_path_traversal_and_arbitrary_file_access_blocked(self):
        jid, out_dir, secret_file = self._create_completed_job_with_artifact()
        for bad_name in ("..%2Fsecret.txt", "../secret.txt", "..\\server.py", "pothole_best.pt", "secret.txt"):
            r = self.client.get(f"/api/v2/jobs/{jid}/artifacts/{bad_name}")
            self.assertIn(r.status_code, (400, 404))

        # Even if a manifest entry pointed to a symlink, resolve_artifact_entry rejects symlinks
        link_path = os.path.join(out_dir, "sym_summary.json")
        try:
            os.symlink(secret_file, link_path)
            symlink_supported = True
        except (OSError, NotImplementedError):
            symlink_supported = False

        if symlink_supported:
            from roadvision_api_v2.artifacts import resolve_artifact_entry
            from roadvision_api_v2.errors import NotFoundError
            fake_job = {
                "job_id": jid,
                "output_dir": out_dir,
                "artifact_manifest": {
                    "artifacts": [{
                        "artifact_id": "session_summary_json",
                        "filename": "sym_summary.json",
                        "available": True,
                    }]
                },
            }
            with self.assertRaises(NotFoundError):
                resolve_artifact_entry(fake_job, "session_summary_json", self.cfg.resolved_artifact_dir)

    def test_cross_job_artifact_isolation(self):
        jid_a, _, _ = self._create_completed_job_with_artifact()
        j_b = self.jm.create_job(
            original_filename="b.mp4",
            validated_input_path=os.path.join(self.cfg.resolved_upload_dir, "b.mp4"),
            mode="live",
            config_snapshot={"mode": "live"},
            output_dir=os.path.join(self.cfg.resolved_artifact_dir, "jobB"),
            total_frames=2,
        )
        # Job B is QUEUED -> cannot download Job A's artifact via Job B
        r = self.client.get(f"/api/v2/jobs/{j_b['job_id']}/artifacts/session_summary_json")
        self.assertEqual(r.status_code, 409)

    def test_cors_origin_restricted(self):
        r_ok = self.client.get("/api/v2/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(r_ok.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")

        r_evil = self.client.get("/api/v2/health", headers={"Origin": "https://evil.example.com"})
        self.assertIsNone(r_evil.headers.get("Access-Control-Allow-Origin"))

    def test_error_responses_do_not_expose_paths_or_tracebacks(self):
        r = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(_make_mp4_bytes()), "a.mp4"), "mode": "live", "weights": "C:\\Windows\\secret.pt"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 400)
        body = r.get_data(as_text=True)
        self.assertNotIn("Traceback", body)
        self.assertNotIn(self._tmp.name, body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
