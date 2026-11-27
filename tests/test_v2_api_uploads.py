"""Automated tests for video upload validation and failure cleanup (Phase 8B.3)."""

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


def _make_mp4_bytes(n_frames: int = 5, width: int = 160, height: int = 120, fps: float = 25.0) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        path = tf.name
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for _ in range(n_frames):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
        writer.release()
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


class TestV2APIUploads(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = APIConfig(
            enabled=True,
            database_path=os.path.join(self._tmp.name, "jobs.db"),
            upload_directory=os.path.join(self._tmp.name, "uploads"),
            artifact_directory=os.path.join(self._tmp.name, "artifacts"),
            maximum_upload_size=256 * 1024,  # 256 KiB for fast oversized test
            maximum_video_resolution=(640, 480),
            maximum_video_duration=10.0,
            maximum_processed_frames=500,
            max_queued_jobs=2,
            project_root=self._tmp.name,
        )
        self.cfg.validate()
        self.cfg.ensure_runtime_dirs()

        self.app = Flask(__name__)
        register_v2_api(self.app, self.cfg)
        self.client = self.app.test_client()
        self.valid_bytes = _make_mp4_bytes()

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_upload_returns_202(self):
        resp = self.client.post(
            "/api/v2/jobs",
            data={
                "video": (io.BytesIO(self.valid_bytes), "road_sample.mp4"),
                "mode": "live",
                "enable_depth": "false",
                "max_frames": "5",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 202)
        data = resp.get_json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "QUEUED")
        self.assertEqual(data["mode"], "live")
        self.assertEqual(data["status_url"], f"/api/v2/jobs/{data['job_id']}")

    def test_missing_and_empty_uploads_rejected(self):
        resp_missing = self.client.post("/api/v2/jobs", data={"mode": "live"})
        self.assertEqual(resp_missing.status_code, 400)

        resp_empty = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(b""), "empty.mp4"), "mode": "live"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_empty.status_code, 400)
        self.assertEqual(os.listdir(self.cfg.resolved_upload_dir), [])

    def test_unsupported_and_corrupt_videos_rejected_and_cleaned(self):
        resp_ext = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(b"hello"), "notes.txt"), "mode": "live"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_ext.status_code, 415)

        resp_corrupt = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(b"not_a_real_mp4_stream"), "fake.mp4"), "mode": "live"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_corrupt.status_code, 415)
        self.assertEqual(os.listdir(self.cfg.resolved_upload_dir), [])

    def test_oversized_upload_rejected_413_and_cleaned(self):
        big_bytes = b"0" * (self.cfg.maximum_upload_size + 1024)
        resp = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(big_bytes), "huge.mp4"), "mode": "live"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(os.listdir(self.cfg.resolved_upload_dir), [])

    def test_invalid_mode_params_and_unsafe_filename(self):
        resp_mode = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(self.valid_bytes), "a.mp4"), "mode": "turbo"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_mode.status_code, 400)

        resp_override = self.client.post(
            "/api/v2/jobs",
            data={
                "video": (io.BytesIO(self.valid_bytes), "a.mp4"),
                "mode": "live",
                "checkpoint_path": "C:/evil/weights.pt",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_override.status_code, 400)

        resp_unsafe = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(self.valid_bytes), "../../evil/../../clip.mp4"), "mode": "live"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_unsafe.status_code, 202)
        jid = resp_unsafe.get_json()["job_id"]
        job_dir = os.path.join(self.cfg.resolved_upload_dir, jid)
        self.assertTrue(os.path.isdir(job_dir))
        self.assertEqual(os.listdir(job_dir), [f"{jid}_input.mp4"])

    def test_queue_full_cleans_up_uploaded_file(self):
        for i in range(2):
            r = self.client.post(
                "/api/v2/jobs",
                data={"video": (io.BytesIO(self.valid_bytes), f"ok_{i}.mp4"), "mode": "live"},
                content_type="multipart/form-data",
            )
            self.assertEqual(r.status_code, 202)

        before_dirs = set(os.listdir(self.cfg.resolved_upload_dir))
        r_full = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(self.valid_bytes), "overflow.mp4"), "mode": "live"},
            content_type="multipart/form-data",
        )
        self.assertEqual(r_full.status_code, 429)
        after_dirs = set(os.listdir(self.cfg.resolved_upload_dir))
        self.assertEqual(before_dirs, after_dirs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
