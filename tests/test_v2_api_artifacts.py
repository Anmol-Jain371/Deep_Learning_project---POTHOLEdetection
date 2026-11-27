"""Artifact manifest, Range video streaming, and partial artifact tests (Phase 8B.4)."""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from flask import Flask

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_api_v2 import register_v2_api
from roadvision_api_v2.artifacts import build_artifact_manifest
from roadvision_api_v2.configuration import APIConfig
from roadvision_api_v2.job_manager import JOB_STATUS_CANCELLED, JOB_STATUS_COMPLETED, JobManager


class TestV2APIArtifacts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = APIConfig(
            enabled=True,
            database_path=os.path.join(self._tmp.name, "jobs.db"),
            upload_directory=os.path.join(self._tmp.name, "uploads"),
            artifact_directory=os.path.join(self._tmp.name, "artifacts"),
            project_root=self._tmp.name,
        )
        self.cfg.ensure_runtime_dirs()
        self.app = Flask(__name__)
        self.rt = register_v2_api(self.app, self.cfg)
        self.client = self.app.test_client()
        self.jm: JobManager = self.rt["job_manager"]

    def tearDown(self):
        self._tmp.cleanup()

    def _create_video_artifact_job(self, status=JOB_STATUS_COMPLETED):
        out_dir = os.path.join(self.cfg.resolved_artifact_dir, "job_vid")
        os.makedirs(out_dir, exist_ok=True)
        j = self.jm.create_job(
            original_filename="road.mp4",
            validated_input_path=os.path.join(self.cfg.resolved_upload_dir, "road.mp4"),
            mode="balanced",
            config_snapshot={"mode": "balanced"},
            output_dir=out_dir,
            total_frames=5,
        )
        jid = j["job_id"]
        stem = f"{jid}_input"
        vid_path = os.path.join(out_dir, f"phase8a_balanced_{stem}_annotated.mp4")
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
        for _ in range(5):
            writer.write(np.full((120, 160, 3), 64, dtype=np.uint8))
        writer.release()

        sum_path = os.path.join(out_dir, f"phase8a_balanced_{stem}_session_summary.json")
        with open(sum_path, "w", encoding="utf-8") as fh:
            fh.write('{"mode": "balanced", "total_frames": 5}')

        manifest, avail = build_artifact_manifest(
            job_id=jid,
            mode="balanced",
            stem=stem,
            output_dir=out_dir,
            job_status=status,
            video_codec="mp4v",
            video_browser_compatible=False,
        )

        self.jm.claim_next_job()
        self.jm.mark_running(jid, total_frames=5)
        if status == JOB_STATUS_CANCELLED:
            self.jm.request_cancellation(jid)
        self.jm.complete_job(
            jid,
            processed_frames=5,
            total_frames=5,
            detection_observation_count=0,
            unique_candidate_track_count=0,
            depth_invocation_count=0,
            scheduler_request_count=0,
            failed_depth_count=0,
            measurement_state_counts={},
            throughput_fps=15.0,
            artifact_manifest=manifest,
            artifact_availability=avail,
            summary={"mode": "balanced", "total_frames": 5},
        )
        return jid, vid_path

    def test_artifact_manifest_and_http_range_video_streaming(self):
        jid, vid_path = self._create_video_artifact_job(JOB_STATUS_COMPLETED)
        r_man = self.client.get(f"/api/v2/jobs/{jid}/artifacts")
        self.assertEqual(r_man.status_code, 200)
        mdata = r_man.get_json()
        self.assertFalse(mdata["partial"])
        ids = {a["artifact_id"] for a in mdata["artifacts"]}
        self.assertIn("annotated_video", ids)
        self.assertIn("session_summary_json", ids)

        # Full download
        r_full = self.client.get(f"/api/v2/jobs/{jid}/artifacts/annotated_video")
        self.assertEqual(r_full.status_code, 200)
        self.assertEqual(r_full.headers.get("Accept-Ranges"), "bytes")
        fsize = os.path.getsize(vid_path)
        self.assertEqual(len(r_full.data), fsize)

        # HTTP Range 206 partial request
        r_range = self.client.get(
            f"/api/v2/jobs/{jid}/artifacts/annotated_video",
            headers={"Range": "bytes=0-127"},
        )
        self.assertEqual(r_range.status_code, 206)
        self.assertEqual(len(r_range.data), min(128, fsize))
        self.assertIn("bytes 0-", r_range.headers.get("Content-Range", ""))

        # Invalid Range -> 416
        r_bad_range = self.client.get(
            f"/api/v2/jobs/{jid}/artifacts/annotated_video",
            headers={"Range": f"bytes={fsize + 500}-{fsize + 900}"},
        )
        self.assertEqual(r_bad_range.status_code, 416)

    def test_cancelled_job_marks_partial_artifacts(self):
        jid, _ = self._create_video_artifact_job(JOB_STATUS_CANCELLED)
        r_man = self.client.get(f"/api/v2/jobs/{jid}/artifacts")
        self.assertEqual(r_man.status_code, 200)
        mdata = r_man.get_json()
        self.assertTrue(mdata["partial"])
        self.assertTrue(all(a["partial"] for a in mdata["artifacts"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
