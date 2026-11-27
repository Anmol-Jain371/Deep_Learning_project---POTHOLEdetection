"""Automated REST API integration tests for jobs, results, pagination, and safeguards (Phase 8B.4)."""

import csv
import io
import json
import math
import os
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np
from flask import Flask

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_api_v2 import register_v2_api, start_v2_worker, stop_v2_worker
from roadvision_api_v2.configuration import APIConfig


def _make_mp4_bytes(n_frames: int = 8) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        path = tf.name
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
        for _ in range(n_frames):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


class _StubResultPipeline:
    def __init__(self, cfg, session_id, device):
        self.cfg = cfg
        self.session_id = session_id
        self.device = device

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
        os.makedirs(output_dir, exist_ok=True)
        stem = stem_override or "clip"
        prefix = f"phase8a_{self.cfg.mode}_{stem}"
        total = max_frames or 6
        if on_init:
            on_init({"total_frames": total, "source_fps": 25.0, "width": 160, "height": 120})

        observations = []
        perf_rows = []
        state_counts = {
            "FRESH_VALID": 0, "FRESH_INVALID": 0, "CACHED_VALID": 0,
            "EXPIRED": 0, "NOT_EVALUATED": 0,
        }
        for i in range(total):
            if should_cancel and should_cancel():
                break
            state = "FRESH_VALID" if i == 0 else ("CACHED_VALID" if i < 4 else "FRESH_INVALID")
            state_counts[state] += 1
            valid = state in ("FRESH_VALID", "CACHED_VALID")
            observations.append({
                "frame_context": {"source_session_id": self.session_id, "frame_id": f"{self.session_id}_f{i:06d}", "frame_idx": i, "timestamp_sec": i * 0.04, "frame_wh": [160, 120], "mode": self.cfg.mode, "camera_profile": "sedan_dashcam", "gps_lat": None, "gps_lon": None, "has_gps": False},
                "detection": {"track_id": 1 if i < 4 else 2, "track_generation_id": 0, "box_xyxy": [10.0, 20.0, 50.0, 60.0], "detector_confidence": 0.85 if i < 4 else 0.55, "class_id": 0, "model_identity": "yolov8n_phase2_controlled", "detector_input_size": 640, "is_below_horizon_cutoff": True},
                "measurement": {
                    "source_session_id": self.session_id, "frame_id": f"{self.session_id}_f{i:06d}", "frame_idx": i, "timestamp_sec": i * 0.04, "track_id": 1 if i < 4 else 2, "track_generation_id": 0, "box_xyxy": [10.0, 20.0, 50.0, 60.0], "detector_confidence": 0.85 if i < 4 else 0.55,
                    "model_identity": "depth_anything_v2_small", "depth_input_resolution": 336, "inference_precision": "fp16_weights",
                    "raw_relative_cavity_score": 0.12 if state == "FRESH_VALID" else None,
                    "normalized_relative_score": 1.4 if state == "FRESH_VALID" else None,
                    "smoothed_relative_score": 0.12 if valid else None,
                    "measurement_validity": valid, "measurement_state": state,
                    "measurement_age_sec": 0.0 if state == "FRESH_VALID" else (i * 0.04 if state == "CACHED_VALID" else None),
                    "ref_diagnostics": {"used_ref_samples": 120 if state == "FRESH_VALID" else None, "inlier_ratio": 0.72 if state == "FRESH_VALID" else None},
                    "failure_reasons": [] if valid else ["insufficient_positive_depression"],
                    "warnings": [], "is_water_suspected": False, "fallback_used": False, "fallback_reason": None,
                },
                "horizon_diag": None,
            })
            perf_rows.append({"frame_id": f"{self.session_id}_f{i:06d}", "frame_idx": i, "mode": self.cfg.mode, "total_frame_ms": 12.5})
            if on_progress:
                on_progress({
                    "processed_frames": i + 1, "total_frames": total,
                    "detection_observation_count": len(observations), "unique_candidate_track_count": 2,
                    "depth_invocation_count": 2, "scheduler_request_count": 2, "failed_depth_count": 0,
                    "measurement_state_counts": dict(state_counts), "throughput_fps": 80.0,
                })

        if save_json:
            with open(os.path.join(output_dir, f"{prefix}_frames.json"), "w", encoding="utf-8") as fh:
                json.dump(observations, fh, allow_nan=False)
            with open(os.path.join(output_dir, f"{prefix}_tracks.json"), "w", encoding="utf-8") as fh:
                json.dump({
                    "1": {"source_session_id": self.session_id, "track_id": 1, "track_generation_id": 0, "first_seen_frame_idx": 0, "last_seen_frame_idx": 3, "observation_count": 4, "box_xyxy": [10.0, 20.0, 50.0, 60.0], "latest_state": "CACHED_VALID"},
                    "2": {"source_session_id": self.session_id, "track_id": 2, "track_generation_id": 0, "first_seen_frame_idx": 4, "last_seen_frame_idx": 5, "observation_count": 2, "box_xyxy": [10.0, 20.0, 50.0, 60.0], "latest_state": "FRESH_INVALID"},
                }, fh)
        if save_csv:
            with open(os.path.join(output_dir, f"{prefix}_performance.csv"), "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(perf_rows[0].keys()))
                w.writeheader()
                w.writerows(perf_rows)
        summary = {
            "session_id": self.session_id, "mode": self.cfg.mode, "source": source,
            "total_frames": total, "source_fps": 25.0, "elapsed_sec": 0.1, "end_to_end_fps": 60.0,
            "depth_enabled": True, "depth_model_id": "depth_anything_v2_small", "depth_input_size": 336,
            "scheduler_strategy": "E_budget_aware", "scheduler_depth_requests": 2, "depth_calls": 2,
            "failed_depth_inferences": 0, "unique_tracks": 2, "total_hazard_observations": len(observations),
            "measurement_state_counts": state_counts, "cancelled": False, "model_init_sec": 0.02, "artifact_generation_sec": 0.01,
        }
        with open(os.path.join(output_dir, f"{prefix}_session_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh)
        return summary


class TestV2API(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = APIConfig(
            enabled=True,
            database_path=os.path.join(self._tmp.name, "jobs.db"),
            upload_directory=os.path.join(self._tmp.name, "uploads"),
            artifact_directory=os.path.join(self._tmp.name, "artifacts"),
            default_page_size=2,
            max_page_size=5,
            project_root=self._tmp.name,
        )
        self.cfg.ensure_runtime_dirs()
        self.app = Flask(__name__)
        register_v2_api(
            self.app,
            self.cfg,
            pipeline_factory=lambda cfg, sid, dev: _StubResultPipeline(cfg, sid, dev),
            device="cpu",
        )
        self.client = self.app.test_client()
        self.mp4_bytes = _make_mp4_bytes(6)

    def tearDown(self):
        stop_v2_worker(self.app)
        self._tmp.cleanup()

    def _submit_and_wait(self):
        start_v2_worker(self.app, enforce_process_lock=False)
        resp = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(self.mp4_bytes), "sample.mp4"), "mode": "balanced", "save_video": "false", "max_frames": "6"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 202)
        jid = resp.get_json()["job_id"]
        for _ in range(80):
            st = self.client.get(f"/api/v2/jobs/{jid}").get_json()
            if st["status"] == "COMPLETED":
                return jid
            time.sleep(0.05)
        self.fail("Job did not complete")

    def test_unknown_job_id_and_premature_results(self):
        r404 = self.client.get("/api/v2/jobs/11111111-1111-4111-8111-111111111111")
        self.assertEqual(r404.status_code, 404)
        self.assertEqual(self.client.get("/api/v2/jobs/not-a-uuid").status_code, 400)

        # Submit without starting worker -> status is QUEUED -> summary returns 409
        r_sub = self.client.post(
            "/api/v2/jobs",
            data={"video": (io.BytesIO(self.mp4_bytes), "queued.mp4"), "mode": "live"},
            content_type="multipart/form-data",
        )
        jid = r_sub.get_json()["job_id"]
        self.assertEqual(self.client.get(f"/api/v2/jobs/{jid}/summary").status_code, 409)
        self.assertEqual(self.client.get(f"/api/v2/jobs/{jid}/observations").status_code, 409)

        # Cancel queued job -> 202 and terminal CANCELLED
        rc = self.client.post(f"/api/v2/jobs/{jid}/cancel")
        self.assertEqual(rc.status_code, 202)
        self.assertEqual(rc.get_json()["status"], "CANCELLED")
        # Second cancel on terminal state -> 409
        self.assertEqual(self.client.post(f"/api/v2/jobs/{jid}/cancel").status_code, 409)

    def test_list_summary_observations_tracks_performance_and_terminology(self):
        jid = self._submit_and_wait()

        r_list = self.client.get("/api/v2/jobs?page=1&page_size=5")
        self.assertEqual(r_list.status_code, 200)
        self.assertEqual(r_list.get_json()["total"], 1)

        r_sum = self.client.get(f"/api/v2/jobs/{jid}/summary")
        self.assertEqual(r_sum.status_code, 200)
        sdata = r_sum.get_json()
        self.assertIn("uncalibrated relative inverse-depth", sdata["scientific_disclaimer"].lower())
        # Fresh valid coverage must be separate from cached availability
        cov = sdata["measurement_coverage"]
        self.assertEqual(cov["denominator_total_observations"], 6)
        self.assertEqual(cov["counts_by_state"]["FRESH_VALID"], 1)
        self.assertEqual(cov["counts_by_state"]["CACHED_VALID"], 3)
        self.assertAlmostEqual(cov["fresh_valid_rate_over_observations"], round(1 / 6, 4))

        # Observation pagination + null semantics + filters
        r_obs1 = self.client.get(f"/api/v2/jobs/{jid}/observations?page=1&page_size=2")
        self.assertEqual(r_obs1.status_code, 200)
        odata1 = r_obs1.get_json()
        self.assertEqual(len(odata1["observations"]), 2)
        self.assertEqual(odata1["total"], 6)
        self.assertEqual(odata1["total_pages"], 3)

        r_cached = self.client.get(f"/api/v2/jobs/{jid}/observations?measurement_state=CACHED_VALID&page_size=5")
        cached_json = r_cached.get_json()
        cached_obs = cached_json["observations"]
        self.assertEqual(cached_json["total"], 3)
        self.assertEqual(len(cached_obs), 3)
        for item in cached_obs:
            self.assertIsNone(item["measurement"]["raw_relative_cavity_score"])
            self.assertIsNotNone(item["measurement"]["smoothed_relative_score"])

        # Verify NaN / Infinity values are never emitted as invalid JSON numbers
        from roadvision_api_v2.schemas import sanitize_json_value
        sanitized = sanitize_json_value({"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 1.25})
        self.assertEqual(sanitized, {"a": None, "b": None, "c": None, "d": 1.25})
        dumped = json.dumps(sanitized, allow_nan=False)
        self.assertNotIn("NaN", dumped)
        self.assertNotIn("Infinity", dumped)

        r_track2 = self.client.get(f"/api/v2/jobs/{jid}/observations?track_id=2&min_conf=0.5&max_conf=0.6")
        self.assertEqual(len(r_track2.get_json()["observations"]), 2)

        # Tracks pagination
        r_tr = self.client.get(f"/api/v2/jobs/{jid}/tracks?page=1&page_size=1")
        self.assertEqual(r_tr.status_code, 200)
        tdata = r_tr.get_json()
        self.assertEqual(len(tdata["tracks"]), 1)
        self.assertEqual(tdata["unique_session_track_count"], 2)

        # Performance retrieval
        r_perf = self.client.get(f"/api/v2/jobs/{jid}/performance?page=1&page_size=3")
        self.assertEqual(r_perf.status_code, 200)
        pdata = r_perf.get_json()
        self.assertEqual(len(pdata["frames"]), 3)
        self.assertEqual(pdata["total"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
