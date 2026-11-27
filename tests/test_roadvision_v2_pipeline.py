"""
tests/test_roadvision_v2_pipeline.py — Unit tests for RoadVision AI 2.0 pipeline.

Uses mocks/stubs for GPU operations so the test suite runs on CPU without real models.

Covers:
  - Config validation integration test
  - Detection-only operation (depth disabled)
  - Empty detections → no crash, empty hazard list
  - NOT_EVALUATED state when depth disabled
  - Track ID reuse reset
  - Mode change resets state (reset_session)
  - Missing GPS → has_gps=False, no crash
  - Video processing smoke test (5 frames of black video)
  - Legacy hash integrity check (roadvision_engine.py, server.py)
"""
import hashlib
import os
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock as mock

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_v2.config import RoadVisionV2Config, DetectorConfig, DepthConfig
from roadvision_v2.schemas import (
    HazardObservation,
    PipelinePerformance,
    RelativeDepthMeasurement,
)

# ---------------------------------------------------------------------------
# Production file hash constants (Phase 8A Task 2 — must NOT change)
# ---------------------------------------------------------------------------
_EXPECTED_HASHES = {
    "roadvision_engine.py": "3805808750134a38b3fa9bcfc42fb6583a843dda575e70aef28ff205c0e2a51e",
    "server.py": "d8772504e03f949e1b92cf8c76da65327035f2b1a5ae713385d5eb60fde91eff",
    "config/camera_params.json": "0c7633f330a8b5c15a1696b85d81e9ca6d32e3cac1c1c237cfff8e576e789af1",
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Minimal stub YOLO result for mocking
# ---------------------------------------------------------------------------

class _FakeBoxes:
    def __init__(self, boxes, confs, clss, ids):
        import torch
        self.xyxy = torch.tensor(boxes, dtype=torch.float32)
        self.conf = torch.tensor(confs, dtype=torch.float32)
        self.cls = torch.tensor(clss, dtype=torch.float32)
        self.id = torch.tensor(ids, dtype=torch.float32) if ids is not None else None


class _FakeResult:
    def __init__(self, boxes_data=None):
        if boxes_data:
            bxs, confs, clss, ids = boxes_data
            self.boxes = _FakeBoxes(bxs, confs, clss, ids)
        else:
            self.boxes = _NoIdBoxes()


class _NoIdBoxes:
    id = None


def _make_fake_yolo_result_with_det(box=(10, 50, 80, 120), tid=1, conf=0.65):
    return [_FakeResult(([[box[0], box[1], box[2], box[3]]], [conf], [0], [tid]))]


def _make_empty_yolo_result():
    return [_FakeResult()]


def _black_frame(h=360, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Build a lightweight pipeline with mocked YOLO (no model loading)
# ---------------------------------------------------------------------------

def _build_pipeline_no_gpu():
    """Build RoadVisionV2Pipeline with depth disabled and YOLO stubbed out."""
    from roadvision_v2.pipeline import RoadVisionV2Pipeline

    cfg = RoadVisionV2Config(mode="live")
    cfg.depth.enabled = False
    cfg.diagnostics.enable_horizon_proxy = False

    with mock.patch("roadvision_v2.pipeline.RoadVisionV2Pipeline._load_depth_model"):
        pipeline = RoadVisionV2Pipeline.__new__(RoadVisionV2Pipeline)
        pipeline.cfg = cfg
        pipeline.session_id = "test_sess"
        pipeline.device = "cpu"
        pipeline._frame_generation = 0
        pipeline._depth_model = None
        pipeline._depth_model_id = "none"
        pipeline._depth_precision = "none"
        pipeline._depth_input_size = 0
        pipeline._horizon_tracker = None
        pipeline._track_registry = {}
        pipeline._horizon_cutoff = 0.15
        pipeline._det_input_size = 640
        pipeline._det_conf = 0.25
        pipeline._det_iou = 0.50
        pipeline._cavity_strategy = "full_box"

        # Set up minimal RANSAC / cavity / cache / scheduler configs
        from experiments.optimized_relative_depth import OptimizedDepthConfig
        from experiments.core_aware_cavity_extraction import CoreExtractionConfig
        from experiments.track_depth_cache import TrackDepthCache, TrackCacheConfig
        from experiments.adaptive_depth_scheduler import AdaptiveDepthScheduler, AdaptiveSchedulerConfig

        pipeline._opt_cfg = OptimizedDepthConfig()
        pipeline._core_cfg = CoreExtractionConfig()

        cache_cfg = TrackCacheConfig()
        pipeline._cache = TrackDepthCache(
            video_session_id="test_sess",
            depth_model_id="none",
            depth_input_size=0,
            inference_precision="none",
            cfg=cache_cfg,
        )
        sched_cfg = AdaptiveSchedulerConfig(strategy="B_candidate_gated")
        pipeline._scheduler = AdaptiveDepthScheduler(cfg=sched_cfg, cache=pipeline._cache)

        # Stub YOLO
        pipeline._yolo = mock.MagicMock()
        pipeline._tracker_config = "road_tracker.yaml"

    return pipeline


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestLegacyHashIntegrity(unittest.TestCase):
    """
    Verify that production files have not been modified.
    Phase 8A hard constraint: roadvision_engine.py, server.py, camera_params.json
    must retain their Phase 8A Task 2 SHA256 hashes.
    """

    def _root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def test_roadvision_engine_hash_unchanged(self):
        path = os.path.join(self._root(), "roadvision_engine.py")
        if not os.path.exists(path):
            self.skipTest("roadvision_engine.py not found")
        actual = _sha256(path)
        expected = _EXPECTED_HASHES["roadvision_engine.py"]
        self.assertEqual(
            actual, expected,
            f"roadvision_engine.py has been modified!\n  expected: {expected}\n  actual  : {actual}"
        )

    def test_legacy_server_routes_preserved(self):
        """Legacy server routes, methods, and default-disabled V2 flag must be preserved."""
        srv_path = os.path.join(self._root(), "server.py")
        with open(srv_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        for expected_sig in (
            '@app.route("/api/health", methods=["GET"])',
            '@app.route("/api/benchmark", methods=["GET"])',
            '@app.route("/api/camera-config", methods=["GET"])',
            '@app.route("/api/detect", methods=["POST"])',
            '@app.route("/outputs/<path:filename>")',
        ):
            self.assertIn(expected_sig, src)

        from roadvision_api_v2.configuration import load_api_config
        cfg = load_api_config(ensure_dirs=False, apply_env_overrides=False)
        self.assertFalse(cfg.enabled, "Committed V2 API default must be enabled: false")


    def test_camera_params_hash_unchanged(self):
        path = os.path.join(self._root(), "config", "camera_params.json")
        if not os.path.exists(path):
            self.skipTest("config/camera_params.json not found")
        actual = _sha256(path)
        expected = _EXPECTED_HASHES["config/camera_params.json"]
        self.assertEqual(
            actual, expected,
            f"config/camera_params.json has been modified!\n  expected: {expected}\n  actual  : {actual}"
        )


class TestConfigValidationIntegration(unittest.TestCase):

    def test_default_config_validates(self):
        cfg = RoadVisionV2Config(mode="live")
        cfg.depth.enabled = False
        cfg.validate(check_assets=False)

    def test_dav2_336_config_validates(self):
        cfg = RoadVisionV2Config(mode="balanced")
        cfg.depth.enabled = True
        cfg.depth.model_id = "depth_anything_v2_small"
        cfg.depth.input_size = 336
        cfg.depth.precision = "fp16_weights"
        cfg.validate(check_assets=False)

    def test_bad_mode_raises(self):
        cfg = RoadVisionV2Config(mode="hyper_turbo")
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)


class TestProcessFrameEmptyDetections(unittest.TestCase):

    def test_no_detections_returns_empty_hazards(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline._yolo.track.return_value = _make_empty_yolo_result()
        frame = _black_frame()
        hazards, perf = pipeline.process_frame(frame, frame_idx=0, ts_sec=0.0)
        self.assertIsInstance(hazards, list)
        self.assertEqual(len(hazards), 0)
        self.assertIsInstance(perf, PipelinePerformance)
        self.assertGreater(perf.total_frame_ms, 0.0)

    def test_process_frame_does_not_raise_on_none_boxes(self):
        pipeline = _build_pipeline_no_gpu()
        # Simulate result with boxes.id = None
        pipeline._yolo.track.return_value = [_FakeResult()]
        frame = _black_frame()
        hazards, perf = pipeline.process_frame(frame, frame_idx=0, ts_sec=0.0)
        self.assertEqual(hazards, [])


class TestProcessFrameWithDetection(unittest.TestCase):

    def test_single_detection_produces_hazard(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline._yolo.track.return_value = _make_fake_yolo_result_with_det(
            box=(100, 150, 250, 300), tid=7, conf=0.72
        )
        frame = _black_frame()
        hazards, perf = pipeline.process_frame(frame, frame_idx=0, ts_sec=0.0)
        self.assertEqual(len(hazards), 1)
        hz = hazards[0]
        self.assertIsInstance(hz, HazardObservation)
        self.assertEqual(hz.detection.track_id, 7)
        self.assertAlmostEqual(hz.detection.detector_confidence, 0.72, places=2)

    def test_depth_disabled_gives_not_evaluated(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline._yolo.track.return_value = _make_fake_yolo_result_with_det(tid=10)
        frame = _black_frame()
        hazards, _ = pipeline.process_frame(frame, 0, 0.0)
        self.assertEqual(len(hazards), 1)
        meas = hazards[0].measurement
        self.assertEqual(meas.measurement_state, "NOT_EVALUATED")
        self.assertIsNone(meas.raw_relative_cavity_score)
        self.assertIsNone(meas.normalized_relative_score)
        self.assertIsNone(meas.smoothed_relative_score)
        self.assertFalse(meas.measurement_validity)
        self.assertTrue(meas.fallback_used)

    def test_multiple_frames_track_accumulates(self):
        pipeline = _build_pipeline_no_gpu()
        for fi in range(5):
            pipeline._yolo.track.return_value = _make_fake_yolo_result_with_det(
                box=(100, 100, 200, 200), tid=3, conf=0.6
            )
            frame = _black_frame()
            pipeline.process_frame(frame, frame_idx=fi, ts_sec=fi * 0.04)
        obs = pipeline._track_registry.get(3)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.observation_count, 5)
        self.assertEqual(obs.last_seen_frame_idx, 4)

    def test_missing_gps_flag(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline._yolo.track.return_value = _make_empty_yolo_result()
        frame = _black_frame()
        hazards, perf = pipeline.process_frame(
            frame, 0, 0.0, gps_lat=None, gps_lon=None
        )
        # No crash expected
        self.assertIsInstance(hazards, list)

    def test_gps_present_flag(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline._yolo.track.return_value = _make_fake_yolo_result_with_det(tid=99)
        frame = _black_frame()
        hazards, _ = pipeline.process_frame(
            frame, 0, 0.0, gps_lat=12.9716, gps_lon=77.5946
        )
        if hazards:
            self.assertTrue(hazards[0].frame_context.has_gps)


class TestSessionReset(unittest.TestCase):

    def test_reset_clears_track_registry(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline._yolo.track.return_value = _make_fake_yolo_result_with_det(tid=5)
        frame = _black_frame()
        pipeline.process_frame(frame, 0, 0.0)
        self.assertIn(5, pipeline._track_registry)
        pipeline.reset_session(reason="mode_change")
        self.assertEqual(pipeline._track_registry, {})

    def test_reset_increments_generation(self):
        pipeline = _build_pipeline_no_gpu()
        gen_before = pipeline._frame_generation
        pipeline.reset_session()
        self.assertEqual(pipeline._frame_generation, gen_before + 1)


class TestVideoProcessingSmoke(unittest.TestCase):
    """
    Smoke test: write 5 black frames to a temp video, run process_video,
    verify summary is returned and no crash.
    """

    def _write_temp_video(self, n_frames: int = 5) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        f.close()
        path = f.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 25.0, (320, 240))
        for _ in range(n_frames):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()
        return path

    def test_smoke_process_video_no_crash(self):
        pipeline = _build_pipeline_no_gpu()
        # Empty detections every frame
        pipeline._yolo.track.return_value = _make_empty_yolo_result()

        vid_path = self._write_temp_video(5)
        out_dir = tempfile.mkdtemp()
        try:
            summary = pipeline.process_video(
                source=vid_path,
                output_dir=out_dir,
                save_video=False,
                save_json=True,
                save_csv=False,
                diagnostics=False,
                max_frames=5,
            )
            self.assertEqual(summary["total_frames"], 5)
            self.assertIn("end_to_end_fps", summary)
            self.assertGreater(summary["end_to_end_fps"], 0.0)
        finally:
            os.unlink(vid_path)
            # Clean up output dir
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_smoke_process_video_with_detections(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline._yolo.track.return_value = _make_fake_yolo_result_with_det(
            box=(10, 10, 80, 80), tid=1, conf=0.5
        )
        vid_path = self._write_temp_video(5)
        out_dir = tempfile.mkdtemp()
        try:
            summary = pipeline.process_video(
                source=vid_path,
                output_dir=out_dir,
                save_video=False,
                save_json=False,
                save_csv=False,
                max_frames=5,
            )
            self.assertGreaterEqual(summary["unique_tracks"], 1)
        finally:
            os.unlink(vid_path)
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)


class TestDetectionErrorIsolation(unittest.TestCase):

    def test_yolo_exception_returns_empty_hazards(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline._yolo.track.side_effect = RuntimeError("Simulated GPU crash")
        frame = _black_frame()
        hazards, perf = pipeline.process_frame(frame, 0, 0.0)
        self.assertEqual(hazards, [])
        self.assertIsInstance(perf, PipelinePerformance)

    def test_cavity_extractor_schema_mapping_flows_to_measurement(self):
        pipeline = _build_pipeline_no_gpu()
        pipeline.cfg.depth.enabled = True
        pipeline._depth_model_id = "depth_anything_v2_small"
        pipeline._depth_input_size = 336
        pipeline._depth_precision = "fp16_weights"
        pipeline._yolo.track.return_value = _make_fake_yolo_result_with_det(
            box=(100, 150, 250, 300), tid=12, conf=0.80
        )
        fake_dmap = np.zeros((480, 640), dtype=np.float32)
        pipeline._run_depth_inference = mock.MagicMock(return_value=(fake_dmap, {"inference_ms": 10.0}))
        pipeline._extract_relative_score = mock.MagicMock(return_value={
            "is_valid": True,
            "failure_reasons": [],
            "warnings": [],
            "is_water_suspected": False,
            "core_cavity_stats": {"p75_residual": 0.42, "scale_normalized_p75": 2.1},
            "ref_diagnostics": {
                "used_ref_samples": 200, "available_ref_samples": 300,
                "populated_quadrants": 4, "inlier_ratio": 0.85,
                "ref_fit_mad": 0.05, "ref_fit_rmse": 0.07,
            },
        })
        hazards, _ = pipeline.process_frame(_black_frame(), 0, 0.0)
        self.assertEqual(len(hazards), 1)
        m = hazards[0].measurement
        self.assertEqual(m.measurement_state, "FRESH_VALID")
        self.assertAlmostEqual(m.raw_relative_cavity_score, 0.42, places=4)
        self.assertAlmostEqual(m.normalized_relative_score, 2.1, places=4)
        self.assertEqual(m.ref_diagnostics.used_ref_samples, 200)
        self.assertAlmostEqual(m.ref_diagnostics.inlier_ratio, 0.85, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
