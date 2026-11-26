"""
tests/test_roadvision_v2_schemas.py — Unit tests for RoadVision AI 2.0 typed data contracts.

Covers:
  - RelativeDepthMeasurement.__post_init__ nulls out scores on invalid states
  - FRESH_INVALID → raw_score and normalized_score are None; smoothed also None
  - CACHED_VALID retains smoothed_score but raw/normalized are None
  - to_dict() serialization produces only JSON-safe types (no numpy)
  - Invalid measurement_state raises ValueError
  - All dataclasses instantiate without required args (defaults exist)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_v2.schemas import (
    MEASUREMENT_STATES,
    FrameContext,
    DetectionObservation,
    TrackObservation,
    ReferenceFitDiagnostics,
    SchedulingDecision,
    RelativeDepthMeasurement,
    HazardObservation,
    PipelinePerformance,
    _clean_json,
)


def _make_rdm(
    state: str = "FRESH_VALID",
    measurement_validity: bool = True,
    raw_score: float = 0.123,
    norm_score: float = 1.45,
    smoothed_score: float = 0.111,
) -> RelativeDepthMeasurement:
    return RelativeDepthMeasurement(
        source_session_id="test_session",
        frame_id="test_f000001",
        frame_idx=1,
        timestamp_sec=0.04,
        track_id=42,
        track_generation_id=0,
        box_xyxy=(10.0, 20.0, 80.0, 100.0),
        detector_confidence=0.75,
        model_identity="depth_anything_v2_small",
        depth_input_resolution=336,
        inference_precision="fp16_weights",
        raw_relative_cavity_score=raw_score,
        normalized_relative_score=norm_score,
        smoothed_relative_score=smoothed_score,
        measurement_validity=measurement_validity,
        measurement_state=state,
        measurement_age_sec=None,
    )


class TestMeasurementStates(unittest.TestCase):

    def test_known_states_complete(self):
        self.assertIn("FRESH_VALID", MEASUREMENT_STATES)
        self.assertIn("FRESH_INVALID", MEASUREMENT_STATES)
        self.assertIn("CACHED_VALID", MEASUREMENT_STATES)
        self.assertIn("EXPIRED", MEASUREMENT_STATES)
        self.assertIn("NOT_EVALUATED", MEASUREMENT_STATES)

    def test_invalid_state_raises_value_error(self):
        with self.assertRaises(ValueError):
            _make_rdm(state="TOTALLY_MADE_UP", measurement_validity=True)

    def test_empty_state_raises(self):
        with self.assertRaises(ValueError):
            _make_rdm(state="", measurement_validity=True)


class TestFreshValidMeasurement(unittest.TestCase):

    def test_fresh_valid_scores_retained(self):
        rdm = _make_rdm(state="FRESH_VALID", measurement_validity=True,
                        raw_score=0.5, norm_score=2.0, smoothed_score=0.45)
        self.assertIsNotNone(rdm.raw_relative_cavity_score)
        self.assertIsNotNone(rdm.normalized_relative_score)
        self.assertIsNotNone(rdm.smoothed_relative_score)

    def test_fresh_valid_to_dict_no_numpy(self):
        rdm = _make_rdm(state="FRESH_VALID", measurement_validity=True)
        d = rdm.to_dict()
        self._assert_no_numpy(d)

    def _assert_no_numpy(self, obj):
        if isinstance(obj, dict):
            for v in obj.values():
                self._assert_no_numpy(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._assert_no_numpy(v)
        else:
            self.assertNotIsInstance(obj, np.generic,
                msg=f"Found numpy scalar {type(obj)} in serialized output")
            self.assertNotIsInstance(obj, np.ndarray,
                msg="Found numpy array in serialized output")


class TestFreshInvalidMeasurement(unittest.TestCase):

    def test_fresh_invalid_nulls_raw_and_norm(self):
        rdm = _make_rdm(state="FRESH_INVALID", measurement_validity=False,
                        raw_score=0.9, norm_score=3.0, smoothed_score=0.5)
        self.assertIsNone(rdm.raw_relative_cavity_score)
        self.assertIsNone(rdm.normalized_relative_score)

    def test_fresh_invalid_nulls_smoothed(self):
        """Non-CACHED_VALID state must also null smoothed_score when invalid."""
        rdm = _make_rdm(state="FRESH_INVALID", measurement_validity=False,
                        smoothed_score=0.5)
        self.assertIsNone(rdm.smoothed_relative_score)

    def test_fresh_invalid_serializes_cleanly(self):
        rdm = _make_rdm(state="FRESH_INVALID", measurement_validity=False)
        d = rdm.to_dict()
        self.assertIsNone(d["raw_relative_cavity_score"])
        self.assertIsNone(d["normalized_relative_score"])
        self.assertIsNone(d["smoothed_relative_score"])


class TestCachedValidMeasurement(unittest.TestCase):

    def test_cached_valid_nulls_raw_norm_keeps_smoothed(self):
        """CACHED_VALID: raw/norm = None, but smoothed_score is NOT nulled."""
        rdm = _make_rdm(state="CACHED_VALID", measurement_validity=False,
                        raw_score=0.9, norm_score=3.0, smoothed_score=0.42)
        # raw/norm should be null because measurement_validity=False
        self.assertIsNone(rdm.raw_relative_cavity_score)
        self.assertIsNone(rdm.normalized_relative_score)
        # smoothed is preserved for CACHED_VALID
        self.assertIsNotNone(rdm.smoothed_relative_score)
        self.assertAlmostEqual(rdm.smoothed_relative_score, 0.42)

    def test_cached_valid_state_field(self):
        rdm = _make_rdm(state="CACHED_VALID", measurement_validity=False, smoothed_score=0.3)
        self.assertEqual(rdm.measurement_state, "CACHED_VALID")


class TestExpiredMeasurement(unittest.TestCase):

    def test_expired_nulls_all_scores(self):
        rdm = _make_rdm(state="EXPIRED", measurement_validity=False,
                        raw_score=0.4, norm_score=1.1, smoothed_score=0.3)
        self.assertIsNone(rdm.raw_relative_cavity_score)
        self.assertIsNone(rdm.normalized_relative_score)
        self.assertIsNone(rdm.smoothed_relative_score)


class TestNotEvaluatedMeasurement(unittest.TestCase):

    def test_not_evaluated_nulls_all_scores(self):
        rdm = _make_rdm(state="NOT_EVALUATED", measurement_validity=False,
                        raw_score=0.0, norm_score=0.0, smoothed_score=0.0)
        self.assertIsNone(rdm.raw_relative_cavity_score)
        self.assertIsNone(rdm.normalized_relative_score)
        self.assertIsNone(rdm.smoothed_relative_score)


class TestToDict(unittest.TestCase):

    def test_to_dict_has_all_fields(self):
        rdm = _make_rdm(state="FRESH_VALID", measurement_validity=True)
        d = rdm.to_dict()
        for field in (
            "source_session_id", "frame_id", "frame_idx", "timestamp_sec",
            "track_id", "track_generation_id", "box_xyxy", "detector_confidence",
            "model_identity", "depth_input_resolution", "inference_precision",
            "raw_relative_cavity_score", "normalized_relative_score", "smoothed_relative_score",
            "measurement_validity", "measurement_state", "measurement_age_sec",
            "failure_reasons", "warnings", "is_water_suspected",
            "fallback_used", "fallback_reason",
        ):
            self.assertIn(field, d, f"Missing field: {field}")

    def test_to_dict_box_is_list(self):
        rdm = _make_rdm()
        d = rdm.to_dict()
        self.assertIsInstance(d["box_xyxy"], (list, tuple))
        self.assertEqual(len(d["box_xyxy"]), 4)

    def test_ref_diagnostics_serializes(self):
        rdm = _make_rdm(state="FRESH_VALID", measurement_validity=True)
        d = rdm.to_dict()
        self.assertIn("ref_diagnostics", d)
        self.assertIsInstance(d["ref_diagnostics"], dict)


class TestCleanJson(unittest.TestCase):

    def test_numpy_scalar_converted(self):
        v = np.float32(1.5)
        result = _clean_json(v)
        self.assertIsInstance(result, float)

    def test_nested_dict_cleaned(self):
        d = {"a": np.int64(3), "b": [np.float64(1.0), "text"]}
        result = _clean_json(d)
        self.assertIsInstance(result["a"], int)
        self.assertIsInstance(result["b"][0], float)

    def test_plain_values_pass_through(self):
        result = _clean_json({"x": 1, "y": "hello", "z": None})
        self.assertEqual(result, {"x": 1, "y": "hello", "z": None})


class TestOtherDataclasses(unittest.TestCase):

    def test_frame_context_instantiates(self):
        fc = FrameContext(
            source_session_id="s1", frame_id="f1", frame_idx=0,
            timestamp_sec=0.0, frame_wh=(640, 360), mode="live",
        )
        d = fc.to_dict()
        self.assertEqual(d["mode"], "live")

    def test_detection_observation_instantiates(self):
        det = DetectionObservation(
            track_id=1, track_generation_id=0,
            box_xyxy=(0.0, 0.0, 100.0, 80.0),
            detector_confidence=0.6, class_id=0,
            model_identity="yolov8n", detector_input_size=640,
        )
        d = det.to_dict()
        self.assertEqual(d["track_id"], 1)

    def test_scheduling_decision_serializes(self):
        sd = SchedulingDecision(
            frame_id="f1", frame_idx=0, timestamp_sec=0.0,
            strategy="E_budget_aware", run_depth=False,
            frame_reasons=["NO_CANDIDATES"], per_track_reasons={},
        )
        d = sd.to_dict()
        self.assertFalse(d["run_depth"])

    def test_pipeline_performance_serializes(self):
        perf = PipelinePerformance(frame_id="f0", frame_idx=0, mode="balanced")
        d = perf.to_dict()
        self.assertIn("total_frame_ms", d)

    def test_ref_fit_diagnostics_instantiates(self):
        rd = ReferenceFitDiagnostics(used_ref_samples=400, inlier_ratio=0.72)
        d = rd.to_dict()
        self.assertEqual(d["used_ref_samples"], 400)

    def test_track_observation_instantiates(self):
        to = TrackObservation(
            source_session_id="s1", track_id=5, track_generation_id=0,
            first_seen_frame_idx=0, last_seen_frame_idx=10,
            observation_count=11, box_xyxy=(5.0, 5.0, 50.0, 40.0),
            latest_state="CACHED_VALID",
        )
        d = to.to_dict()
        self.assertEqual(d["track_id"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
