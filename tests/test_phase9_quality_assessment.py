"""
tests/test_phase9_quality_assessment.py — Unit and integration tests for Phase 9
uncertainty-aware hazard measurement quality assessment.
"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.core_aware_cavity_extraction import extract_cavity_with_strategy
from experiments.optimized_relative_depth import OptimizedDepthConfig
from roadvision_v2.config import QualityConfig
from roadvision_v2.quality import (
    assess_observation_quality,
    build_track_quality_summary,
    compute_track_temporal_stability,
    explain_diagnostic_codes,
    summarize_session_quality,
)
from roadvision_v2.schemas import (
    RelativeDepthMeasurement,
    ReferenceFitDiagnostics,
    TrackObservation,
)


class TestPhase9QualityAssessment(unittest.TestCase):
    def setUp(self):
        self.q_cfg = QualityConfig()

    def test_high_quality_synthetic_reference_surface(self):
        H, W = 120, 160
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        # Clean planar road surface with slight tilt + depressed cavity inside box
        depth_map = 1.0 + 0.002 * yy - 0.001 * xx
        box = (55.0, 40.0, 105.0, 80.0)
        cx, cy = 80.0, 60.0
        ell = ((xx - cx) / 18.0) ** 2 + ((yy - cy) / 14.0) ** 2
        depth_map[ell <= 1.0] -= 0.18 * (1.0 - ell[ell <= 1.0])

        opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
        ext = extract_cavity_with_strategy(
            depth_map=depth_map,
            box_xyxy=box,
            strategy="elliptical_core",
            opt_cfg=opt_cfg,
        )
        self.assertTrue(ext["is_valid"])
        qa = assess_observation_quality(
            measurement_state="FRESH_VALID",
            measurement_validity=True,
            detector_confidence=0.88,
            ref_diagnostics=ext["ref_diagnostics"],
            cavity_diagnostics={
                "cavity_sample_count": ext["core_cavity_stats"]["px"],
                "positive_frac": ext["core_cavity_stats"]["positive_frac"],
                "iqr_residual": ext["core_cavity_stats"]["iqr_residual"],
                "p75_residual": ext["core_cavity_stats"]["p75_residual"],
            },
            is_water_suspected=ext["is_water_suspected"],
            box_xyxy=box,
            frame_wh=(W, H),
            cfg=self.q_cfg,
        )
        self.assertEqual(qa["quality_category"], "HIGH_QUALITY")
        self.assertIsNotNone(qa["quality_index"])
        self.assertGreaterEqual(qa["quality_index"], 0.75)
        self.assertFalse(qa["is_calibrated_probability"])
        self.assertTrue(qa["is_experimental_heuristic"])

    def test_noisy_reference_surface_lowers_quality(self):
        rng = np.random.default_rng(7)
        H, W = 120, 160
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        depth_map = 1.0 + rng.normal(0.0, 0.08, size=(H, W))
        box = (55.0, 40.0, 105.0, 80.0)
        depth_map[48:72, 65:95] -= 0.20

        ext = extract_cavity_with_strategy(depth_map=depth_map, box_xyxy=box, strategy="elliptical_core")
        qa = assess_observation_quality(
            measurement_state="FRESH_VALID" if ext["is_valid"] else "FRESH_INVALID",
            measurement_validity=ext["is_valid"],
            detector_confidence=0.85,
            ref_diagnostics=ext["ref_diagnostics"],
            cavity_diagnostics={
                "cavity_sample_count": ext["core_cavity_stats"]["px"],
                "positive_frac": ext["core_cavity_stats"]["positive_frac"],
                "iqr_residual": ext["core_cavity_stats"]["iqr_residual"],
                "p75_residual": ext["core_cavity_stats"]["raw_diag_p75"],
            },
            failure_reasons=ext["failure_reasons"],
            cfg=self.q_cfg,
        )
        # Noisy plane either fails validity (INVALID) or has lower geometric quality than clean plane
        if not ext["is_valid"]:
            self.assertEqual(qa["quality_category"], "INVALID")
            self.assertIsNone(qa["quality_index"])
        else:
            self.assertLess(qa["components"]["geometric_reference_quality"], 0.88)

    def test_insufficient_samples_and_missing_diagnostics(self):
        qa_missing = assess_observation_quality(
            measurement_state="FRESH_VALID",
            measurement_validity=True,
            detector_confidence=0.92,
            ref_diagnostics=None,
            cavity_diagnostics=None,
            cfg=self.q_cfg,
        )
        self.assertEqual(qa_missing["quality_category"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(qa_missing["quality_index"])
        self.assertIsNone(qa_missing["components"]["geometric_reference_quality"])
        self.assertIn("missing_ref_diagnostics", qa_missing["missing_evidence"])

        # Temporal stability with < 3 fresh samples returns None for variability metrics
        stab = compute_track_temporal_stability(
            [
                {
                    "measurement_state": "FRESH_VALID",
                    "measurement_validity": True,
                    "raw_relative_cavity_score": 0.05,
                    "model_identity": "depth_anything_v2_small",
                    "depth_input_resolution": 336,
                    "inference_precision": "fp16_weights",
                },
                {
                    "measurement_state": "FRESH_VALID",
                    "measurement_validity": True,
                    "raw_relative_cavity_score": 0.06,
                    "model_identity": "depth_anything_v2_small",
                    "depth_input_resolution": 336,
                    "inference_precision": "fp16_weights",
                },
            ],
            cfg=self.q_cfg,
        )
        self.assertEqual(stab["evidence_sufficiency"], "INSUFFICIENT")
        self.assertIsNone(stab["median_relative_score"])
        self.assertIsNone(stab["mad_relative_score"])
        self.assertIsNotNone(stab["insufficient_evidence_reason"])

    def test_invalid_and_expired_never_valid_despite_high_confidence(self):
        for state in ("FRESH_INVALID", "EXPIRED"):
            qa = assess_observation_quality(
                measurement_state=state,
                measurement_validity=False,
                detector_confidence=0.99,
                ref_diagnostics={
                    "used_ref_samples": 512,
                    "populated_quadrants": 4,
                    "inlier_ratio": 0.95,
                    "ref_fit_mad": 0.005,
                },
                cavity_diagnostics={"cavity_sample_count": 300, "positive_frac": 0.9},
                failure_reasons=["unexpected_residual_polarity"] if state == "FRESH_INVALID" else [],
                cfg=self.q_cfg,
            )
            self.assertEqual(qa["quality_category"], "INVALID")
            self.assertIsNone(qa["quality_index"])
            self.assertEqual(qa["components"]["measurement_freshness"], 0.0)
            self.assertTrue(len(qa["diagnostic_explanations"]) >= 1)

    def test_cached_aging_and_repeated_cache_never_inflates_quality(self):
        fresh_qa = assess_observation_quality(
            measurement_state="FRESH_VALID",
            measurement_validity=True,
            detector_confidence=0.85,
            ref_diagnostics={
                "used_ref_samples": 400,
                "populated_quadrants": 4,
                "inlier_ratio": 0.86,
                "ref_fit_mad": 0.01,
            },
            cavity_diagnostics={"cavity_sample_count": 200, "positive_frac": 0.92, "iqr_residual": 0.02, "p75_residual": 0.05},
            is_water_suspected=False,
            cfg=self.q_cfg,
        )
        src_fresh = {
            **fresh_qa,
            "raw_ref_diagnostics": {
                "used_ref_samples": 400,
                "populated_quadrants": 4,
                "inlier_ratio": 0.86,
                "ref_fit_mad": 0.01,
            },
            "raw_cavity_diagnostics": {"cavity_sample_count": 200, "positive_frac": 0.92, "iqr_residual": 0.02, "p75_residual": 0.05},
            "is_water_suspected": False,
            "source_timestamp_sec": 1.0,
        }
        young_cached = assess_observation_quality(
            measurement_state="CACHED_VALID",
            measurement_validity=True,
            detector_confidence=0.85,
            measurement_age_sec=0.05,
            source_fresh_quality=src_fresh,
            cfg=self.q_cfg,
        )
        old_cached = assess_observation_quality(
            measurement_state="CACHED_VALID",
            measurement_validity=True,
            detector_confidence=0.85,
            measurement_age_sec=0.40,
            source_fresh_quality=src_fresh,
            cfg=self.q_cfg,
        )
        self.assertLessEqual(young_cached["quality_index"], fresh_qa["quality_index"])
        self.assertLess(old_cached["quality_index"], young_cached["quality_index"])

        # Repeated cached observations must NOT satisfy minimum fresh sample count!
        repeated_obs = [
            {
                "measurement_state": "FRESH_VALID",
                "measurement_validity": True,
                "raw_relative_cavity_score": 0.05,
                "model_identity": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "inference_precision": "fp16_weights",
            }
        ] + [
            {
                "measurement_state": "CACHED_VALID",
                "measurement_validity": True,
                "raw_relative_cavity_score": None,
                "smoothed_relative_score": 0.05,
                "measurement_age_sec": 0.04 * i,
                "model_identity": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "inference_precision": "fp16_weights",
            }
            for i in range(1, 10)
        ]
        stab = compute_track_temporal_stability(repeated_obs, cfg=self.q_cfg)
        self.assertEqual(stab["fresh_valid_count"], 1)
        self.assertEqual(stab["cached_valid_count"], 9)
        self.assertEqual(stab["evidence_sufficiency"], "INSUFFICIENT")
        self.assertIsNone(stab["median_relative_score"])

    def test_temporal_outliers_and_incompatible_configs_not_mixed(self):
        obs = [
            {
                "measurement_state": "FRESH_VALID",
                "measurement_validity": True,
                "raw_relative_cavity_score": 0.050,
                "model_identity": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "inference_precision": "fp16_weights",
                "ref_diagnostics": {"ref_fit_mad": 0.01},
            },
            {
                "measurement_state": "FRESH_VALID",
                "measurement_validity": True,
                "raw_relative_cavity_score": 0.052,
                "model_identity": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "inference_precision": "fp16_weights",
                "ref_diagnostics": {"ref_fit_mad": 0.011},
            },
            {
                "measurement_state": "FRESH_VALID",
                "measurement_validity": True,
                "raw_relative_cavity_score": 0.350,  # outlier spike
                "model_identity": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "inference_precision": "fp16_weights",
                "ref_diagnostics": {"ref_fit_mad": 0.015},
            },
            {
                "measurement_state": "FRESH_VALID",
                "measurement_validity": True,
                "raw_relative_cavity_score": 0.051,
                "model_identity": "depth_anything_v2_small",
                "depth_input_resolution": 336,
                "inference_precision": "fp16_weights",
                "ref_diagnostics": {"ref_fit_mad": 0.010},
            },
            {
                "measurement_state": "FRESH_VALID",
                "measurement_validity": True,
                "raw_relative_cavity_score": 9.99,  # different depth model -> not mixed
                "model_identity": "midas_small",
                "depth_input_resolution": 256,
                "inference_precision": "fp16_autocast",
            },
        ]
        stab = compute_track_temporal_stability(obs, cfg=self.q_cfg)
        self.assertEqual(stab["evidence_sufficiency"], "SUFFICIENT")
        self.assertTrue(stab["incompatible_depth_configs_detected"])
        self.assertAlmostEqual(stab["median_relative_score"], 0.0515, places=4)

    def test_track_aggregation_json_nulls_and_backward_compat(self):
        # Verify explainable rejections preserve codes
        expl = explain_diagnostic_codes(
            ["insufficient_reference_samples", "low_inlier_ratio", "suspected_water_reflection"],
            warnings=["boundary_truncated:bottom"],
        )
        codes = [e["code"] for e in expl]
        self.assertEqual(
            codes,
            [
                "insufficient_reference_samples",
                "low_inlier_ratio",
                "suspected_water_reflection",
                "boundary_truncated:bottom",
            ],
        )

        # Build track summary with mixed fresh valid, invalid, cached
        obs_list = []
        for idx, st in enumerate(["FRESH_VALID", "CACHED_VALID", "FRESH_VALID", "FRESH_INVALID", "FRESH_VALID"]):
            valid = st in ("FRESH_VALID", "CACHED_VALID")
            qa = assess_observation_quality(
                measurement_state=st,
                measurement_validity=valid,
                detector_confidence=0.82,
                ref_diagnostics={"used_ref_samples": 300, "populated_quadrants": 4, "inlier_ratio": 0.8, "ref_fit_mad": 0.015}
                if st == "FRESH_VALID"
                else None,
                cavity_diagnostics={"cavity_sample_count": 150, "positive_frac": 0.85, "iqr_residual": 0.02, "p75_residual": 0.05}
                if st == "FRESH_VALID"
                else None,
                failure_reasons=["low_inlier_ratio"] if st == "FRESH_INVALID" else [],
                measurement_age_sec=0.04 if st == "CACHED_VALID" else 0.0,
                cfg=self.q_cfg,
            )
            obs_list.append(
                {
                    "measurement": {
                        "measurement_state": st,
                        "measurement_validity": valid,
                        "raw_relative_cavity_score": 0.05 + 0.002 * idx if st == "FRESH_VALID" else None,
                        "model_identity": "depth_anything_v2_small",
                        "depth_input_resolution": 336,
                        "inference_precision": "fp16_weights",
                        "failure_reasons": ["low_inlier_ratio"] if st == "FRESH_INVALID" else [],
                        "quality_assessment": qa,
                    }
                }
            )
        ts = build_track_quality_summary(obs_list, cfg=self.q_cfg)
        self.assertEqual(ts["fresh_valid_count"], 3)
        self.assertEqual(ts["fresh_invalid_count"], 1)
        self.assertEqual(ts["cached_measurement_count"], 1)
        self.assertEqual(ts["evidence_sufficiency"], "SUFFICIENT")
        self.assertEqual(ts["dominant_rejection_reasons"][0]["code"], "low_inlier_ratio")

        # Strict JSON null serialization (no NaN)
        dumped = json.dumps(ts, allow_nan=False)
        self.assertIn("median_quality_index", dumped)

        ss = summarize_session_quality(obs_list, [ts])
        self.assertEqual(ss["sufficient_evidence_track_count"], 1)
        json.dumps(ss, allow_nan=False)

        # Backward compatibility: schemas without quality fields still instantiate cleanly
        legacy_m = RelativeDepthMeasurement(
            source_session_id="s1",
            frame_id="f1",
            frame_idx=0,
            timestamp_sec=0.0,
            track_id=1,
            track_generation_id=0,
            box_xyxy=(0.0, 0.0, 10.0, 10.0),
            detector_confidence=0.8,
            model_identity="none",
            depth_input_resolution=None,
            inference_precision=None,
            raw_relative_cavity_score=None,
            normalized_relative_score=None,
            smoothed_relative_score=None,
            measurement_validity=False,
            measurement_state="NOT_EVALUATED",
            measurement_age_sec=None,
            ref_diagnostics=ReferenceFitDiagnostics(),
        )
        self.assertIsNone(legacy_m.quality_assessment)
        legacy_t = TrackObservation(
            source_session_id="s1",
            track_id=1,
            track_generation_id=0,
            first_seen_frame_idx=0,
            last_seen_frame_idx=0,
            observation_count=1,
            box_xyxy=(0.0, 0.0, 10.0, 10.0),
            latest_state="NOT_EVALUATED",
        )
        self.assertIsNone(legacy_t.quality_summary)


if __name__ == "__main__":
    unittest.main()
