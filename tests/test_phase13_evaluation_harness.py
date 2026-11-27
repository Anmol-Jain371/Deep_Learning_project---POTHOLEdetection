"""
tests/test_phase13_evaluation_harness.py — Lightweight CPU tests for Phase 13
scientific evaluation utilities (statistical guards, geometry synthetic ablation,
quality variants, and failure-case reporting).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.phase13.geometry_ablation import run_geometry_ablation
from experiments.phase13.quality_geo_maint_eval import run_quality_geo_maint_and_negatives
from experiments.phase13.statistics import group_level_bootstrap_ci, paired_difference_summary, summarize_array


class TestPhase13EvaluationHarness(unittest.TestCase):
    def test_statistics_refuses_pseudoreplicated_ci_on_small_groups(self):
        small_ci = group_level_bootstrap_ci([0.4, 0.5, 0.45], min_groups_for_ci=5)
        self.assertIsNone(small_ci["ci95_low"])
        self.assertIsNone(small_ci["ci95_high"])

        ok_ci = group_level_bootstrap_ci([0.4, 0.5, 0.45, 0.48, 0.42, 0.46], min_groups_for_ci=5)
        self.assertIsNotNone(ok_ci["ci95_low"])
        self.assertLessEqual(ok_ci["ci95_low"], ok_ci["ci95_high"])

        summ = summarize_array([1.0, 2.0, 3.0, None])
        self.assertEqual(summ["n"], 3)
        self.assertAlmostEqual(summ["median"], 2.0)

        pdiff = paired_difference_summary([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        self.assertEqual(pdiff["n_pairs"], 3)
        self.assertAlmostEqual(pdiff["mean_diff"], 0.5)

    def test_geometry_synthetic_ablation_and_phase10_fidelity(self):
        res = run_geometry_ablation(smoke=True, seed=42)
        self.assertEqual(res["status"], "COMPLETED")
        ref_ab = res["reference_plane_ablation"]
        self.assertGreater(ref_ab["D_optimized_msac_ransac"]["gate_concordance_pct"], 85.0)
        self.assertLess(
            ref_ab["D_optimized_msac_ransac"]["outlier_mean_plane_rmse"],
            ref_ab["B_ols_plane"]["outlier_mean_plane_rmse"],
        )
        p10 = res["phase10_geometry_evaluation"]
        self.assertGreater(p10["reference_mask_true_cells"], 0)
        self.assertGreater(p10["null_cells_preserved"], 0)
        self.assertTrue(p10["cross_section_gap_preserved"])
        self.assertEqual(p10["extra_depth_inference_calls"], 0)

    def test_quality_variants_and_failure_registry(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with tempfile.TemporaryDirectory() as tmp:
            res = run_quality_geo_maint_and_negatives(project_root, tmp)
            self.assertEqual(res["status"], "COMPLETED")
            self.assertTrue(os.path.isfile(os.path.join(tmp, "failure_cases.csv")))
            probes = {p["case"]: p for p in res["phase9_quality_ablation"]["synthetic_probes"]}
            self.assertEqual(probes["invalid_high_det_confidence"]["C_deployed"], "INVALID")
            self.assertEqual(probes["expired_cached_observation"]["C_deployed"], "INVALID")
            self.assertEqual(probes["low_det_conf_boundary_clipped_missing_temporal"]["C_deployed"], "HIGH_QUALITY")
            self.assertEqual(probes["low_det_conf_boundary_clipped_missing_temporal"]["E_conservative_gated"], "MODERATE_QUALITY")


if __name__ == "__main__":
    unittest.main()
