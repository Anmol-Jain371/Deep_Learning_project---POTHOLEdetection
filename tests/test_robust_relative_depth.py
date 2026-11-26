"""
Automated unit tests for Phase 5A robust relative depth extraction, neighbor exclusion,
plane fitting invariances, quality failure detection, and non-fabrication of invalid depths.
"""

import os
import sys
import unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.robust_relative_depth import (
    ALL_METHODS,
    RobustDepthConfig,
    build_adaptive_annular_region,
    extract_relative_depth_for_box,
)


class TestRobustRelativeDepthUnit(unittest.TestCase):
    def _make_tilted_plane_with_cavity(self, h=240, w=320, a=0.04, b=-0.02, c=10.0, cav_depth=2.5):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        plane = a * yy + b * xx + c
        depth = plane.copy()
        # Elliptical depression centered at (150, 160) inside box (120, 110, 200, 190)
        mask = (((yy - 150.0) / 28.0) ** 2 + ((xx - 160.0) / 30.0) ** 2) <= 1.0
        depth[mask] -= cav_depth
        return depth, plane, mask

    def test_inclined_plane_recovery_and_polarity(self):
        depth, plane, _ = self._make_tilted_plane_with_cavity(a=0.08, b=0.03, cav_depth=3.0)
        box = (120, 110, 200, 190)
        for m in ("ols_plane", "ransac_plane", "huber_irls_plane"):
            out = extract_relative_depth_for_box(depth, box, method=m)
            self.assertTrue(out["is_valid"], f"{m} should be valid: {out['failure_reasons']}")
            self.assertAlmostEqual(out["relative_cavity_stats"]["p75_residual"], 3.0, delta=0.15)
            self.assertGreater(out["relative_cavity_stats"]["positive_depression_frac"], 0.38)

        # Flat median baseline suffers severe plane-fit RMSE on inclined road surface
        med_out = extract_relative_depth_for_box(depth, box, method="median_baseline")
        ols_out = extract_relative_depth_for_box(depth, box, method="ols_plane")
        self.assertGreater(med_out["ref_diagnostics"]["ref_fit_rmse"], 1.5)
        self.assertLess(ols_out["ref_diagnostics"]["ref_fit_rmse"], 1e-6)

    def test_neighbor_box_excluded_from_annular_ring(self):
        depth, _, _ = self._make_tilted_plane_with_cavity(cav_depth=2.0)
        # Inject deep neighboring cavity adjacent to target box
        depth[110:190, 208:255] -= 15.0
        target = (120, 110, 200, 190)
        neighbor = (208, 110, 255, 190)

        out_clean = extract_relative_depth_for_box(depth, target, method="ols_plane", neighbor_boxes=[neighbor])
        self.assertGreater(out_clean["ref_diagnostics"]["neighbor_excluded_px"], 100)
        self.assertAlmostEqual(out_clean["relative_cavity_stats"]["p75_residual"], 2.0, delta=0.2)

    def test_outlier_resistance_huber_and_ransac_vs_ols(self):
        depth, _, _ = self._make_tilted_plane_with_cavity(a=0.05, b=-0.02, cav_depth=2.0)
        # Corrupt 18% of outer annular top strip with huge closer-object spike (+20.0)
        depth[85:105, 100:220] += 20.0
        box = (120, 110, 200, 190)

        out_ols = extract_relative_depth_for_box(depth, box, method="ols_plane")
        out_ran = extract_relative_depth_for_box(depth, box, method="ransac_plane")
        out_hub = extract_relative_depth_for_box(depth, box, method="huber_irls_plane")

        err_ols = abs(out_ols["relative_cavity_stats"]["raw_diagnostic_p75"] - 2.0)
        err_ran = abs(out_ran["relative_cavity_stats"]["p75_residual"] - 2.0)
        err_hub = abs(out_hub["relative_cavity_stats"]["p75_residual"] - 2.0)
        self.assertTrue(out_ran["is_valid"])
        self.assertTrue(out_hub["is_valid"])
        self.assertLess(err_ran, 0.25)
        self.assertLess(err_hub, 0.25)
        self.assertGreater(err_ols, 1.5)

    def test_invalid_cases_stay_invalid_never_zero_depth(self):
        # 1. Flat road without depression -> unexpected_residual_polarity
        yy, xx = np.mgrid[0:240, 0:320].astype(np.float64)
        flat = 0.03 * yy + 5.0
        out_flat = extract_relative_depth_for_box(flat, (120, 110, 200, 190), method="huber_irls_plane")
        self.assertFalse(out_flat["is_valid"])
        self.assertIn("unexpected_residual_polarity", out_flat["failure_reasons"])
        self.assertIsNone(out_flat["relative_cavity_stats"]["p75_residual"])

        # 2. Non-finite depth map -> invalid_or_nonfinite_depth
        bad = flat.copy()
        bad[120:160, 140:180] = np.nan
        out_nan = extract_relative_depth_for_box(bad, (120, 110, 200, 190), method="huber_irls_plane")
        self.assertFalse(out_nan["is_valid"])
        self.assertIn("invalid_or_nonfinite_depth", out_nan["failure_reasons"])
        self.assertIsNone(out_nan["relative_cavity_stats"]["median_residual"])

        # 3. Specular water patch -> suspected_water_reflection
        depth, _, _ = self._make_tilted_plane_with_cavity(cav_depth=2.5)
        fake_bgr = np.full((240, 320, 3), 80, dtype=np.uint8)
        # Smooth bright glare inside box triggers SpecularAquaplaneDetector (V>185, S<45, low Laplacian var)
        fake_bgr[120:180, 125:195] = (235, 235, 235)
        fake_bgr = cv2.GaussianBlur(fake_bgr, (25, 25), 7.0)
        out_wat = extract_relative_depth_for_box(depth, (120, 110, 200, 190), method="huber_irls_plane", img_bgr=fake_bgr)
        self.assertFalse(out_wat["is_valid"])
        self.assertTrue(out_wat["is_water_suspected"])
        self.assertIn("suspected_water_reflection", out_wat["failure_reasons"])
        self.assertIsNone(out_wat["relative_cavity_stats"]["p75_residual"])

    def test_positive_affine_scale_and_shift_invariance(self):
        depth, _, _ = self._make_tilted_plane_with_cavity(cav_depth=2.2)
        box = (120, 110, 200, 190)
        scaled_depth = 3.75 * depth + 42.0

        o1 = extract_relative_depth_for_box(depth, box, method="huber_irls_plane")
        o2 = extract_relative_depth_for_box(scaled_depth, box, method="huber_irls_plane")
        self.assertAlmostEqual(o2["relative_cavity_stats"]["p75_residual"], 3.75 * o1["relative_cavity_stats"]["p75_residual"], delta=0.05)
        # Scale-normalized P75 is invariant (within tiny floor effects) when noise > 1e-4
        depth_noisy = depth + np.random.default_rng(0).normal(0.0, 0.05, depth.shape)
        scaled_noisy = 3.75 * depth_noisy + 42.0
        n1 = extract_relative_depth_for_box(depth_noisy, box, method="huber_irls_plane")
        n2 = extract_relative_depth_for_box(scaled_noisy, box, method="huber_irls_plane")
        self.assertAlmostEqual(n1["relative_cavity_stats"]["scale_normalized_p75"], n2["relative_cavity_stats"]["scale_normalized_p75"], delta=0.25)


if __name__ == "__main__":
    unittest.main()
