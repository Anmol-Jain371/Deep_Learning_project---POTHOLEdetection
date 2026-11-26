import unittest
import numpy as np

from experiments.core_aware_cavity_extraction import (
    CAVITY_STRATEGIES,
    CoreExtractionConfig,
    extract_cavity_with_strategy,
)


def _plane(h=240, w=320, a=0.05, b=-0.02, c=15.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return a * yy + b * xx + c


def _ellipse(h=240, w=320, cy=150.0, cx=160.0, ry=24.0, rx=28.0, depth=3.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return np.where(r2 <= 1.0, depth * np.maximum(0.0, 1.0 - r2) ** 0.75, 0.0)


class TestCoreAwareCavityExtraction(unittest.TestCase):
    def test_synthetic_12_scenarios_frozen_criteria(self):
        h, w = 240, 320
        std_box = (115, 105, 205, 195)
        p = _plane(h, w)
        rng = np.random.default_rng(123)

        # Frozen Task 4 scenario list: (name, depth_map, box, expect_valid_geom, core_strategy)
        scenarios = [
            ("centered", p - _ellipse(h, w, 150, 160, 26, 30, 3.5), std_box, True),
            ("off_center", p - _ellipse(h, w, 138, 142, 20, 22, 3.5), std_box, True),
            ("irregular", p - np.maximum(_ellipse(h, w, 142, 148, 18, 22, 3.5), _ellipse(h, w, 162, 174, 16, 20, 3.0)), std_box, True),
            ("multiple", p - _ellipse(h, w, 132, 138, 19, 19, 3.2) - _ellipse(h, w, 168, 180, 19, 19, 3.0), std_box, True),
            ("flat_road", p.copy(), std_box, False),
            ("raised_bump", p + _ellipse(h, w, 150, 160, 26, 30, 3.5), std_box, False),
            ("shadow_step", p - _ellipse(h, w, 150, 160, 25, 28, 3.5) + np.where(np.arange(w)[None, :] < 128, 0.6, 0.0), std_box, True),
            ("ref_bias_curb", (p - _ellipse(h, w, 150, 160, 26, 30, 3.5)) + np.where((np.arange(h)[:, None] >= 65) & (np.arange(h)[:, None] <= 98), 10.0, 0.0), std_box, True),
            ("contaminated_bbox_edges", (p - _ellipse(h, w, 150, 160, 20, 24, 3.5)) + np.where((np.abs(np.arange(h)[:, None] - 150) >= 28) | (np.abs(np.arange(w)[None, :] - 160) >= 28), 2.8, 0.0) * ((np.arange(h)[:, None] >= 105) & (np.arange(h)[:, None] <= 195) & (np.arange(w)[None, :] >= 115) & (np.arange(w)[None, :] <= 205)), std_box, True),
            ("severe_noise", p - _ellipse(h, w, 150, 160, 26, 30, 3.5) + rng.normal(0.0, 0.45, size=(h, w)), std_box, True),
            ("weak_depression", p - _ellipse(h, w, 150, 160, 26, 30, 0.6) + rng.normal(0.0, 0.02, size=(h, w)), std_box, True),
            ("partial_clipped", p - _ellipse(h, w, 218, 284, 20, 24, 3.5), (254, 194, 319, 239), True),
        ]

        # Test rect_trim_20 & elliptical_core on all 12 + compare contaminated_bbox_edges vs full_box!
        for name, dmap, bx, exp_valid in scenarios:
            strat = "rect_trim_20" if name == "contaminated_bbox_edges" else "elliptical_core"
            out_core = extract_cavity_with_strategy(dmap, bx, strategy=strat)
            self.assertEqual(out_core["is_valid"], exp_valid, f"Failed scenario {name}: reasons={out_core['failure_reasons']}")
            self.assertFalse(out_core["selection_biased"])
            self.assertIn("full_box_preserved_stats", out_core)

            if name == "contaminated_bbox_edges":
                out_full = extract_cavity_with_strategy(dmap, bx, strategy="full_box")
                self.assertFalse(out_full["is_valid"])
                self.assertTrue(out_core["is_valid"])

    def test_off_center_sensitivity_and_multiscale_instability(self):
        p = _plane(240, 320)
        # Put off-center band along upper interior (cy=122, cx=160, ry=16, rx=45 -> >26% inside full_box, almost absent in trim_30!)
        edge_cav = p - _ellipse(240, 320, cy=122.0, cx=160.0, ry=16.0, rx=45.0, depth=4.0)
        out_full = extract_cavity_with_strategy(edge_cav, (115, 105, 205, 195), strategy="full_box")
        out_t30 = extract_cavity_with_strategy(edge_cav, (115, 105, 205, 195), strategy="rect_trim_30")
        out_ms = extract_cavity_with_strategy(edge_cav, (115, 105, 205, 195), strategy="multi_scale_agreement")
        # Excessive 30% trim misses off-center cavity; multi_scale_agreement catches instability!
        self.assertGreater(out_full["core_cavity_stats"]["raw_diag_p75"], 0.1)
        self.assertLess(out_t30["core_cavity_stats"]["raw_diag_p75"], 0.05)
        self.assertIn("unstable_multiscale_interior", out_ms["failure_reasons"])

    def test_selection_biased_flag_and_min_interior_px(self):
        p = _plane(240, 320) - _ellipse(240, 320, 150, 160, 22, 24, 3.0)
        out_e = extract_cavity_with_strategy(p, (115, 105, 205, 195), strategy="spatially_coherent_residual")
        self.assertTrue(out_e["selection_biased"])
        self.assertTrue(out_e["is_valid"])

        # Tiny 7x7 box centered on clean small depression -> 30% trim has 3x3=9 px < min_interior_px=16
        p_tiny = _plane(240, 320) - _ellipse(240, 320, 150, 150, 3.0, 3.0, 3.0)
        out_tiny = extract_cavity_with_strategy(p_tiny, (147, 147, 154, 154), strategy="rect_trim_30")
        self.assertFalse(out_tiny["is_valid"])
        self.assertIn("insufficient_or_nonfinite_interior", out_tiny["failure_reasons"])
        self.assertIsNone(out_tiny["core_cavity_stats"]["p75_residual"])

    def test_scale_invariant_quality_gating_across_model_ranges(self):
        # Verify identical quality verdict at DAv2 magnitude (0.02x) vs MiDaS magnitude (20x) under realistic noise
        rng = np.random.default_rng(7)
        p = _plane(240, 320) - _ellipse(240, 320, 150, 160, 26, 30, 3.0) + rng.normal(0.0, 0.05, size=(240, 320))
        o_small = extract_cavity_with_strategy(0.02 * p, (115, 105, 205, 195), strategy="elliptical_core")
        o_large = extract_cavity_with_strategy(20.0 * p, (115, 105, 205, 195), strategy="elliptical_core")
        self.assertTrue(o_small["is_valid"] and o_large["is_valid"])
        self.assertAlmostEqual(o_small["core_cavity_stats"]["scale_normalized_p75"], o_large["core_cavity_stats"]["scale_normalized_p75"], places=1)


if __name__ == "__main__":
    unittest.main()
