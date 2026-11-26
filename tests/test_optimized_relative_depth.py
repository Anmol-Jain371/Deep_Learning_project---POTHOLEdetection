import unittest
import numpy as np

from experiments.robust_relative_depth import RobustDepthConfig, extract_relative_depth_for_box
from experiments.optimized_relative_depth import (
    SAMPLING_MODES,
    OptimizedDepthConfig,
    build_and_sample_annular_region,
    extract_optimized_relative_depth_for_box,
)


def _synth_plane(h=240, w=320, a=0.06, b=-0.03, c=15.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return a * yy + b * xx + c


def _synth_cavity(h=240, w=320, cy=150.0, cx=160.0, ry=24.0, rx=28.0, depth=3.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return np.where(r2 <= 1.0, depth * np.maximum(0.0, 1.0 - r2) ** 0.75, 0.0)


class TestOptimizedRelativeDepth(unittest.TestCase):
    def test_parity_with_baseline_and_curb_immunity(self):
        plane = _synth_plane(240, 320, a=0.06, b=-0.03, c=15.0)
        depth = plane - _synth_cavity(240, 320, depth=3.5)
        # Inject strong unannotated curb block covering top annular strip
        depth[72:108, 82:238] += 14.0
        box = (120, 110, 200, 190)

        base_out = extract_relative_depth_for_box(depth, box, method="ransac_plane", cfg=RobustDepthConfig())
        opt_out = extract_optimized_relative_depth_for_box(
            depth, box, cfg=OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
        )

        self.assertTrue(base_out["is_valid"])
        self.assertTrue(opt_out["is_valid"])
        self.assertAlmostEqual(
            opt_out["relative_cavity_stats"]["p75_residual"],
            base_out["relative_cavity_stats"]["p75_residual"],
            places=3,
        )
        self.assertLess(opt_out["ref_diagnostics"]["ref_fit_mad"], 1e-4)

    def test_sampling_modes_preserve_spatial_support(self):
        plane = _synth_plane(240, 320)
        # Wide cavity inside wide aspect box (Top/Bottom pixels >> Left/Right pixels) -> test stratification preserves all 4 sides!
        depth = plane - _synth_cavity(240, 320, cy=145.0, cx=160.0, ry=18.0, rx=75.0, depth=3.0)
        box = (70, 125, 250, 165)
        for mode in SAMPLING_MODES:
            cfg = OptimizedDepthConfig(sampling_mode=mode, max_ref_samples=256)
            reg = build_and_sample_annular_region(depth, box, cfg=cfg)
            out = extract_optimized_relative_depth_for_box(depth, box, cfg=cfg)
            self.assertTrue(out["is_valid"], f"Mode {mode} unexpectedly invalid: {out['failure_reasons']}")
            if mode != "full":
                self.assertLessEqual(reg["sampled_ref_samples"], 256)
            self.assertGreaterEqual(reg["populated_quadrants"], 2)
            if mode == "stratified_quadrant":
                # Every populated full-ring quadrant retains samples under budget=256
                for fq, sq in zip(reg["full_quadrant_counts"], reg["quadrant_counts"]):
                    if fq >= cfg.min_quadrant_samples:
                        self.assertGreaterEqual(sq, cfg.min_quadrant_samples)

    def test_early_stopping_on_clean_surface_vs_noisy(self):
        plane = _synth_plane(240, 320)
        clean_depth = plane - _synth_cavity(240, 320, depth=3.0)
        box = (120, 110, 200, 190)

        out_clean = extract_optimized_relative_depth_for_box(
            clean_depth, box, cfg=OptimizedDepthConfig(max_ref_samples=512, ransac_max_iters=128, ransac_min_iters=16, ransac_batch_size=16)
        )
        self.assertTrue(out_clean["ref_diagnostics"]["early_stopped"])
        self.assertEqual(out_clean["ref_diagnostics"]["iters_evaluated"], 16)

        # With 45% heavy scattered outliers (w ~ 0.55 => N_req ~ 26 > 16), RANSAC adaptively expands beyond 16
        rng = np.random.default_rng(12)
        noisy = clean_depth.copy()
        m = np.zeros_like(noisy, dtype=bool)
        m[74:226, 84:236] = True
        m[106:194, 116:204] = False
        oy, ox = np.where(m)
        pick = rng.choice(len(oy), size=int(0.45 * len(oy)), replace=False)
        noisy[oy[pick], ox[pick]] += 15.0
        out_noisy = extract_optimized_relative_depth_for_box(
            noisy, box, cfg=OptimizedDepthConfig(max_ref_samples=512, ransac_max_iters=128, ransac_min_iters=16, ransac_batch_size=16)
        )
        self.assertTrue(out_noisy["is_valid"])
        self.assertGreater(out_noisy["ref_diagnostics"]["iters_evaluated"], 16)
        self.assertAlmostEqual(out_noisy["relative_cavity_stats"]["p75_residual"], out_clean["relative_cavity_stats"]["p75_residual"], places=3)

    def test_invalid_cases_stay_invalid_never_zero_depth(self):
        # 1. Flat surface without depression
        flat = _synth_plane(240, 320, a=0.0, b=0.0, c=10.0)
        out_flat = extract_optimized_relative_depth_for_box(flat, (120, 110, 200, 190))
        self.assertFalse(out_flat["is_valid"])
        self.assertIn("unexpected_residual_polarity", out_flat["failure_reasons"])
        self.assertIsNone(out_flat["relative_cavity_stats"]["p75_residual"])

        # 2. One-sided/insufficient spatial support & tiny sample count
        nan_sided = flat.copy()
        nan_sided[:, :200] = np.nan
        nan_sided -= _synth_cavity(240, 320, depth=3.0)
        out_nan = extract_optimized_relative_depth_for_box(nan_sided, (195, 110, 225, 190))
        self.assertFalse(out_nan["is_valid"])
        self.assertIsNone(out_nan["relative_cavity_stats"]["p75_residual"])

    def test_positive_affine_scale_and_shift_invariance(self):
        plane = _synth_plane(240, 320, a=0.04, b=-0.02, c=8.0)
        d1 = plane - _synth_cavity(240, 320, depth=3.0)
        rng = np.random.default_rng(7)
        d1 += rng.normal(0.0, 0.05, size=d1.shape)
        d2 = 4.5 * d1 + 75.0
        box = (120, 110, 200, 190)

        o1 = extract_optimized_relative_depth_for_box(d1, box)
        o2 = extract_optimized_relative_depth_for_box(d2, box)
        self.assertTrue(o1["is_valid"] and o2["is_valid"])
        self.assertAlmostEqual(o2["relative_cavity_stats"]["p75_residual"], 4.5 * o1["relative_cavity_stats"]["p75_residual"], places=2)
        self.assertAlmostEqual(o2["relative_cavity_stats"]["scale_normalized_p75"], o1["relative_cavity_stats"]["scale_normalized_p75"], places=1)


if __name__ == "__main__":
    unittest.main()
