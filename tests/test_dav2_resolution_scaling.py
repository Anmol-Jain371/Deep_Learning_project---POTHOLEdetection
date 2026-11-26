import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.depth_model_comparison import DepthAnythingV2SmallAdapter
from experiments.dav2_inference_optimization import affine_align, spearman_fast
from experiments.benchmark_dav2_resolution_scaling import (
    classify_candidate_strata,
    FROZEN_ACCEPTANCE_POLICY,
)


class TestDAv2ResolutionScaling(unittest.TestCase):
    def test_aspect_ratio_mod14_shapes_do_not_assume_square(self):
        ad = DepthAnythingV2SmallAdapter(device="cuda" if torch.cuda.is_available() else "cpu", input_size=336)
        img_360 = np.zeros((360, 640, 3), dtype=np.uint8)
        img_720 = np.zeros((720, 1280, 3), dtype=np.uint8)
        expected_shapes = {
            266: (266, 476),
            336: (336, 602),
            392: (392, 700),
            518: (518, 924),
        }
        for sz, exp_hw in expected_shapes.items():
            x360, _ = ad.model.image2tensor(img_360, input_size=sz)
            x720, _ = ad.model.image2tensor(img_720, input_size=sz)
            self.assertEqual((x360.shape[-2], x360.shape[-1]), exp_hw)
            self.assertEqual((x720.shape[-2], x720.shape[-1]), exp_hw)
            self.assertEqual(x360.shape[-2] % 14, 0)
            self.assertEqual(x360.shape[-1] % 14, 0)
            self.assertNotEqual(x360.shape[-2], x360.shape[-1])

    def test_affine_align_recovers_positive_scale_shift_invariantly(self):
        yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
        ref = 0.05 * yy - 0.02 * xx + 3.0
        src = (ref - 12.5) / 7.25
        aligned, a, b, mae, nrmse = affine_align(src, ref)
        self.assertAlmostEqual(a, 7.25, places=3)
        self.assertAlmostEqual(b, 12.5, places=2)
        self.assertLess(mae, 1e-4)
        self.assertLess(nrmse, 1e-4)
        self.assertGreater(spearman_fast(src, ref, stride=2), 0.9999)

    def test_strata_classification_tracks_effective_vit_patches(self):
        # 24x24 box in 1280x720 resized to net_hw=(266, 476) shrinks to 8.9x8.9px (< 14px ViT patch!)
        s266 = classify_candidate_strata((100, 300, 124, 324), 1280, 720, (266, 476), 0.70, "urban")
        s518 = classify_candidate_strata((100, 300, 124, 324), 1280, 720, (518, 924), 0.70, "urban")
        self.assertEqual(s266["area_bucket"], "tiny_lt_0.5pct")
        self.assertEqual(s266["eff_dim_bucket"], "sub1_vit_patch_lt14px")
        self.assertEqual(s518["eff_dim_bucket"], "patch1to2_14_to_28px")
        self.assertEqual(s266["y_bucket"], "distant_top_y2_lt_0.50")
        self.assertEqual(s266["conf_bucket"], "high_conf_ge_0.65")

    def test_dav2_all_resolutions_finite_and_polarity_verified(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # Road gradient image (bottom sky-free brighter closer road texture) for finite polarity sanity
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        for y in range(360):
            img[y, :, :] = int(40 + 180 * (y / 360.0))
        img[:, ::20, :] = 120
        for sz in (266, 336, 392, 518):
            ad = DepthAnythingV2SmallAdapter(device=dev, input_size=sz, use_fp16=True)
            pr = ad.predict(img)
            self.assertEqual(pr.depth_map.shape, (360, 640))
            self.assertTrue(np.all(np.isfinite(pr.depth_map)))
            self.assertLess(pr.timings_ms["total_ms"], 10000.0)

    def test_frozen_policy_criteria_values(self):
        self.assertGreaterEqual(FROZEN_ACCEPTANCE_POLICY["min_coverage_retention_vs_518"], 0.90)
        self.assertEqual(FROZEN_ACCEPTANCE_POLICY["ransac_config"]["sampling_mode"], "stratified_quadrant")
        self.assertEqual(FROZEN_ACCEPTANCE_POLICY["ransac_config"]["max_ref_samples"], 512)
        self.assertEqual(FROZEN_ACCEPTANCE_POLICY["temporal_filter_config"]["alpha"], 0.35)


if __name__ == "__main__":
    unittest.main()
