import hashlib
import json
import os
import unittest
import numpy as np

from experiments.shadow_depth_pipeline import (
    MEASUREMENT_STATES,
    ShadowDepthPipeline,
    ShadowMeasurementResult,
)
from experiments.optimized_relative_depth import OptimizedDepthConfig
from experiments.validity_aware_temporal_filter import TemporalFilterConfig


def _synth_scene(h=360, w=640, a=0.05, b=-0.02, c=250.0, cavity=12.0, box=(260, 150, 360, 230)):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    plane = a * yy + b * xx + c
    x1, y1, x2, y2 = box
    cy, cx = 0.5 * (y1 + y2), 0.5 * (x1 + x2)
    ry, rx = 0.45 * (y2 - y1), 0.45 * (x2 - x1)
    r2 = ((yy - cy) / max(1.0, ry)) ** 2 + ((xx - cx) / max(1.0, rx)) ** 2
    dmap = (plane - np.where(r2 <= 1.0, cavity * np.maximum(0.0, 1.0 - r2) ** 0.75, 0.0)).astype(np.float32)
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    return img, dmap


class TestShadowDepthPipeline(unittest.TestCase):
    def test_measurement_schema_and_shared_consistency(self):
        pipe = ShadowDepthPipeline(enable_adaptive_variant=True)
        img, dmap = _synth_scene()
        box = (260.0, 150.0, 360.0, 230.0)
        res = pipe.process_frame("f001", 0, 0.0, img, dmap, [box], [0.88], [7])

        for mid in ("A_legacy_annular_ema", "B_opt_ransac_valid_ema", "C_opt_ransac_rollmed_ema", "D_opt_ransac_adaptive_1024"):
            self.assertIn(mid, res["results_by_method"])
            item: ShadowMeasurementResult = res["results_by_method"][mid][0]
            d = item.to_dict()
            # JSON-serializable cleanly without numpy scalar crashes
            json.dumps(d)
            self.assertIn(item.status, MEASUREMENT_STATES)
            self.assertEqual(item.frame_id, "f001")
            self.assertEqual(item.track_id, 7)
            self.assertEqual(item.box_xyxy, list(box))
            self.assertEqual(item.detector_conf, 0.88)
            self.assertEqual(item.camera_profile, "sedan_dashcam")
            self.assertTrue(item.is_valid)
            self.assertEqual(item.status, "VALID")

    def test_invalid_propagation_and_stale_expiration(self):
        flt_cfg = TemporalFilterConfig(alpha=0.35, rolling_window=5, max_stale_frames=3, max_missed_frames_reset=8)
        pipe = ShadowDepthPipeline(filter_cfg=flt_cfg)
        box = (260.0, 150.0, 360.0, 230.0)
        img, d_valid = _synth_scene(cavity=14.0, box=(260, 150, 360, 230))
        _, d_flat = _synth_scene(cavity=0.0, box=(260, 150, 360, 230))

        # Frame 0: valid baseline
        r0 = pipe.process_frame("f0", 0, 0.0, img, d_valid, [box], [0.9], [11])
        b0 = r0["results_by_method"]["B_opt_ransac_valid_ema"][0]
        self.assertEqual(b0.status, "VALID")
        init_score = b0.smoothed_relative_score
        self.assertIsNotNone(init_score)

        # Frames 1..3: invalid flat frames -> status=STALE, numerical estimate NOT updated, raw_relative_p75 is None
        for idx in (1, 2, 3):
            rk = pipe.process_frame(f"f{idx}", idx, idx * 0.1, img, d_flat, [box], [0.9], [11])
            bk = rk["results_by_method"]["B_opt_ransac_valid_ema"][0]
            self.assertFalse(bk.is_valid)
            self.assertEqual(bk.status, "STALE")
            self.assertIsNone(bk.raw_relative_p75)
            self.assertEqual(bk.smoothed_relative_score, init_score)
            self.assertEqual(bk.stale_age_frames, idx)

        # Frame 4: stale age=4 > max_stale_frames=3 -> EXPIRED and smoothed_relative_score=None (never held forever!)
        r4 = pipe.process_frame("f4", 4, 0.4, img, d_flat, [box], [0.9], [11])
        b4 = r4["results_by_method"]["B_opt_ransac_valid_ema"][0]
        self.assertEqual(b4.status, "EXPIRED")
        self.assertIsNone(b4.smoothed_relative_score)
        self.assertEqual(b4.stale_age_frames, 4)

    def test_track_lifecycle_resets_and_id_reuse(self):
        flt_cfg = TemporalFilterConfig(alpha=0.5, rolling_window=3, max_stale_frames=2, max_missed_frames_reset=5)
        pipe = ShadowDepthPipeline(filter_cfg=flt_cfg)
        box = (260.0, 150.0, 360.0, 230.0)
        img, d_low = _synth_scene(cavity=8.0)
        _, d_high = _synth_scene(cavity=26.0)

        pipe.process_frame("f0", 0, 0.0, img, d_low, [box], [0.9], [5])
        pipe.process_frame("f1", 1, 0.1, img, d_low, [box], [0.9], [5])
        # Track missing frames 2..8 (>= max_missed_frames_reset=5); ID=5 reused at frame 9 with high cavity
        for idx in range(2, 9):
            pipe.process_frame(f"f{idx}", idx, idx * 0.1, img, d_low, [], [], [])
        r9 = pipe.process_frame("f9", 9, 0.9, img, d_high, [box], [0.9], [5])
        b9 = r9["results_by_method"]["B_opt_ransac_valid_ema"][0]
        c9 = r9["results_by_method"]["C_opt_ransac_rollmed_ema"][0]
        self.assertEqual(b9.track_age_frames, 0)
        self.assertAlmostEqual(b9.smoothed_relative_score, b9.raw_relative_p75, places=4)
        self.assertAlmostEqual(c9.smoothed_relative_score, c9.raw_relative_p75, places=4)

    def test_missing_depth_and_failure_isolation(self):
        pipe = ShadowDepthPipeline()
        img, dmap = _synth_scene()
        box = (260.0, 150.0, 360.0, 230.0)

        # Case 1: depth_map=None -> B/C emit NOT_EVALUATED without crashing A
        rm = pipe.process_frame("f_none", 0, 0.0, img, None, [box], [0.8], [1])
        self.assertEqual(rm["results_by_method"]["A_legacy_annular_ema"][0].status, "INVALID")
        self.assertEqual(rm["results_by_method"]["B_opt_ransac_valid_ema"][0].status, "NOT_EVALUATED")
        self.assertIsNone(rm["results_by_method"]["B_opt_ransac_valid_ema"][0].smoothed_relative_score)

        # Case 2: Exception inside B_filter leaves Legacy A and C untouched!
        def _boom():
            raise RuntimeError("Simulated filter fault")

        pipe.fault_hooks["B_filter"] = _boom
        rf = pipe.process_frame("f_fault", 1, 0.1, img, dmap, [box], [0.8], [2])
        self.assertEqual(rf["results_by_method"]["A_legacy_annular_ema"][0].status, "VALID")
        self.assertEqual(rf["results_by_method"]["B_opt_ransac_valid_ema"][0].status, "NOT_EVALUATED")
        self.assertIn("RuntimeError", rf["results_by_method"]["B_opt_ransac_valid_ema"][0].error_message)
        self.assertEqual(rf["results_by_method"]["C_opt_ransac_rollmed_ema"][0].status, "VALID")

    def test_deterministic_sampling_and_resolution_coord_mapping(self):
        p1 = ShadowDepthPipeline(opt_cfg=OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512, seed=42))
        p2 = ShadowDepthPipeline(opt_cfg=OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512, seed=42))
        img360, d360 = _synth_scene(h=360, w=640, cavity=14.0, box=(260, 150, 360, 230))
        img720, d720 = _synth_scene(h=720, w=1280, cavity=14.0, box=(520, 300, 720, 460))

        o1 = p1.process_frame("f360", 0, 0.0, img360, d360, [(260, 150, 360, 230)], [0.9], [3])
        o2 = p2.process_frame("f360", 0, 0.0, img360, d360, [(260, 150, 360, 230)], [0.9], [3])
        b1 = o1["results_by_method"]["B_opt_ransac_valid_ema"][0]
        b2 = o2["results_by_method"]["B_opt_ransac_valid_ema"][0]
        self.assertEqual(b1.raw_relative_p75, b2.raw_relative_p75)
        self.assertEqual(b1.ref_mad, b2.ref_mad)

        # Resolution & camera-profile switches inside active track stay bounded & clean
        p1.set_camera_profile("suv_crossover")
        o720 = p1.process_frame("f720", 1, 0.1, img720, d720, [(520, 300, 720, 460)], [0.9], [3])
        b720 = o720["results_by_method"]["B_opt_ransac_valid_ema"][0]
        self.assertEqual(b720.status, "VALID")
        self.assertEqual(b720.camera_profile, "suv_crossover")
        self.assertGreater(b720.raw_relative_p75, 4.0)

    def test_production_file_hashes_preserved(self):
        expected = {
            "roadvision_engine.py": "3805808750134A38B3FA9BCFC42FB6583A843DDA575E70AEF28FF205C0E2A51E",
            "server.py": "D8772504E03F949E1B92CF8C76DA65327035F2B1A5AE713385D5EB60FDE91EFF",
            "config/camera_params.json": "0C7633F330A8B5C15A1696B85D81E9CA6D32E3CAC1C1C237CFFF8E576E789AF1",
        }
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for rel, exp_hash in expected.items():
            with open(os.path.join(root, rel), "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest().upper()
            self.assertEqual(h, exp_hash, f"Production file modified unexpectedly: {rel}")


if __name__ == "__main__":
    unittest.main()
