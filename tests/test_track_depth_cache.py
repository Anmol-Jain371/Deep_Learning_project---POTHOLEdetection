import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.track_depth_cache import (
    CACHE_STATES,
    TrackCacheConfig,
    TrackDepthCache,
)


class TestTrackDepthCache(unittest.TestCase):
    def test_states_fresh_cached_expired_and_no_zero_fabrication(self):
        cache = TrackDepthCache(
            video_session_id="sess_1",
            depth_model_id="dav2_small",
            depth_input_size=336,
            inference_precision="fp16_weights",
            cfg=TrackCacheConfig(max_cache_age_sec=0.30, max_invalid_stale_sec=0.20, max_absence_reset_sec=0.60),
        )
        bx = (100.0, 150.0, 200.0, 220.0)
        cache.observe_detection(1, 0, 0.00, bx)
        # Before evaluation -> NOT_EVALUATED and None
        st0 = cache.read_non_keyframe_state(1, 0, 0.00)
        self.assertEqual(st0["state"], "NOT_EVALUATED")
        self.assertIsNone(st0["relative_score"])

        # Record FRESH_VALID
        f1 = cache.record_fresh_measurement(1, "f0", 0, 0.00, bx, True, 2.5, 0.04, 0.12, {"inlier_ratio": 0.90})
        self.assertEqual(f1["state"], "FRESH_VALID")
        self.assertTrue(f1["is_fresh"])
        self.assertFalse(f1["is_cached"])
        self.assertEqual(f1["fresh_valid_count"], 1)
        ema0 = f1["ema_fixed_score"]

        # Read CACHED_VALID at dt=0.10s -> does NOT increment fresh_valid_count or update EMA!
        c1 = cache.read_non_keyframe_state(1, 2, 0.10)
        self.assertEqual(c1["state"], "CACHED_VALID")
        self.assertFalse(c1["is_fresh"])
        self.assertTrue(c1["is_cached"])
        self.assertAlmostEqual(c1["cache_age_sec"], 0.10, places=4)
        self.assertEqual(c1["fresh_valid_count"], 1)
        self.assertEqual(c1["cached_read_count"], 1)
        self.assertEqual(c1["ema_fixed_score"], ema0)

        # Read again at dt=0.35s (> max_cache_age_sec=0.30s) -> EXPIRED & None (never 0.0!)
        exp1 = cache.read_non_keyframe_state(1, 9, 0.35)
        self.assertEqual(exp1["state"], "EXPIRED")
        self.assertTrue(exp1["is_expired"])
        self.assertIsNone(exp1["relative_score"])
        self.assertIsNone(exp1["ema_fixed_score"])

    def test_cached_observations_never_pollute_temporal_ema_even_repeated_reads(self):
        cache = TrackDepthCache("s", "dav2_small", 336, cfg=TrackCacheConfig(max_cache_age_sec=1.0, fixed_alpha=0.5, tau_sec=0.20))
        bx = (10.0, 10.0, 60.0, 60.0)
        cache.observe_detection(5, 0, 0.0, bx)
        r0 = cache.record_fresh_measurement(5, "f0", 0, 0.0, bx, True, 10.0, 0.1, 0.2, {"inlier_ratio": 0.9})
        # 20 repeated cache reads MUST NOT shift EMA toward 10 on subsequent 20.0 fresh update!
        for k in range(1, 21):
            c = cache.read_non_keyframe_state(5, k, 0.01 * k)
            self.assertEqual(c["ema_fixed_score"], 10.0)
            self.assertEqual(c["fresh_valid_count"], 1)

        r2 = cache.record_fresh_measurement(5, "f21", 21, 0.21, bx, True, 20.0, 0.2, 0.4, {"inlier_ratio": 0.9})
        self.assertAlmostEqual(r2["ema_fixed_score"], 15.0, places=4)
        self.assertEqual(r2["fresh_valid_count"], 2)

        # FRESH_INVALID also never updates EMA numerical state
        r3 = cache.record_fresh_measurement(5, "f22", 22, 0.25, bx, False, None, None, None, {"inlier_ratio": 0.2}, ("low_inlier_ratio",))
        self.assertEqual(r3["state"], "FRESH_INVALID")
        self.assertIsNone(r3["ema_fixed_score"])
        # Internal entry keeps EMA intact at 15.0 rather than 0.0
        self.assertAlmostEqual(cache.get_entry(5).ema_fixed_score, 15.0, places=4)

    def test_timestamp_expiration_handles_10fps_vs_25fps_consistently(self):
        cfg = TrackCacheConfig(max_cache_age_sec=0.35)
        c25 = TrackDepthCache("v25", "dav2", 336, cfg=cfg)
        c10 = TrackDepthCache("v10", "dav2", 336, cfg=cfg)
        bx = (20, 20, 80, 80)
        c25.observe_detection(1, 0, 0.0, bx); c25.record_fresh_measurement(1, "f0", 0, 0.0, bx, True, 3.0, 0.1, 0.1, {})
        c10.observe_detection(1, 0, 0.0, bx); c10.record_fresh_measurement(1, "f0", 0, 0.0, bx, True, 3.0, 0.1, 0.1, {})

        # At t=0.30s -> 7.5 frames at 25 FPS, 3 frames at 10 FPS -> BOTH still CACHED_VALID
        self.assertEqual(c25.read_non_keyframe_state(1, 7, 0.30)["state"], "CACHED_VALID")
        self.assertEqual(c10.read_non_keyframe_state(1, 3, 0.30)["state"], "CACHED_VALID")
        # At t=0.40s -> BOTH EXPIRED regardless of frame index (10 vs 4)!
        self.assertEqual(c25.read_non_keyframe_state(1, 10, 0.40)["state"], "EXPIRED")
        self.assertEqual(c10.read_non_keyframe_state(1, 4, 0.40)["state"], "EXPIRED")

    def test_track_id_reuse_gap_and_spatial_jump_reset_generation(self):
        cache = TrackDepthCache("s", "dav2", 336, cfg=TrackCacheConfig(max_absence_reset_sec=0.50, max_reuse_centroid_norm=1.5))
        bx1 = (50.0, 200.0, 120.0, 260.0)
        e1, n1, _ = cache.observe_detection(7, 0, 0.0, bx1)
        self.assertTrue(n1)
        cache.record_fresh_measurement(7, "f0", 0, 0.0, bx1, True, 4.2, 0.1, 0.2, {})
        g1 = e1.track_generation_id

        # Case A: Disappearance > max_absence_reset_sec -> new generation, cannot read old cache!
        e2, n2, r2 = cache.observe_detection(7, 20, 0.80, bx1)
        self.assertTrue(n2)
        self.assertEqual(r2, "TRACK_ID_REUSED_AFTER_GAP")
        self.assertGreater(e2.track_generation_id, g1)
        self.assertEqual(cache.read_non_keyframe_state(7, 20, 0.80)["state"], "NOT_EVALUATED")

        # Record on gen 2 then simulate sudden spatial jump after 0.25s with zero IoU
        cache.record_fresh_measurement(7, "f20", 20, 0.80, bx1, True, 9.0, 0.2, 0.3, {})
        bx_far = (450.0, 120.0, 520.0, 180.0)
        e3, n3, r3 = cache.observe_detection(7, 26, 1.05, bx_far)
        self.assertTrue(n3)
        self.assertEqual(r3, "TRACK_ID_REUSED_SPATIAL_JUMP")
        self.assertIsNone(e3.last_valid_relative_score)


if __name__ == "__main__":
    unittest.main()
