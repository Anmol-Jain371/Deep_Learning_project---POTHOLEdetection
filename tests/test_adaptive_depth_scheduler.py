import os
import sys
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.adaptive_depth_scheduler import (
    AdaptiveDepthScheduler,
    AdaptiveSchedulerConfig,
    BoundedAsyncDepthWorker,
)
from experiments.track_depth_cache import TrackDepthCache, TrackCacheConfig
from tests.test_shadow_depth_pipeline import TestShadowDepthPipeline


class TestAdaptiveDepthScheduler(unittest.TestCase):
    def test_no_candidates_skip_new_track_and_shared_frame_inference(self):
        sched = AdaptiveDepthScheduler(AdaptiveSchedulerConfig(strategy="D_track_aware"))
        # 0 candidates -> skipped with NO_CANDIDATES
        d0 = sched.decide_frame(0, 0.00, [], [])
        self.assertFalse(d0["run_depth"])
        self.assertEqual(d0["frame_reasons"], ["NO_CANDIDATES"])

        # Two tracks appear on frame 1 -> single shared depth call serving both!
        boxes = [(100, 150, 180, 210), (300, 180, 390, 240)]
        d1 = sched.decide_frame(1, 0.04, boxes, [1, 2])
        self.assertTrue(d1["run_depth"])
        self.assertEqual(d1["shared_across_tracks"], 2)
        self.assertEqual(sched.depth_calls, 1)
        for tid, bx in zip([1, 2], boxes):
            sched.cache.record_fresh_measurement(tid, "f1", 1, 0.04, bx, True, 2.0, 0.05, 0.1, {"inlier_ratio": 0.9})

        # Steady boxes at dt=0.04s -> skipped (CACHED_HOLD)
        d2 = sched.decide_frame(2, 0.08, [(102, 151, 182, 211), (301, 180, 391, 240)], [1, 2])
        self.assertFalse(d2["run_depth"])
        self.assertEqual(d2["frame_reasons"], ["CACHED_HOLD"])

    def test_fixed_interval_scheduling_k3(self):
        sched = AdaptiveDepthScheduler(AdaptiveSchedulerConfig(strategy="C_fixed_interval", fixed_interval_k=3))
        bx = [(50, 50, 120, 110)]
        runs = [sched.decide_frame(i, i * 0.04, bx, [10])["run_depth"] for i in range(7)]
        self.assertEqual(runs, [True, False, False, True, False, False, True])

    def test_geometry_change_expiration_and_invalid_retry(self):
        cfg = AdaptiveSchedulerConfig(
            strategy="D_track_aware",
            area_change_threshold=0.30,
            centroid_shift_threshold=0.35,
            min_box_iou=0.50,
            refresh_interval_sec=0.30,
            max_measurement_age_sec=0.45,
            invalid_retry_base_sec=0.10,
        )
        sched = AdaptiveDepthScheduler(cfg, TrackDepthCache("s", "dav2", 336, cfg=TrackCacheConfig(max_cache_age_sec=0.45)))
        bx = (100, 100, 200, 180)
        sched.decide_frame(0, 0.00, [bx], [1])
        sched.cache.record_fresh_measurement(1, "f0", 0, 0.00, bx, True, 2.0, 0.05, 0.1, {"inlier_ratio": 0.9})

        # Big area jump (>=30%) triggers BOX_AREA_CHANGE immediately
        bx_big = (85, 85, 225, 205)
        da = sched.decide_frame(1, 0.04, [bx_big], [1])
        self.assertTrue(da["run_depth"])
        self.assertIn("BOX_AREA_CHANGE", da["frame_reasons"])
        sched.cache.record_fresh_measurement(1, "f1", 1, 0.04, bx_big, True, 2.2, 0.06, 0.1, {"inlier_ratio": 0.9})

        # Low IoU / shift without huge area change triggers LOW_BOX_OVERLAP
        bx_shift = (180, 85, 320, 205)
        ds = sched.decide_frame(2, 0.08, [bx_shift], [1])
        self.assertTrue(ds["run_depth"])
        self.assertIn("LOW_BOX_OVERLAP", ds["frame_reasons"])

        # Record FRESH_INVALID -> immediate dt=0.04s < cooldown=0.10s skips; dt=0.16s triggers INVALID_RETRY
        sched.cache.record_fresh_measurement(1, "f2", 2, 0.08, bx_shift, False, None, None, None, {"inlier_ratio": 0.2}, ("low_inlier_ratio",))
        r_hold = sched.decide_frame(3, 0.12, [bx_shift], [1])
        self.assertFalse(r_hold["run_depth"])
        r_retry = sched.decide_frame(6, 0.24, [bx_shift], [1])
        self.assertTrue(r_retry["run_depth"])
        self.assertIn("INVALID_RETRY", r_retry["frame_reasons"])

    def test_scene_change_reset_budget_skip_and_determinism(self):
        cfg = AdaptiveSchedulerConfig(
            strategy="E_budget_aware",
            budget_target_ema_ms=25.0,
            budget_min_gap_sec=0.15,
            starvation_override_sec=0.50,
            refresh_interval_sec=0.10,
        )
        s1 = AdaptiveDepthScheduler(cfg)
        s2 = AdaptiveDepthScheduler(cfg)
        f_dark = np.zeros((72, 128, 3), dtype=np.uint8)
        f_bright = np.full((72, 128, 3), 240, dtype=np.uint8)
        bx = [(10, 10, 50, 50)]

        # Identical initial calls must be 100% deterministic
        o1 = s1.decide_frame(0, 0.00, bx, [1], frame_bgr=f_dark)
        o2 = s2.decide_frame(0, 0.00, bx, [1], frame_bgr=f_dark)
        self.assertEqual(o1["run_depth"], o2["run_depth"])
        self.assertEqual(o1["frame_reasons"], o2["frame_reasons"])
        s1.cache.record_fresh_measurement(1, "f0", 0, 0.00, bx[0], True, 3.0, 0.1, 0.1, {"inlier_ratio": 0.9})

        # Simulate high EMA latency & periodic trigger at dt=0.11s -> SKIPPED_BY_BUDGET
        s1.update_observed_latency(90.0)
        ob = s1.decide_frame(3, 0.11, bx, [1], frame_bgr=f_dark)
        self.assertFalse(ob["run_depth"])
        self.assertEqual(ob["frame_reasons"], ["SKIPPED_BY_BUDGET"])

        # Scene cut overrides & resets cache
        oc = s1.decide_frame(4, 0.14, bx, [1], frame_bgr=f_bright)
        self.assertTrue(oc["run_depth"])
        self.assertIn("SCENE_CHANGE", oc["frame_reasons"])
        self.assertGreaterEqual(s1.cache.reset_events, 1)

    def test_async_worker_bounded_queue_and_stale_out_of_order_rejection(self):
        def slow_fake_depth(img):
            time.sleep(0.06)
            return {"depth_map": np.ones((36, 64), dtype=np.float32)}

        worker = BoundedAsyncDepthWorker(slow_fake_depth, max_result_age_sec=0.15)
        worker.start()
        try:
            dummy = np.zeros((36, 64, 3), dtype=np.uint8)
            ok1 = worker.submit(0, 0.00, dummy, {1: (1, 1, 10, 10)})
            self.assertTrue(ok1)
            # Burst submissions while worker sleeps -> bounded queue drops excess safely
            burst_accepted = [worker.submit(i, i * 0.04, dummy, {1: (1, 1, 10, 10)}) for i in range(1, 5)]
            self.assertGreater(worker.dropped_queue_full, 0)
            self.assertFalse(all(burst_accepted))

            time.sleep(0.18)
            # Poll much later (t=0.50s > max_result_age=0.15s) -> rejected cleanly as stale!
            res = worker.poll_latest_valid(current_frame_idx=12, current_ts_sec=0.50)
            self.assertIsNone(res)
            self.assertGreater(worker.rejected_stale_or_order, 0)
        finally:
            worker.stop()

    def test_production_hashes_still_preserved(self):
        # Directly reuse canonical Phase 5C production hash verifier
        t = TestShadowDepthPipeline(methodName="test_production_file_hashes_preserved")
        t.test_production_file_hashes_preserved()


if __name__ == "__main__":
    unittest.main()
