import unittest
import numpy as np

from experiments.validity_aware_temporal_filter import (
    FILTER_MODES,
    TemporalFilterConfig,
    ValidityAwareTrackFilter,
    evaluate_temporal_filter_suite,
)


class TestValidityAwareTemporalFilter(unittest.TestCase):
    def test_invalid_observations_never_pollute_numeric_state(self):
        cfg = TemporalFilterConfig(alpha=0.4, rolling_window=5, max_stale_frames=2, max_missed_frames_reset=6)
        flt = ValidityAwareTrackFilter(mode="validity_aware_ema", cfg=cfg)
        leg = ValidityAwareTrackFilter(mode="legacy_ema_zero_clamp", cfg=cfg)

        r0_f = flt.update(1, 10.0, True, 0, 0.0)
        r0_l = leg.update(1, 10.0, True, 0, 0.0)
        self.assertEqual(r0_f["filtered_value"], 10.0)
        self.assertEqual(r0_l["filtered_value"], 10.0)

        # Invalid step with extreme glitch 999.0 and then NaN
        r1_f = flt.update(1, 999.0, False, 1, 0.1)
        r1_l = leg.update(1, 999.0, False, 1, 0.1)
        self.assertFalse(r1_f["did_update"])
        self.assertTrue(r1_f["is_stale"])
        self.assertEqual(r1_f["filtered_value"], 10.0)   # Held cleanly without absorbing 999.0!
        self.assertLess(r1_l["filtered_value"], 7.0)     # Legacy clamped 999 -> 0.0 and pulled EMA down!

        r2_f = flt.update(1, float("nan"), False, 2, 0.2)
        self.assertTrue(r2_f["is_stale"])
        self.assertFalse(r2_f["is_expired"])
        self.assertEqual(r2_f["frames_since_last_valid"], 2)
        self.assertAlmostEqual(r2_f["time_since_last_valid_sec"], 0.2, places=3)

        # Step 3 exceeds max_stale_frames=2 -> expires explicitly to None (no indefinite reuse!)
        r3_f = flt.update(1, None, False, 3, 0.3)
        self.assertTrue(r3_f["is_stale"])
        self.assertTrue(r3_f["is_expired"])
        self.assertIsNone(r3_f["filtered_value"])
        self.assertEqual(r3_f["track_age_frames"], 3)
        self.assertAlmostEqual(r3_f["track_age_sec"], 0.3, places=3)

    def test_never_valid_track_stays_none_and_never_fabricates_zero(self):
        for mode in ("raw", "validity_aware_ema", "rolling_median_validity_ema"):
            flt = ValidityAwareTrackFilter(mode=mode)
            r0 = flt.update(5, 0.0, False, 0, 0.0)
            r1 = flt.update(5, 12.0, False, 1, 0.1)
            self.assertIsNone(r0["filtered_value"])
            self.assertIsNone(r1["filtered_value"])
            self.assertEqual(r1["invalid_count"], 2)
            self.assertEqual(r1["valid_count"], 0)

    def test_track_reset_and_absence_expiration(self):
        cfg = TemporalFilterConfig(alpha=0.5, max_stale_frames=2, max_missed_frames_reset=5)
        flt = ValidityAwareTrackFilter(mode="rolling_median_validity_ema", cfg=cfg)

        flt.update(10, 4.0, True, 0, 0.0)
        flt.update(10, 4.0, True, 1, 0.1)
        # Re-appearing after gap >= max_missed_frames_reset=5 resets state completely
        r_reappear = flt.update(10, 20.0, True, 8, 0.8)
        self.assertEqual(r_reappear["filtered_value"], 20.0)
        self.assertEqual(r_reappear["track_age_frames"], 0)
        self.assertEqual(r_reappear["valid_count"], 1)

        # Explicit step_missing_tracks cleanup
        expired = flt.step_missing_tracks(active_track_ids=[], frame_idx=15)
        self.assertIn(10, expired)
        self.assertIsNone(flt.get_state(10))

    def test_rolling_median_resists_transient_valid_spike(self):
        cfg = TemporalFilterConfig(alpha=0.5, rolling_window=5)
        ema = ValidityAwareTrackFilter(mode="validity_aware_ema", cfg=cfg)
        med = ValidityAwareTrackFilter(mode="rolling_median_validity_ema", cfg=cfg)
        stream = [10.0, 10.0, 10.0, 40.0, 10.0]
        for i, val in enumerate(stream):
            re = ema.update(1, val, True, i, i * 0.1)
            rm = med.update(1, val, True, i, i * 0.1)
            if i == 3:
                # Single spike of 40.0 moves plain EMA strongly (`0.5*40+0.5*10 = 25`), whereas rolling median ignores it!
                self.assertGreater(re["filtered_value"], 20.0)
                self.assertEqual(rm["filtered_value"], 10.0)

    def test_synthetic_temporal_suite_executes_cleanly(self):
        summary = evaluate_temporal_filter_suite()
        self.assertIn("invalid_dropouts_with_glitches", summary)
        self.assertIn("prolonged_invalid_expiry", summary)
        self.assertGreater(
            summary["prolonged_invalid_expiry"]["validity_aware_ema"]["expired_steps"], 0
        )
        for m in FILTER_MODES:
            self.assertIn(m, summary["abrupt_step"])


if __name__ == "__main__":
    unittest.main()
