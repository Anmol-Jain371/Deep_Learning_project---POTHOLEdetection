import os
import sys
import unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.dynamic_horizon_estimator import (
    HorizonEstimatorConfig,
    HorizonPolicyConfig,
    estimate_vanishing_point,
    DynamicHorizonTracker,
)
from tests.test_shadow_depth_pipeline import TestShadowDepthPipeline


def render_synthetic_perspective_road(
    w: int = 640,
    h: int = 360,
    vp_norm: tuple = (0.50, 0.40),
    roll_deg: float = 0.0,
    missing_markings: bool = False,
    spurious_clutter: bool = False,
    shadow_band: bool = False,
    noise_sigma: float = 0.0,
    incompatible_second_vp: bool = False,
    seed: int = 7,
) -> np.ndarray:
    """Render synthetic perspective road scene with known projective vanishing point `vp_norm`."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    vx, vy = vp_norm[0] * w, vp_norm[1] * h

    # Sky above horizon
    h_int = int(np.clip(vy, 0, h - 1))
    img[:h_int, :, :] = (210, 195, 175)
    # Dark asphalt road polygon converging to (vx, vy)
    pts_road = np.array([[int(vx), int(vy)], [w - int(0.06 * w), h - 1], [int(0.06 * w), h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [pts_road], (68, 70, 72))

    # Draw converging curbs + optional center lane lines radiating from (vx, vy)
    ang_rad = np.deg2rad(roll_deg)
    def rot_pt(x, y):
        dx, dy = x - w * 0.5, y - h * 0.5
        return (int(round(w * 0.5 + dx * np.cos(ang_rad) - dy * np.sin(ang_rad))),
                int(round(h * 0.5 + dx * np.sin(ang_rad) + dy * np.cos(ang_rad))))

    targets = [0.06, 0.94] if missing_markings else [0.06, 0.25, 0.48, 0.52, 0.75, 0.94]
    for tf in targets:
        p0 = rot_pt(vx + (tf * w - vx) * 0.14, vy + (h - 1 - vy) * 0.14)
        p1 = rot_pt(tf * w, h - 1)
        col = (225, 225, 225) if (0.20 < tf < 0.80) else (200, 195, 185)
        cv2.line(img, p0, p1, col, max(1, int(round(w / 320.0))))

    if shadow_band:
        sy1, sy2 = int(0.58 * h), int(0.72 * h)
        cv2.rectangle(img, (0, sy1), (w, sy2), (40, 42, 44), -1)
        # Redraw curbs softly through shadow
        for tf in (0.06, 0.94):
            p0 = rot_pt(vx + (tf * w - vx) * 0.14, vy + (h - 1 - vy) * 0.14)
            p1 = rot_pt(tf * w, h - 1)
            cv2.line(img, p0, p1, (165, 165, 165), max(1, int(round(w / 320.0))))

    if spurious_clutter:
        # Vertical poles, horizontal cracks, random distractor segments
        for px in (int(0.12 * w), int(0.88 * w)):
            cv2.line(img, (px, int(0.25 * h)), (px, int(0.85 * h)), (220, 220, 220), 2)
        for py in (int(0.65 * h), int(0.80 * h)):
            cv2.line(img, (int(0.15 * w), py), (int(0.85 * w), py + 2), (210, 210, 210), 2)

    if incompatible_second_vp:
        # Inject strong competing cross-street lines converging to (0.15*w, 0.55*h)
        vx2, vy2 = 0.15 * w, 0.55 * h
        for tx in (0.40, 0.55, 0.70, 0.85):
            cv2.line(img, (int(vx2), int(vy2)), (int(tx * w), h - 1), (235, 235, 235), 2)

    if noise_sigma > 0.0:
        noise = rng.normal(0.0, noise_sigma, size=img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


class TestDynamicHorizonEstimator(unittest.TestCase):
    def test_synthetic_vanishing_point_recovery_and_scale_invariance(self):
        cfg = HorizonEstimatorConfig(detector_mode="lsd", process_width=320)
        img_360 = render_synthetic_perspective_road(640, 360, vp_norm=(0.54, 0.36))
        img_720 = render_synthetic_perspective_road(1280, 720, vp_norm=(0.54, 0.36))

        r360 = estimate_vanishing_point(img_360, cfg=cfg)
        r720 = estimate_vanishing_point(img_720, cfg=cfg)
        self.assertEqual(r360["state"], "VALID")
        self.assertEqual(r720["state"], "VALID")
        self.assertLess(abs(r360["horizon_y_ratio"] - 0.36), 0.025)
        self.assertLess(abs(r720["horizon_y_ratio"] - 0.36), 0.025)
        # Coordinate scaling invariant across resolutions
        self.assertLess(abs(r360["horizon_y_ratio"] - r720["horizon_y_ratio"]), 0.015)

    def test_parallel_degenerate_missing_and_insufficient_lines_fail_cleanly(self):
        cfg = HorizonEstimatorConfig(detector_mode="lsd", process_width=320)
        # Blank road / zero line segments
        blank = np.full((360, 640, 3), 120, dtype=np.uint8)
        rb = estimate_vanishing_point(blank, cfg=cfg)
        self.assertEqual(rb["state"], "UNAVAILABLE")
        self.assertIsNone(rb["horizon_y_ratio"])

        # Strictly parallel angled lines (all identical slope -> det ~ 0)
        par = np.full((360, 640, 3), 80, dtype=np.uint8)
        for offset in (-120, -40, 40, 120):
            cv2.line(par, (220 + offset, 100), (120 + offset, 330), (230, 230, 230), 2)
        rp = estimate_vanishing_point(par, cfg=cfg)
        self.assertIn(rp["state"], ("UNAVAILABLE", "LOW_CONFIDENCE"))
        self.assertNotEqual(rp["state"], "VALID")

    def test_outlier_resistance_and_determinism(self):
        cfg = HorizonEstimatorConfig(detector_mode="lsd", process_width=320)
        noisy_clutter = render_synthetic_perspective_road(
            640, 360, vp_norm=(0.45, 0.44), spurious_clutter=True, noise_sigma=8.0
        )
        r1 = estimate_vanishing_point(noisy_clutter, cfg=cfg)
        r2 = estimate_vanishing_point(noisy_clutter, cfg=cfg)
        self.assertEqual(r1["state"], "VALID")
        self.assertLess(abs(r1["horizon_y_ratio"] - 0.44), 0.03)
        # Determinism
        self.assertEqual(r1["vp_norm"], r2["vp_norm"])
        self.assertEqual(r1["diagnostics"], r2["diagnostics"])

    def test_stale_expiration_resolution_change_and_profile_reset(self):
        pol = HorizonPolicyConfig(policy="D_dynamic_fallback", fixed_horizon_ratio=0.40, max_stale_sec=0.30, max_expired_sec=0.60)
        tracker = DynamicHorizonTracker(HorizonEstimatorConfig(detector_mode="lsd"), pol)

        road = render_synthetic_perspective_road(640, 360, vp_norm=(0.50, 0.34))
        blank = np.full((360, 640, 3), 90, dtype=np.uint8)

        u0 = tracker.update(road, 0, 0.00, session_id="s1", camera_profile="sedan_dashcam")
        self.assertEqual(u0["output_state"], "VALID")
        self.assertFalse(u0["context_reset"])

        # Unavailable at t=0.20s -> STALE hold (< max_expired_sec)
        u1 = tracker.update(blank, 5, 0.20, session_id="s1", camera_profile="sedan_dashcam")
        self.assertEqual(u1["output_state"], "STALE")
        self.assertIsNotNone(u1["smoothed_horizon_ratio"])

        # Unavailable at t=0.75s (> max_expired_sec=0.60s) -> EXPIRED + fallback to 0.40!
        u2 = tracker.update(blank, 18, 0.75, session_id="s1", camera_profile="sedan_dashcam")
        self.assertEqual(u2["output_state"], "EXPIRED")
        self.assertTrue(u2["fallback_used"])
        self.assertEqual(u2["active_horizon_ratio"], 0.40)

        # Camera profile change resets state cleanly
        _ = tracker.update(road, 19, 0.80, session_id="s1", camera_profile="sedan_dashcam")
        u_prof = tracker.update(road, 20, 0.84, session_id="s1", camera_profile="suv_crossover", fixed_horizon_override=0.42)
        self.assertTrue(u_prof["context_reset"])

        # Resolution change resets state cleanly
        road_720 = render_synthetic_perspective_road(1280, 720, vp_norm=(0.50, 0.34))
        u_res = tracker.update(road_720, 21, 0.88, session_id="s1", camera_profile="suv_crossover")
        self.assertTrue(u_res["context_reset"])

    def test_production_hashes_intact(self):
        t = TestShadowDepthPipeline(methodName="test_production_file_hashes_preserved")
        t.test_production_file_hashes_preserved()


if __name__ == "__main__":
    unittest.main()
