"""
tests/test_roadvision_v2_config.py — Unit tests for RoadVision AI 2.0 configuration system.

Covers:
  - Mode validation (live/balanced/survey pass; invalid fails)
  - DAv2 input_size mod-14 check
  - MiDaS input_size=256 enforcement
  - MiDaS rejects fp16_weights
  - Asset missing raises FileNotFoundError
  - enable_depth override works
  - YAML load roundtrip (check_assets=False)
  - Scheduler / cache_temporal sync
"""
import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_v2.config import (
    RoadVisionV2Config,
    DetectorConfig,
    DepthConfig,
    RansacConfig,
    CavityConfig,
    SchedulerConfig,
    CacheTemporalConfig,
    DiagnosticsConfig,
    VALID_MODES,
    VALID_DEPTH_MODELS,
    VALID_PRECISIONS,
    load_v2_config,
)


def _make_minimal_yaml(mode: str = "balanced") -> str:
    """Return a minimal valid YAML string for testing."""
    return textwrap.dedent(f"""\
        version: "2.0.0-test"
        detector:
          model_id: "yolov8n_test"
          checkpoint_path: "__FAKE_DET__"
          tracker_config: "road_tracker.yaml"
          input_size: 640
          conf_threshold: 0.25
          iou_threshold: 0.50
          horizon_cutoff_ratio: 0.15
        ransac:
          sampling_mode: "stratified_quadrant"
          max_ref_samples: 512
          min_iters: 16
          max_iters: 128
          batch_size: 16
          confidence: 0.99
          enable_early_stop: true
          seed: 42
          min_ref_samples: 40
          min_quadrant_samples: 5
          min_populated_quadrants: 2
          max_single_quadrant_frac: 0.80
          min_annular_retention: 0.25
          min_positive_depress_frac: 0.35
        cache_temporal:
          max_cache_age_sec: 0.45
          max_invalid_stale_sec: 0.25
          max_absence_reset_sec: 0.60
          max_reuse_centroid_norm: 1.75
          tau_sec: 0.16
          rolling_window: 5
          fixed_alpha: 0.35
        modes:
          live:
            depth:
              enabled: false
              model_id: "midas_small"
              input_size: 256
              precision: "fp16_autocast"
              checkpoint_path: null
            cavity:
              strategy: "full_box"
              reject_water_patches: true
            scheduler:
              strategy: "B_candidate_gated"
              fixed_interval_k: 3
              refresh_interval_sec: 0.16
              max_measurement_age_sec: 0.45
              budget_target_ema_ms: 33.0
              budget_min_gap_sec: 0.08
            diagnostics:
              enable_horizon_proxy: false
              show_timing_hud: true
          balanced:
            depth:
              enabled: false
              model_id: "midas_small"
              input_size: 256
              precision: "fp16_autocast"
              checkpoint_path: null
            cavity:
              strategy: "elliptical_core"
              reject_water_patches: true
            scheduler:
              strategy: "E_budget_aware"
              fixed_interval_k: 3
              refresh_interval_sec: 0.16
              max_measurement_age_sec: 0.45
              budget_target_ema_ms: 38.0
              budget_min_gap_sec: 0.08
            diagnostics:
              enable_horizon_proxy: false
              show_timing_hud: true
          survey:
            depth:
              enabled: false
              model_id: "midas_small"
              input_size: 256
              precision: "fp16_autocast"
              checkpoint_path: null
            cavity:
              strategy: "elliptical_core"
              reject_water_patches: true
            scheduler:
              strategy: "B_candidate_gated"
              fixed_interval_k: 1
              refresh_interval_sec: 0.0
              max_measurement_age_sec: 0.45
              budget_target_ema_ms: 1000.0
              budget_min_gap_sec: 0.0
            diagnostics:
              enable_horizon_proxy: false
              show_timing_hud: true
    """)


class TestRoadVisionV2ConfigValidation(unittest.TestCase):

    def _write_yaml(self, content: str) -> str:
        """Write content to a temp file and return its path."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        f.write(content)
        f.flush()
        f.close()
        return f.name

    # --- Mode validation ---

    def test_valid_modes_accepted(self):
        """All three canonical modes must be accepted."""
        cfg = RoadVisionV2Config(mode="live")
        cfg.validate(check_assets=False)
        cfg = RoadVisionV2Config(mode="balanced")
        cfg.validate(check_assets=False)
        cfg = RoadVisionV2Config(mode="survey")
        cfg.validate(check_assets=False)

    def test_invalid_mode_raises(self):
        cfg = RoadVisionV2Config(mode="turbo")
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    def test_valid_modes_constant(self):
        self.assertIn("live", VALID_MODES)
        self.assertIn("balanced", VALID_MODES)
        self.assertIn("survey", VALID_MODES)

    # --- Depth model checks ---

    def test_dav2_input_size_mod14_violation_raises(self):
        """DAv2 input_size=300 (not mod 14) must raise ValueError."""
        depth = DepthConfig(enabled=True, model_id="depth_anything_v2_small",
                            input_size=300, precision="fp16_weights")
        cfg = RoadVisionV2Config(depth=depth)
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    def test_dav2_input_size_336_valid(self):
        depth = DepthConfig(enabled=True, model_id="depth_anything_v2_small",
                            input_size=336, precision="fp16_weights")
        cfg = RoadVisionV2Config(depth=depth)
        cfg.validate(check_assets=False)  # must not raise

    def test_dav2_input_size_518_valid(self):
        depth = DepthConfig(enabled=True, model_id="depth_anything_v2_small",
                            input_size=518, precision="fp32")
        cfg = RoadVisionV2Config(depth=depth)
        cfg.validate(check_assets=False)

    def test_midas_input_size_non_256_raises(self):
        depth = DepthConfig(enabled=True, model_id="midas_small",
                            input_size=320, precision="fp16_autocast")
        cfg = RoadVisionV2Config(depth=depth)
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    def test_midas_fp16_weights_raises(self):
        depth = DepthConfig(enabled=True, model_id="midas_small",
                            input_size=256, precision="fp16_weights")
        cfg = RoadVisionV2Config(depth=depth)
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    def test_midas_fp16_autocast_valid(self):
        depth = DepthConfig(enabled=True, model_id="midas_small",
                            input_size=256, precision="fp16_autocast")
        cfg = RoadVisionV2Config(depth=depth)
        cfg.validate(check_assets=False)

    # --- Detector checks ---

    def test_bad_det_input_size_raises(self):
        det = DetectorConfig(input_size=100)  # not mult of 32
        cfg = RoadVisionV2Config(detector=det)
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    def test_bad_conf_threshold_raises(self):
        det = DetectorConfig(conf_threshold=1.1)
        cfg = RoadVisionV2Config(detector=det)
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    # --- RANSAC bounds ---

    def test_bad_ransac_iters_raises(self):
        ransac = RansacConfig(min_iters=200, max_iters=16)
        cfg = RoadVisionV2Config(ransac=ransac)
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    # --- Asset existence check ---

    def test_missing_det_checkpoint_raises(self):
        det = DetectorConfig(checkpoint_path="/nonexistent/model.pt")
        cfg = RoadVisionV2Config(detector=det)
        with self.assertRaises(FileNotFoundError):
            cfg.validate(check_assets=True)

    # --- enable_depth override ---

    def test_enable_depth_override_via_load(self):
        yaml_path = self._write_yaml(_make_minimal_yaml())
        try:
            cfg = load_v2_config(
                config_path=yaml_path,
                mode="balanced",
                enable_depth=False,
                check_assets=False,
            )
            self.assertFalse(cfg.depth.enabled)
        finally:
            os.unlink(yaml_path)

    def test_enable_depth_override_true(self):
        yaml_path = self._write_yaml(_make_minimal_yaml())
        try:
            # balanced mode has depth.enabled=false in the minimal YAML above;
            # override to True with a valid precision
            cfg = load_v2_config(
                config_path=yaml_path,
                mode="balanced",
                enable_depth=True,
                overrides={"depth": {"precision": "fp16_autocast", "input_size": 256}},
                check_assets=False,
            )
            self.assertTrue(cfg.depth.enabled)
        finally:
            os.unlink(yaml_path)

    # --- YAML round-trip ---

    def test_yaml_load_roundtrip_all_modes(self):
        yaml_path = self._write_yaml(_make_minimal_yaml())
        try:
            for mode in ("live", "balanced", "survey"):
                cfg = load_v2_config(
                    config_path=yaml_path,
                    mode=mode,
                    check_assets=False,
                )
                self.assertEqual(cfg.mode, mode)
                d = cfg.to_dict()
                self.assertIn("mode", d)
        finally:
            os.unlink(yaml_path)

    def test_missing_config_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_v2_config(config_path="/nonexistent/config.yaml", mode="live", check_assets=False)

    def test_invalid_mode_in_load_raises(self):
        yaml_path = self._write_yaml(_make_minimal_yaml())
        try:
            with self.assertRaises(ValueError):
                load_v2_config(config_path=yaml_path, mode="extreme", check_assets=False)
        finally:
            os.unlink(yaml_path)

    # --- Cavity strategy validation ---

    def test_valid_cavity_strategy_elliptical(self):
        cav = CavityConfig(strategy="elliptical_core")
        cfg = RoadVisionV2Config(cavity=cav)
        cfg.validate(check_assets=False)

    def test_invalid_cavity_strategy_raises(self):
        cav = CavityConfig(strategy="hexagonal_mosaics")
        cfg = RoadVisionV2Config(cavity=cav)
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    # --- Cache temporal checks ---

    def test_bad_cache_age_raises(self):
        cache = CacheTemporalConfig(max_cache_age_sec=-0.1, tau_sec=0.16)
        cfg = RoadVisionV2Config(cache_temporal=cache)
        with self.assertRaises(ValueError):
            cfg.validate(check_assets=False)

    def test_scheduler_max_age_synced_with_cache(self):
        """Scheduler expiry must stay in sync with cache expiry."""
        yaml_path = self._write_yaml(_make_minimal_yaml())
        try:
            cfg = load_v2_config(config_path=yaml_path, mode="live", check_assets=False)
            self.assertEqual(cfg.scheduler.max_measurement_age_sec, cfg.cache_temporal.max_cache_age_sec)
        finally:
            os.unlink(yaml_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
