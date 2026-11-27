"""
tests/test_phase10_geometry_visualization.py — Focused tests for Phase 10 relative
2.5D cavity surface construction, residual polarity, null preservation, grid downsampling,
cross-sectional profiles, snapshot limits, security, and backward compatibility.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.core_aware_cavity_extraction import extract_cavity_with_strategy
from experiments.optimized_relative_depth import OptimizedDepthConfig
from roadvision_v2.config import GeometrySnapshotConfig, RoadVisionV2Config
from roadvision_v2.geometry import (
    RESIDUAL_POLARITY_DOC,
    GeometrySnapshotManager,
    build_relative_surface_snapshot,
    compute_cross_section_profiles,
)


class TestPhase10GeometryVisualization(unittest.TestCase):
    def _make_synthetic_depth(self, cavity_depth=0.22, add_nans=False):
        H, W = 120, 160
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        depth_map = 1.0 + 0.0015 * yy - 0.0008 * xx
        box = (55.0, 40.0, 105.0, 80.0)
        if cavity_depth != 0.0:
            ell = ((xx - 80.0) / 18.0) ** 2 + ((yy - 60.0) / 14.0) ** 2
            depth_map[ell <= 1.0] -= cavity_depth * (1.0 - ell[ell <= 1.0])
        if add_nans:
            depth_map[56:64, 76:84] = np.nan
        img_bgr = np.full((H, W, 3), 90, dtype=np.uint8)
        return depth_map, box, img_bgr

    def test_planar_and_cavity_residuals_and_polarity(self):
        # Planar surface yields near-zero residuals inside ROI
        d_flat, box, img_bgr = self._make_synthetic_depth(cavity_depth=0.0)
        opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
        ext_flat = extract_cavity_with_strategy(d_flat, box, strategy="elliptical_core", opt_cfg=opt_cfg, return_arrays=True)
        snap_flat = build_relative_surface_snapshot(
            snapshot_id="snap_flat",
            session_id="s1",
            frame_idx=0,
            timestamp_sec=0.0,
            track_id=1,
            track_generation_id=0,
            depth_map=d_flat,
            extract_out=ext_flat,
            img_bgr=img_bgr,
            max_grid_size=24,
        )
        self.assertEqual(snap_flat["residual_polarity"], RESIDUAL_POLARITY_DOC)
        self.assertFalse(snap_flat["is_metric_3d"])
        self.assertLess(abs(snap_flat["max_residual"]), 1e-3)

        # Cavity surface yields positive signed residual inside depressed cavity
        d_cav, box, img_bgr = self._make_synthetic_depth(cavity_depth=0.25)
        ext_cav = extract_cavity_with_strategy(d_cav, box, strategy="elliptical_core", opt_cfg=opt_cfg, return_arrays=True)
        self.assertTrue(ext_cav["is_valid"])
        snap_cav = build_relative_surface_snapshot(
            snapshot_id="snap_cav",
            session_id="s1",
            frame_idx=1,
            timestamp_sec=0.04,
            track_id=1,
            track_generation_id=0,
            depth_map=d_cav,
            extract_out=ext_cav,
            img_bgr=img_bgr,
            max_grid_size=32,
        )
        gh, gw = snap_cav["grid_shape"]
        self.assertLessEqual(gh, 32)
        self.assertLessEqual(gw, 32)
        self.assertGreater(snap_cav["max_residual"], 0.15)
        ref_count = sum(1 for row in snap_cav["reference_mask_grid"] for v in row if v)
        cav_count = sum(1 for row in snap_cav["cavity_mask_grid"] for v in row if v)
        self.assertGreater(ref_count, 0)
        self.assertGreater(cav_count, 0)
        stats = snap_cav.get("residual_stats") or {}
        self.assertGreater(stats.get("cavity_median_residual", 0.0), stats.get("reference_median_residual", 1.0))
        self.assertTrue(snap_cav["source_crop_data_url"].startswith("data:image/jpeg;base64,"))
        json.dumps(snap_cav, allow_nan=False)

    def test_invalid_masks_preserve_nulls_and_cross_section_gaps(self):
        d_nan, box, img_bgr = self._make_synthetic_depth(cavity_depth=0.25, add_nans=True)
        ext = extract_cavity_with_strategy(d_nan, box, strategy="elliptical_core", return_arrays=True)
        snap = build_relative_surface_snapshot(
            snapshot_id="snap_nan",
            session_id="s1",
            frame_idx=2,
            timestamp_sec=0.08,
            track_id=1,
            track_generation_id=0,
            depth_map=d_nan,
            extract_out=ext,
            img_bgr=None,
            max_grid_size=48,
            include_source_crop=False,
        )
        # Central NaN block must produce None (`null`) + False validity cells without interpolation
        has_null = any(
            snap["signed_residual_grid"][r][c] is None and not snap["validity_mask_grid"][r][c]
            for r in range(snap["grid_shape"][0])
            for c in range(snap["grid_shape"][1])
        )
        self.assertTrue(has_null)
        prof = compute_cross_section_profiles(
            snap["signed_residual_grid"],
            snap["validity_mask_grid"],
            row_idx=snap["grid_shape"][0] // 2,
            col_idx=snap["grid_shape"][1] // 2,
        )
        self.assertTrue(any((not pt["valid"]) and pt["residual"] is None for pt in prof["horizontal_profile"]))

    def test_snapshot_manager_limits_and_manifest(self):
        d_cav, box, img_bgr = self._make_synthetic_depth(cavity_depth=0.25)
        ext = extract_cavity_with_strategy(d_cav, box, strategy="elliptical_core", return_arrays=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            gcfg = GeometrySnapshotConfig(
                enabled=True,
                max_grid_size=24,
                max_snapshots_per_track=2,
                max_snapshots_per_job=3,
            )
            mgr = GeometrySnapshotManager(
                session_id="sess1",
                output_dir=tmpdir,
                mode="balanced",
                stem="clip",
                cfg=gcfg,
            )
            captured = []
            for fidx in (0, 1, 5, 10, 15):
                sid, ts = mgr.maybe_capture(
                    frame_idx=fidx,
                    timestamp_sec=fidx * 0.04,
                    track_id=1,
                    track_generation_id=0,
                    depth_map=d_cav,
                    extract_out=ext,
                    img_bgr=img_bgr,
                    model_identity="depth_anything_v2_small",
                    depth_input_resolution=336,
                    inference_precision="fp16_weights",
                    quality_assessment={"quality_index": 0.85, "quality_category": "HIGH_QUALITY"},
                )
                if sid:
                    captured.append(sid)
            # Frame 1 skipped due to spacing (<3 frames), capped at max_snapshots_per_track=2
            self.assertEqual(len(captured), 2)
            mpath = mgr.write_manifest()
            self.assertTrue(os.path.isfile(mpath))
            with open(mpath, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["snapshot_count"], 2)
            self.assertIsNotNone(manifest["global_max_residual"])

    def test_default_config_keeps_geometry_recording_disabled(self):
        cfg = RoadVisionV2Config(mode="balanced")
        cfg.validate(check_assets=False)
        self.assertFalse(cfg.geometry_snapshot.enabled)


if __name__ == "__main__":
    unittest.main()
