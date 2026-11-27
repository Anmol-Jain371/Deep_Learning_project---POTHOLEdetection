"""
Phase 10: Relative 2.5D Cavity Surface Representation & Bounded Snapshot Artifacts.

Constructs compact, bounded relative cavity surfaces directly from already-computed
depth maps, RANSAC reference plane coefficients, signed residual fields, and
annular/core masks without performing any additional depth model inference.

Scientific & Engineering Guarantees:
- Residual polarity is `ref_plane - inv_depth` (positive = depressed cavity relative
  to the fitted road plane; negative = elevation above plane).
- Coordinates are normalized image-plane ROI coordinates [0, 1]; values are
  uncalibrated relative inverse-depth residuals (NEVER physical cm/mm/volume).
- Missing or non-finite samples remain `None` (`null` in JSON) with explicit
  `validity_mask_grid`; missing holes are NEVER interpolated.
- Geometry snapshots are generated ONLY from FRESH_VALID observations; cached
  observations may link back to their source fresh snapshot ID and timestamp.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadvision_v2.config import GeometrySnapshotConfig

logger = logging.getLogger("roadvision_v2.geometry.surface")

_SAFE_SNAP_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
RESIDUAL_POLARITY_DOC = "ref_plane_minus_inv_depth_positive_depression"


def _downsample_grid_preserve_nulls(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    target_h: int,
    target_w: int,
) -> Tuple[List[List[Optional[float]]], List[List[bool]]]:
    """
    Bin-average a 2D array into (target_h, target_w) while strictly preserving
    invalid/missing cells as None (`null`) without interpolating across gaps.
    """
    H, W = arr.shape[:2]
    row_edges = np.linspace(0, H, target_h + 1, dtype=int)
    col_edges = np.linspace(0, W, target_w + 1, dtype=int)

    values_out: List[List[Optional[float]]] = []
    valid_out: List[List[bool]] = []
    for r in range(target_h):
        r0, r1 = int(row_edges[r]), max(int(row_edges[r]) + 1, int(row_edges[r + 1]))
        r1 = min(H, r1)
        row_vals: List[Optional[float]] = []
        row_valids: List[bool] = []
        for c in range(target_w):
            c0, c1 = int(col_edges[c]), max(int(col_edges[c]) + 1, int(col_edges[c + 1]))
            c1 = min(W, c1)
            sub_valid = valid_mask[r0:r1, c0:c1] & np.isfinite(arr[r0:r1, c0:c1])
            if np.any(sub_valid):
                mean_val = float(np.mean(arr[r0:r1, c0:c1][sub_valid]))
                row_vals.append(round(mean_val, 6))
                row_valids.append(True)
            else:
                row_vals.append(None)
                row_valids.append(False)
        values_out.append(row_vals)
        valid_out.append(row_valids)
    return values_out, valid_out


def _downsample_bool_mask(mask: np.ndarray, target_h: int, target_w: int) -> List[List[bool]]:
    H, W = mask.shape[:2]
    row_edges = np.linspace(0, H, target_h + 1, dtype=int)
    col_edges = np.linspace(0, W, target_w + 1, dtype=int)
    out: List[List[bool]] = []
    for r in range(target_h):
        r0, r1 = int(row_edges[r]), max(int(row_edges[r]) + 1, int(row_edges[r + 1]))
        r1 = min(H, r1)
        row_b: List[bool] = []
        for c in range(target_w):
            c0, c1 = int(col_edges[c]), max(int(col_edges[c]) + 1, int(col_edges[c + 1]))
            c1 = min(W, c1)
            sub = mask[r0:r1, c0:c1]
            row_b.append(bool(np.mean(sub) >= 0.5) if sub.size > 0 else False)
        out.append(row_b)
    return out


def compute_cross_section_profiles(
    residual_grid: Sequence[Sequence[Optional[float]]],
    validity_grid: Sequence[Sequence[bool]],
    row_idx: Optional[int] = None,
    col_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract horizontal and vertical cross-sectional profiles directly from a residual grid
    without interpolating across invalid/missing cells.
    """
    gh = len(residual_grid)
    gw = len(residual_grid[0]) if gh > 0 else 0
    if gh == 0 or gw == 0:
        return {
            "row_idx": 0,
            "col_idx": 0,
            "horizontal_profile": [],
            "vertical_profile": [],
        }
    r_idx = max(0, min(gh - 1, (gh // 2) if row_idx is None else int(row_idx)))
    c_idx = max(0, min(gw - 1, (gw // 2) if col_idx is None else int(col_idx)))

    horiz = [
        {
            "x_norm": round((c + 0.5) / gw, 4),
            "residual": residual_grid[r_idx][c] if validity_grid[r_idx][c] else None,
            "valid": bool(validity_grid[r_idx][c]),
        }
        for c in range(gw)
    ]
    vert = [
        {
            "y_norm": round((r + 0.5) / gh, 4),
            "residual": residual_grid[r][c_idx] if validity_grid[r][c_idx] else None,
            "valid": bool(validity_grid[r][c_idx]),
        }
        for r in range(gh)
    ]
    return {
        "row_idx": r_idx,
        "col_idx": c_idx,
        "row_y_norm": round((r_idx + 0.5) / gh, 4),
        "col_x_norm": round((c_idx + 0.5) / gw, 4),
        "horizontal_profile": horiz,
        "vertical_profile": vert,
    }


def _encode_crop_base64(
    img_bgr: Optional[np.ndarray],
    roi_bounds: Tuple[int, int, int, int],
    max_side: int = 240,
) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None
    H, W = img_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = roi_bounds
    x1, y1 = max(0, int(rx1)), max(0, int(ry1))
    x2, y2 = min(W, int(rx2)), min(H, int(ry2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = min(1.0, float(max_side) / float(max(ch, cw)))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_relative_surface_snapshot(
    *,
    snapshot_id: str,
    session_id: str,
    frame_idx: int,
    timestamp_sec: float,
    track_id: int,
    track_generation_id: int,
    depth_map: np.ndarray,
    extract_out: Dict[str, Any],
    img_bgr: Optional[np.ndarray] = None,
    model_identity: str = "depth_anything_v2_small",
    depth_input_resolution: Optional[int] = None,
    inference_precision: Optional[str] = None,
    quality_assessment: Optional[Dict[str, Any]] = None,
    max_grid_size: int = 48,
    include_source_crop: bool = True,
) -> Dict[str, Any]:
    """
    Construct a bounded 2.5D relative cavity surface snapshot from real depth/RANSAC arrays.
    """
    arrays = extract_out.get("arrays") or {}
    region = arrays.get("region")
    fit_out = arrays.get("fit_out")
    core_mask = arrays.get("core_mask")
    if region is None or fit_out is None or core_mask is None:
        raise ValueError("extract_out missing required 'arrays' (region, fit_out, core_mask)")

    rx1, ry1, rx2, ry2 = [int(v) for v in region["outer_xyxy"]]
    bx1, by1, bx2, by2 = [int(v) for v in region["box_xyxy"]]
    h_out, w_out = int(ry2 - ry1), int(rx2 - rx1)
    if h_out <= 0 or w_out <= 0:
        raise ValueError(f"Invalid outer ROI dimensions ({h_out}x{w_out})")

    depth_roi = depth_map[ry1:ry2, rx1:rx2].astype(np.float64)
    coeffs = fit_out.get("coeffs")
    if coeffs is not None and len(coeffs) == 3:
        inv_h = 2.0 / max(1.0, float(h_out - 1))
        inv_w = 2.0 / max(1.0, float(w_out - 1))
        cy_vec = (np.arange(h_out, dtype=np.float64) * inv_h - 1.0)[:, None]
        cx_vec = (np.arange(w_out, dtype=np.float64) * inv_w - 1.0)[None, :]
        ref_plane_roi = float(coeffs[0]) * cy_vec + float(coeffs[1]) * cx_vec + float(coeffs[2])
    else:
        # Fallback: place crater ref plane & fill surround with NaN so nothing is fabricated
        ref_plane_roi = np.full((h_out, w_out), np.nan, dtype=np.float64)
        ref_plane_roi[by1 - ry1 : by2 - ry1, bx1 - rx1 : bx2 - rx1] = fit_out["ref_plane_crater"]

    signed_residual_roi = ref_plane_roi - depth_roi
    validity_roi = np.isfinite(depth_roi) & np.isfinite(ref_plane_roi) & np.isfinite(signed_residual_roi)

    ref_ring_roi = np.zeros((h_out, w_out), dtype=bool)
    valid_ring = region.get("valid_ring_mask")
    if valid_ring is None:
        valid_ring = region.get("valid_ring")
    if valid_ring is not None and valid_ring.shape == (h_out, w_out):
        ref_ring_roi = valid_ring.astype(bool)

    cavity_roi = np.zeros((h_out, w_out), dtype=bool)
    ly0, ly1 = max(0, by1 - ry1), min(h_out, by2 - ry1)
    lx0, lx1 = max(0, bx1 - rx1), min(w_out, bx2 - rx1)
    ch, cw = max(0, ly1 - ly0), max(0, lx1 - lx0)
    if ch > 0 and cw > 0 and core_mask.shape[0] >= ch and core_mask.shape[1] >= cw:
        cavity_roi[ly0:ly1, lx0:lx1] = core_mask[:ch, :cw].astype(bool)

    # Adaptive grid sizing bounded by max_grid_size while preserving aspect ratio
    aspect = float(w_out) / float(max(1, h_out))
    if w_out >= h_out:
        gw = min(w_out, max(12, int(max_grid_size)))
        gh = min(h_out, max(10, int(round(gw / max(aspect, 0.25)))))
    else:
        gh = min(h_out, max(12, int(max_grid_size)))
        gw = min(w_out, max(10, int(round(gh * min(aspect, 4.0)))))
    gh = min(gh, int(max_grid_size))
    gw = min(gw, int(max_grid_size))

    residual_grid, valid_grid = _downsample_grid_preserve_nulls(signed_residual_roi, validity_roi, gh, gw)
    depth_grid, _ = _downsample_grid_preserve_nulls(depth_roi, validity_roi, gh, gw)
    plane_grid, _ = _downsample_grid_preserve_nulls(ref_plane_roi, np.isfinite(ref_plane_roi), gh, gw)
    cavity_grid = _downsample_bool_mask(cavity_roi, gh, gw)
    reference_grid = _downsample_bool_mask(ref_ring_roi, gh, gw)

    finite_vals = [
        residual_grid[r][c]
        for r in range(gh)
        for c in range(gw)
        if valid_grid[r][c] and residual_grid[r][c] is not None
    ]
    min_res = round(float(min(finite_vals)), 6) if finite_vals else None
    max_res = round(float(max(finite_vals)), 6) if finite_vals else None

    cav_vals = [
        residual_grid[r][c]
        for r in range(gh)
        for c in range(gw)
        if valid_grid[r][c] and cavity_grid[r][c] and residual_grid[r][c] is not None
    ]
    ref_vals = [
        residual_grid[r][c]
        for r in range(gh)
        for c in range(gw)
        if valid_grid[r][c] and reference_grid[r][c] and residual_grid[r][c] is not None
    ]
    if finite_vals:
        f_arr = np.asarray(finite_vals, dtype=np.float64)
        p5_res = round(float(np.percentile(f_arr, 5.0)), 6)
        med_res = round(float(np.median(f_arr)), 6)
        p95_res = round(float(np.percentile(f_arr, 95.0)), 6)
        std_res = round(float(np.std(f_arr)), 6)
    else:
        p5_res = med_res = p95_res = std_res = None

    residual_stats = {
        "valid_cell_count": len(finite_vals),
        "total_cell_count": int(gh * gw),
        "p5_residual": p5_res,
        "median_residual": med_res,
        "p95_residual": p95_res,
        "std_residual": std_res,
        "cavity_cell_count": len(cav_vals),
        "cavity_median_residual": round(float(np.median(cav_vals)), 6) if cav_vals else None,
        "reference_cell_count": len(ref_vals),
        "reference_median_residual": round(float(np.median(ref_vals)), 6) if ref_vals else None,
    }

    bbox_in_roi_norm = [
        round((bx1 - rx1) / float(w_out), 4),
        round((by1 - ry1) / float(h_out), 4),
        round((bx2 - rx1) / float(w_out), 4),
        round((by2 - ry1) / float(h_out), 4),
    ]

    crop_b64 = _encode_crop_base64(img_bgr, (rx1, ry1, rx2, ry2)) if include_source_crop else None
    cross_sections = compute_cross_section_profiles(residual_grid, valid_grid)

    return {
        "snapshot_id": snapshot_id,
        "session_id": session_id,
        "frame_idx": int(frame_idx),
        "timestamp_sec": round(float(timestamp_sec), 4),
        "track_id": int(track_id),
        "track_generation_id": int(track_generation_id),
        "box_xyxy": [float(v) for v in extract_out.get("box_xyxy", [bx1, by1, bx2, by2])],
        "roi_xyxy": [rx1, ry1, rx2, ry2],
        "bbox_in_roi_norm": bbox_in_roi_norm,
        "grid_shape": [gh, gw],
        "source_roi_shape": [h_out, w_out],
        "coordinate_system": "normalized_roi_image_plane",
        "residual_polarity": RESIDUAL_POLARITY_DOC,
        "residual_units": "uncalibrated_relative_inverse_depth_residual",
        "is_metric_3d": False,
        "min_residual": min_res,
        "max_residual": max_res,
        "residual_stats": residual_stats,
        "plane_representation": {
            "coeffs_normalized_roi": [float(c) for c in coeffs] if coeffs is not None else None,
            "inlier_ratio": fit_out.get("inlier_ratio"),
            "cond_number": fit_out.get("cond_number"),
            "plane_degenerate": bool(fit_out.get("plane_degenerate", False)),
        },
        "signed_residual_grid": residual_grid,
        "relative_depth_grid": depth_grid,
        "reference_plane_grid": plane_grid,
        "cavity_mask_grid": cavity_grid,
        "reference_mask_grid": reference_grid,
        "validity_mask_grid": valid_grid,
        "default_cross_sections": cross_sections,
        "source_crop_data_url": crop_b64,
        "model_metadata": {
            "depth_model_id": model_identity,
            "depth_input_resolution": depth_input_resolution,
            "inference_precision": inference_precision,
            "cavity_strategy": extract_out.get("strategy"),
        },
        "quality_assessment": quality_assessment,
        "scientific_disclaimer": (
            "Experimental 2.5D relative inverse-depth surface derived from local RANSAC "
            "reference-plane residuals in normalized image-plane coordinates. NOT a metric "
            "cm/mm point cloud or calibrated physical 3D reconstruction."
        ),
    }


class GeometrySnapshotManager:
    """Manages bounded per-track and per-job relative 3D surface snapshot artifacts."""

    def __init__(
        self,
        session_id: str,
        output_dir: Path,
        mode: str,
        stem: str,
        cfg: Optional[GeometrySnapshotConfig] = None,
    ) -> None:
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.stem = stem
        self.cfg = cfg or GeometrySnapshotConfig(enabled=True)
        self.geom_dir = self.output_dir / "geometry"
        self.per_track_counts: Dict[Tuple[int, int], int] = {}
        self.per_track_last_frame: Dict[Tuple[int, int], int] = {}
        self.snapshots_meta: List[Dict[str, Any]] = []
        self.last_snapshot_by_track: Dict[Tuple[int, int], Tuple[str, float]] = {}

    def maybe_capture(
        self,
        *,
        frame_idx: int,
        timestamp_sec: float,
        track_id: int,
        track_generation_id: int,
        depth_map: np.ndarray,
        extract_out: Dict[str, Any],
        img_bgr: Optional[np.ndarray],
        model_identity: str,
        depth_input_resolution: Optional[int],
        inference_precision: Optional[str],
        quality_assessment: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[float]]:
        if not self.cfg.enabled:
            return None, None
        if len(self.snapshots_meta) >= self.cfg.max_snapshots_per_job:
            return None, None

        tkey = (int(track_id), int(track_generation_id))
        if self.per_track_counts.get(tkey, 0) >= self.cfg.max_snapshots_per_track:
            return None, None

        # Space snapshots slightly across track lifetime unless it's the first valid frame
        last_f = self.per_track_last_frame.get(tkey)
        if last_f is not None and (frame_idx - last_f) < 3:
            return None, None

        qi = (quality_assessment or {}).get("quality_index")
        if self.cfg.min_quality_for_snapshot > 0.0 and (qi is None or qi < self.cfg.min_quality_for_snapshot):
            return None, None

        snap_id = f"snap_t{track_id}_g{track_generation_id}_f{frame_idx:06d}"
        if not _SAFE_SNAP_ID_RE.match(snap_id):
            return None, None

        self.geom_dir.mkdir(parents=True, exist_ok=True)
        snapshot = build_relative_surface_snapshot(
            snapshot_id=snap_id,
            session_id=self.session_id,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            track_id=track_id,
            track_generation_id=track_generation_id,
            depth_map=depth_map,
            extract_out=extract_out,
            img_bgr=img_bgr,
            model_identity=model_identity,
            depth_input_resolution=depth_input_resolution,
            inference_precision=inference_precision,
            quality_assessment=quality_assessment,
            max_grid_size=self.cfg.max_grid_size,
            include_source_crop=self.cfg.include_source_crop,
        )

        rel_file = f"geometry/{snap_id}.json"
        snap_path = self.output_dir / rel_file
        tmp = snap_path.with_name(f".{snap_path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(snapshot, allow_nan=False), encoding="utf-8")
        os.replace(tmp, snap_path)

        self.per_track_counts[tkey] = self.per_track_counts.get(tkey, 0) + 1
        self.per_track_last_frame[tkey] = int(frame_idx)
        self.last_snapshot_by_track[tkey] = (snap_id, float(timestamp_sec))

        self.snapshots_meta.append(
            {
                "snapshot_id": snap_id,
                "relative_path": rel_file,
                "frame_idx": int(frame_idx),
                "timestamp_sec": round(float(timestamp_sec), 4),
                "track_id": int(track_id),
                "track_generation_id": int(track_generation_id),
                "grid_shape": snapshot["grid_shape"],
                "min_residual": snapshot["min_residual"],
                "max_residual": snapshot["max_residual"],
                "quality_index": qi,
                "quality_category": (quality_assessment or {}).get("quality_category"),
                "depth_model_id": model_identity,
                "depth_input_resolution": depth_input_resolution,
                "size_bytes": int(snap_path.stat().st_size),
            }
        )
        return snap_id, float(timestamp_sec)

    def write_manifest(self) -> Optional[Path]:
        if not self.cfg.enabled:
            return None
        mins = [m["min_residual"] for m in self.snapshots_meta if m.get("min_residual") is not None]
        maxs = [m["max_residual"] for m in self.snapshots_meta if m.get("max_residual") is not None]
        manifest = {
            "session_id": self.session_id,
            "mode": self.mode,
            "geometry_enabled": True,
            "snapshot_count": len(self.snapshots_meta),
            "max_snapshots_per_track": self.cfg.max_snapshots_per_track,
            "max_snapshots_per_job": self.cfg.max_snapshots_per_job,
            "max_grid_size": self.cfg.max_grid_size,
            "residual_polarity": RESIDUAL_POLARITY_DOC,
            "residual_units": "uncalibrated_relative_inverse_depth_residual",
            "global_min_residual": round(float(min(mins)), 6) if mins else None,
            "global_max_residual": round(float(max(maxs)), 6) if maxs else None,
            "snapshots": self.snapshots_meta,
            "scientific_disclaimer": (
                "Experimental 2.5D relative cavity surfaces derived from FRESH_VALID "
                "inverse-depth observations; NOT metric physical 3D reconstructions."
            ),
        }
        m_path = self.output_dir / f"phase8a_{self.mode}_{self.stem}_geometry_manifest.json"
        tmp = m_path.with_name(f".{m_path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, m_path)
        return m_path
