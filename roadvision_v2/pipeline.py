"""
========================================================================================
RoadVision AI 2.0 -- Modular V2 Pipeline (Phase 8A Task 6)
========================================================================================
Integrates validated research components (Phases 1–7B) into a fully isolated prototype
pipeline that runs strictly in parallel with the legacy `roadvision_engine.py`.

Design contracts:
  - ALL relative scores are uncalibrated inverse-depth residuals.
    Physical metric cm/mm are NEVER computed or emitted.
  - Invalid / unavailable measurements are `None` (never 0.0).
  - Cached reads are explicitly labelled CACHED_VALID, never FRESH_VALID.
  - GPU OOM and depth errors are isolated; the detection path always continues.
  - Missing GPS → has_gps=False, no crash.
  - Horizon diagnostics are optional research readouts only; they do NOT affect scores.
========================================================================================
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Local package imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_v2.config import (
    RoadVisionV2Config,
    CacheTemporalConfig,
    SchedulerConfig,
    RansacConfig,
    CavityConfig,
    DepthConfig,
    DetectorConfig,
)
from roadvision_v2.schemas import (
    MEASUREMENT_STATES,
    FrameContext,
    DetectionObservation,
    TrackObservation,
    ReferenceFitDiagnostics,
    SchedulingDecision,
    RelativeDepthMeasurement,
    HazardObservation,
    PipelinePerformance,
)
from roadvision_v2.quality import (
    assess_observation_quality,
    build_track_quality_summary,
    compute_track_temporal_stability,
    summarize_session_quality,
)
from roadvision_v2.geometry import GeometrySnapshotManager

logger = logging.getLogger("roadvision_v2.pipeline")


# ---------------------------------------------------------------------------
# Lazy experimental imports (errors isolated per-component)
# ---------------------------------------------------------------------------

def _import_depth_adapters():
    """Import depth adapters lazily to avoid side-effects at package import time."""
    from experiments.depth_model_comparison import (
        MiDaSSmallAdapter,
        DepthAnythingV2SmallAdapter,
    )
    return MiDaSSmallAdapter, DepthAnythingV2SmallAdapter


def _import_ransac():
    from experiments.optimized_relative_depth import (
        OptimizedDepthConfig,
        build_and_sample_annular_region,
        fit_ransac_optimized,
    )
    return OptimizedDepthConfig, build_and_sample_annular_region, fit_ransac_optimized


def _import_cavity():
    from experiments.core_aware_cavity_extraction import (
        CoreExtractionConfig,
        extract_cavity_with_strategy,
    )
    return CoreExtractionConfig, extract_cavity_with_strategy


def _import_scheduler():
    from experiments.adaptive_depth_scheduler import (
        AdaptiveDepthScheduler,
        AdaptiveSchedulerConfig,
    )
    return AdaptiveDepthScheduler, AdaptiveSchedulerConfig


def _import_cache():
    from experiments.track_depth_cache import (
        TrackDepthCache,
        TrackCacheConfig,
    )
    return TrackDepthCache, TrackCacheConfig


def _import_horizon():
    from experiments.dynamic_horizon_estimator import (
        DynamicHorizonTracker,
        HorizonEstimatorConfig,
    )
    return DynamicHorizonTracker, HorizonEstimatorConfig


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------
_STATE_COLORS = {
    "FRESH_VALID": (0, 220, 0),
    "FRESH_INVALID": (0, 100, 220),
    "CACHED_VALID": (220, 180, 0),
    "EXPIRED": (120, 120, 120),
    "NOT_EVALUATED": (60, 60, 60),
}


def _annotate_frame(
    img: np.ndarray,
    hazards: List[HazardObservation],
    sched_decision: Optional[Dict[str, Any]],
    perf: PipelinePerformance,
    show_timing_hud: bool = True,
    show_rejection_reasons: bool = False,
) -> np.ndarray:
    """Draw bounding boxes, relative scores, state badges, and optional HUD on a BGR frame."""
    out = img.copy()
    H, W = out.shape[:2]

    for hz in hazards:
        det = hz.detection
        meas = hz.measurement
        x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
        color = _STATE_COLORS.get(meas.measurement_state, (180, 180, 180))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Score label
        if meas.smoothed_relative_score is not None:
            score_str = f"T{det.track_id} | rel={meas.smoothed_relative_score:.3f} [{meas.measurement_state[:2]}]"
        elif meas.measurement_state == "NOT_EVALUATED":
            score_str = f"T{det.track_id} | depth off"
        else:
            score_str = f"T{det.track_id} | {meas.measurement_state}"

        cv2.putText(out, score_str, (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if show_rejection_reasons and meas.failure_reasons:
            reason_str = ",".join(meas.failure_reasons[:2])
            cv2.putText(out, reason_str, (x1, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 220), 1, cv2.LINE_AA)

    if show_timing_hud:
        lines = [
            f"Frame {perf.frame_idx:5d} | total={perf.total_frame_ms:5.1f}ms",
            f"det={perf.detect_track_ms:.1f} depth_inf={perf.depth_inf_ms:.1f}ms",
        ]
        if sched_decision:
            run_d = sched_decision.get("run_depth", False)
            reasons = sched_decision.get("frame_reasons", [])
            lines.append(f"depth={'YES' if run_d else 'skip'} | {','.join(reasons[:2])}")
        for i, line in enumerate(lines):
            cv2.putText(out, line, (8, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------

class RoadVisionV2Pipeline:
    """
    Modular V2 pothole detection pipeline.

    Operates in one of three modes:
      live     — detection + ByteTrack only; depth disabled unless explicitly enabled.
      balanced — YOLOv8n + ByteTrack + DAv2 Small 336 + budget/track-aware scheduler.
      survey   — Offline / high-detail: DAv2 Small 518 + candidate-gated depth + diagnostics.

    Relative cavity scores are uncalibrated inverse-depth residuals.
    Physical metric measurements are NEVER produced.
    """

    def __init__(
        self,
        cfg: RoadVisionV2Config,
        session_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        t_init_start = time.perf_counter()
        self.cfg = cfg
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self._frame_generation: int = 0
        self._depth_load_error: Optional[str] = None
        self._last_sched_decision: Dict[str, Any] = {}
        self._last_depth_attempt_ran: bool = False
        self._last_depth_attempt_ok: bool = False
        self._last_depth_error: Optional[str] = None

        # --- Detector + tracker ---
        from ultralytics import YOLO
        det_path = cfg.resolve_path(cfg.detector.checkpoint_path)
        logger.info("Loading V2 detector: %s", det_path)
        self._yolo = YOLO(det_path)
        self._tracker_config = cfg.resolve_path(cfg.detector.tracker_config)
        self._det_input_size = cfg.detector.input_size
        self._det_conf = cfg.detector.conf_threshold
        self._det_iou = cfg.detector.iou_threshold
        self._horizon_cutoff = cfg.detector.horizon_cutoff_ratio

        # --- Depth model ---
        self._depth_model = None
        self._depth_model_id: str = "none"
        self._depth_precision: str = "none"
        self._depth_input_size: int = 0
        if cfg.depth.enabled:
            self._load_depth_model(cfg.depth)

        # --- RANSAC & extraction configs ---
        OptimizedDepthConfig, _, _ = _import_ransac()
        CoreExtractionConfig, _ = _import_cavity()
        self._opt_cfg = OptimizedDepthConfig(
            sampling_mode=cfg.ransac.sampling_mode,
            max_ref_samples=cfg.ransac.max_ref_samples,
            ransac_min_iters=cfg.ransac.min_iters,
            ransac_max_iters=cfg.ransac.max_iters,
            ransac_batch_size=cfg.ransac.batch_size,
            ransac_confidence=cfg.ransac.confidence,
            enable_early_stop=cfg.ransac.enable_early_stop,
            seed=cfg.ransac.seed,
            min_ref_samples=cfg.ransac.min_ref_samples,
            min_quadrant_samples=cfg.ransac.min_quadrant_samples,
            min_populated_quadrants=cfg.ransac.min_populated_quadrants,
            max_single_quadrant_frac=cfg.ransac.max_single_quadrant_frac,
            min_annular_retention=cfg.ransac.min_annular_retention,
            horizon_cutoff_ratio=cfg.detector.horizon_cutoff_ratio,
        )
        self._core_cfg = CoreExtractionConfig(
            reject_water_patches=cfg.cavity.reject_water_patches,
        )
        self._cavity_strategy = cfg.cavity.strategy

        # --- Scheduler + cache ---
        AdaptiveDepthScheduler, AdaptiveSchedulerConfig = _import_scheduler()
        TrackDepthCache, TrackCacheConfig = _import_cache()

        cache_cfg = TrackCacheConfig(
            max_cache_age_sec=cfg.cache_temporal.max_cache_age_sec,
            max_invalid_stale_sec=cfg.cache_temporal.max_invalid_stale_sec,
            max_absence_reset_sec=cfg.cache_temporal.max_absence_reset_sec,
            max_reuse_centroid_norm=cfg.cache_temporal.max_reuse_centroid_norm,
            tau_sec=cfg.cache_temporal.tau_sec,
            rolling_window=cfg.cache_temporal.rolling_window,
            fixed_alpha=cfg.cache_temporal.fixed_alpha,
        )
        self._cache = TrackDepthCache(
            video_session_id=self.session_id,
            depth_model_id=self._depth_model_id,
            depth_input_size=self._depth_input_size,
            inference_precision=self._depth_precision,
            cfg=cache_cfg,
        )

        sched_cfg = AdaptiveSchedulerConfig(
            strategy=cfg.scheduler.strategy,
            fixed_interval_k=cfg.scheduler.fixed_interval_k,
            refresh_interval_sec=cfg.scheduler.refresh_interval_sec,
            max_measurement_age_sec=cfg.scheduler.max_measurement_age_sec,
            area_change_threshold=cfg.scheduler.area_change_threshold,
            centroid_shift_threshold=cfg.scheduler.centroid_shift_threshold,
            min_box_iou=cfg.scheduler.min_box_iou,
            invalid_retry_base_sec=cfg.scheduler.invalid_retry_base_sec,
            invalid_retry_max_sec=cfg.scheduler.invalid_retry_max_sec,
            min_ref_inlier_hold=cfg.scheduler.min_ref_inlier_hold,
            scene_change_l1_thresh=cfg.scheduler.scene_change_l1_thresh,
            budget_target_ema_ms=cfg.scheduler.budget_target_ema_ms,
            budget_min_gap_sec=cfg.scheduler.budget_min_gap_sec,
            starvation_override_sec=cfg.scheduler.starvation_override_sec,
        )
        self._scheduler = AdaptiveDepthScheduler(cfg=sched_cfg, cache=self._cache)

        # --- Optional horizon diagnostics ---
        self._horizon_tracker = None
        if cfg.diagnostics.enable_horizon_proxy:
            try:
                DynamicHorizonTracker, HorizonEstimatorConfig = _import_horizon()
                self._horizon_tracker = DynamicHorizonTracker(HorizonEstimatorConfig())
                logger.info("Horizon proxy diagnostics enabled (research only).")
            except Exception as exc:
                logger.warning("Horizon tracker unavailable: %s", exc)

        # --- Per-session track registry ---
        self._track_registry: Dict[int, TrackObservation] = {}
        self._track_obs_history: Dict[int, List[Dict[str, Any]]] = {}
        self._track_last_fresh_quality: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._geometry_manager: Optional[GeometrySnapshotManager] = None
        self._model_init_sec: float = round(time.perf_counter() - t_init_start, 4)
        logger.info(
            "RoadVisionV2Pipeline ready | session=%s mode=%s device=%s depth=%s init=%.3fs",
            self.session_id, cfg.mode, self.device,
            self._depth_model_id if cfg.depth.enabled else "off",
            self._model_init_sec,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_tracking_state(self) -> None:
        if not hasattr(self, "_track_obs_history"):
            self._track_obs_history = {}
        if not hasattr(self, "_track_last_fresh_quality"):
            self._track_last_fresh_quality = {}
        if not hasattr(self, "_geometry_manager"):
            self._geometry_manager = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_depth_model(self, dcfg: DepthConfig) -> None:
        MiDaSSmallAdapter, DepthAnythingV2SmallAdapter = _import_depth_adapters()
        try:
            if dcfg.model_id == "midas_small":
                use_fp16 = dcfg.precision == "fp16_autocast"
                self._depth_model = MiDaSSmallAdapter(device=self.device, use_fp16=use_fp16)
                self._depth_model_id = "midas_small"
                self._depth_precision = "fp16_autocast" if use_fp16 else "fp32"
                self._depth_input_size = 256
                logger.info("MiDaS Small loaded (fp16_autocast=%s)", use_fp16)
            elif dcfg.model_id == "depth_anything_v2_small":
                use_fp16 = dcfg.precision in ("fp16_autocast", "fp16_weights")
                adapter = DepthAnythingV2SmallAdapter(
                    device=self.device,
                    input_size=dcfg.input_size,
                    use_fp16=use_fp16,
                )
                # Apply FP16 weights conversion if requested (FastDAv2Runner pattern)
                if dcfg.precision == "fp16_weights":
                    import torch
                    adapter.model = adapter.model.half().eval()
                    logger.info("DAv2 model converted to FP16 weights")
                self._depth_model = adapter
                self._depth_model_id = "depth_anything_v2_small"
                self._depth_precision = dcfg.precision
                self._depth_input_size = dcfg.input_size
                logger.info("DAv2 Small loaded (input_size=%d, precision=%s)", dcfg.input_size, dcfg.precision)
            else:
                logger.warning("Unknown depth model_id '%s'; depth disabled.", dcfg.model_id)
                self._depth_load_error = f"unknown_depth_model:{dcfg.model_id}"
        except Exception as exc:
            logger.error("Failed to load depth model: %s", exc)
            self._depth_model = None
            self._depth_load_error = f"{type(exc).__name__}: {exc}"

    def _run_depth_inference(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """Run depth inference with failure isolation. Returns (depth_map, timings_ms)."""
        self._last_depth_error = None
        if self._depth_model is None:
            self._last_depth_error = self._depth_load_error or "depth_model_not_loaded"
            return None, {}
        try:
            import torch
            pred = self._depth_model.predict(img_bgr)
            dmap = pred.depth_map
            if not np.any(np.isfinite(dmap)):
                logger.warning("Depth prediction returned all non-finite values.")
                self._last_depth_error = "non_finite_depth_map"
                return None, pred.timings_ms
            return dmap, pred.timings_ms
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("GPU OOM during depth inference; skipping frame.")
                self._last_depth_error = "gpu_oom_depth_inference"
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                logger.error("Depth inference error: %s", exc)
                self._last_depth_error = f"{type(exc).__name__}: {exc}"
            return None, {}
        except Exception as exc:
            logger.error("Unexpected depth error: %s", exc)
            self._last_depth_error = f"{type(exc).__name__}: {exc}"
            return None, {}

    def _extract_relative_score(
        self,
        depth_map: np.ndarray,
        box_xyxy: Tuple[float, float, float, float],
        all_boxes: List[Tuple[float, float, float, float]],
        img_bgr: np.ndarray,
    ) -> Dict[str, Any]:
        """Call optimized RANSAC + cavity extraction. Returns the full output dict."""
        _, build_and_sample_annular_region, fit_ransac_optimized = _import_ransac()
        _, extract_cavity_with_strategy = _import_cavity()
        neighbor_boxes = [b for b in all_boxes if b != box_xyxy]
        return extract_cavity_with_strategy(
            depth_map=depth_map,
            box_xyxy=box_xyxy,
            strategy=self._cavity_strategy,
            neighbor_boxes=neighbor_boxes,
            img_bgr=img_bgr,
            opt_cfg=self._opt_cfg,
            core_cfg=self._core_cfg,
            return_arrays=bool(getattr(self.cfg, "geometry_snapshot", None) and self.cfg.geometry_snapshot.enabled),
        )

    def _build_ref_diagnostics(self, extract_out: Dict[str, Any]) -> ReferenceFitDiagnostics:
        rd = extract_out.get("ref_diagnostics") or {}
        fo = extract_out.get("fit_out") or {}
        region = extract_out.get("region") or {}
        return ReferenceFitDiagnostics(
            used_ref_samples=rd.get("used_ref_samples", fo.get("used_ref_samples")),
            available_ref_samples=rd.get("available_ref_samples", region.get("available_ref_samples")),
            populated_quadrants=rd.get("populated_quadrants", region.get("populated_quadrants")),
            inlier_ratio=rd.get("inlier_ratio", fo.get("inlier_ratio")),
            ref_fit_mad=rd.get("ref_fit_mad", fo.get("ref_fit_mad")),
            ref_fit_rmse=rd.get("ref_fit_rmse", fo.get("ref_fit_rmse")),
            annular_retention_ratio=rd.get("annular_retention_ratio", region.get("annular_retention_ratio")),
            plane_degenerate=rd.get("plane_degenerate", fo.get("plane_degenerate")),
        )

    def _build_measurement(
        self,
        frame_ctx: FrameContext,
        det: DetectionObservation,
        cache_out: Dict[str, Any],
        extract_out: Optional[Dict[str, Any]] = None,
        was_keyframe: bool = False,
    ) -> RelativeDepthMeasurement:
        """Assemble a RelativeDepthMeasurement from cache output + optional extraction output."""
        state = cache_out.get("state", "NOT_EVALUATED")
        is_valid = cache_out.get("is_valid_available", False)
        age_sec = cache_out.get("cache_age_sec")
        failure_reasons = list(cache_out.get("last_failure_reasons") or [])
        warnings: List[str] = []

        raw_score: Optional[float] = None
        norm_score: Optional[float] = None
        smoothed_score: Optional[float] = cache_out.get("ema_timeaware_score")
        is_water = False
        ref_diag = ReferenceFitDiagnostics()
        fallback_used = False
        fallback_reason: Optional[str] = None
        cavity_diag: Optional[Dict[str, Any]] = None
        truncated_edges: List[str] = []

        if was_keyframe and extract_out is not None:
            core = extract_out.get("core_cavity_stats") or {}
            raw_cand = core.get("p75_residual", extract_out.get("iqr_norm_p75"))
            norm_cand = core.get("scale_normalized_p75", extract_out.get("scale_norm_p75"))
            raw_score = raw_cand if is_valid else None
            norm_score = norm_cand if is_valid else None
            is_water = bool(
                extract_out.get("is_water_suspected", extract_out.get("is_water", False))
            )
            ref_diag = self._build_ref_diagnostics(extract_out)
            cavity_diag = {
                "cavity_sample_count": core.get("px"),
                "positive_frac": core.get("positive_frac"),
                "iqr_residual": core.get("iqr_residual"),
                "p75_residual": core.get("raw_diag_p75", core.get("p75_residual")),
            }
            if extract_out.get("warnings"):
                warnings = list(extract_out["warnings"])
                for w in warnings:
                    if isinstance(w, str) and w.startswith("boundary_truncated:"):
                        truncated_edges = [e for e in w.split(":", 1)[1].split("+") if e]
            if not is_valid and not failure_reasons:
                failure_reasons = list(
                    extract_out.get("failure_reasons") or extract_out.get("reasons") or []
                )

        depth_load_err = getattr(self, "_depth_load_error", None)
        if state == "NOT_EVALUATED" and not self.cfg.depth.enabled:
            fallback_used = True
            fallback_reason = "depth_disabled"
        elif state == "NOT_EVALUATED" and self.cfg.depth.enabled and self._depth_model is None:
            fallback_used = True
            fallback_reason = depth_load_err or "depth_model_unavailable"

        meas = RelativeDepthMeasurement(
            source_session_id=frame_ctx.source_session_id,
            frame_id=frame_ctx.frame_id,
            frame_idx=frame_ctx.frame_idx,
            timestamp_sec=frame_ctx.timestamp_sec,
            track_id=det.track_id,
            track_generation_id=det.track_generation_id,
            box_xyxy=det.box_xyxy,
            detector_confidence=det.detector_confidence,
            model_identity=self._depth_model_id,
            depth_input_resolution=self._depth_input_size if self.cfg.depth.enabled else None,
            inference_precision=self._depth_precision if self.cfg.depth.enabled else None,
            raw_relative_cavity_score=raw_score,
            normalized_relative_score=norm_score,
            smoothed_relative_score=smoothed_score,
            measurement_validity=is_valid,
            measurement_state=state,
            measurement_age_sec=age_sec,
            ref_diagnostics=ref_diag,
            failure_reasons=failure_reasons,
            warnings=warnings,
            is_water_suspected=is_water,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

        if self.cfg.quality.enabled:
            self._ensure_tracking_state()
            track_key = (det.track_id, det.track_generation_id)
            history = self._track_obs_history.get(det.track_id, [])
            temp_hist = history + [{"measurement": meas.to_dict()}]
            temp_stab = compute_track_temporal_stability(temp_hist, cfg=self.cfg.quality)
            src_fresh = self._track_last_fresh_quality.get(track_key)
            qa = assess_observation_quality(
                measurement_state=meas.measurement_state,
                measurement_validity=meas.measurement_validity,
                detector_confidence=det.detector_confidence,
                ref_diagnostics=ref_diag.to_dict() if was_keyframe else None,
                cavity_diagnostics=cavity_diag,
                temporal_stability=temp_stab,
                measurement_age_sec=meas.measurement_age_sec,
                is_water_suspected=is_water if was_keyframe else None,
                box_xyxy=det.box_xyxy,
                frame_wh=frame_ctx.frame_wh,
                truncated_edges=truncated_edges,
                failure_reasons=failure_reasons,
                warnings=warnings,
                fallback_reason=fallback_reason,
                source_fresh_quality=src_fresh,
                cfg=self.cfg.quality,
            )
            meas.quality_assessment = qa
            if meas.measurement_state == "FRESH_VALID":
                self._track_last_fresh_quality[track_key] = {
                    **qa,
                    "raw_ref_diagnostics": ref_diag.to_dict(),
                    "raw_cavity_diagnostics": cavity_diag,
                    "is_water_suspected": is_water,
                    "source_timestamp_sec": frame_ctx.timestamp_sec,
                }

        geom_mgr = getattr(self, "_geometry_manager", None)
        if geom_mgr is not None:
            tkey = (det.track_id, det.track_generation_id)
            if meas.measurement_state == "FRESH_VALID" and extract_out is not None and hasattr(self, "_current_depth_map"):
                try:
                    snap_id, snap_ts = geom_mgr.maybe_capture(
                        frame_idx=frame_ctx.frame_idx,
                        timestamp_sec=frame_ctx.timestamp_sec,
                        track_id=det.track_id,
                        track_generation_id=det.track_generation_id,
                        depth_map=self._current_depth_map,
                        extract_out=extract_out,
                        img_bgr=getattr(self, "_current_img_bgr", None),
                        model_identity=self._depth_model_id,
                        depth_input_resolution=meas.depth_input_resolution,
                        inference_precision=meas.inference_precision,
                        quality_assessment=meas.quality_assessment,
                    )
                    if snap_id:
                        meas.geometry_snapshot_id = snap_id
                        meas.geometry_snapshot_timestamp_sec = snap_ts
                except Exception as exc:
                    logger.warning("Geometry snapshot capture skipped for track %d: %s", det.track_id, exc)
            elif meas.measurement_state == "CACHED_VALID":
                prev_snap = geom_mgr.last_snapshot_by_track.get(tkey)
                if prev_snap:
                    meas.geometry_snapshot_id = prev_snap[0]
                    meas.geometry_snapshot_timestamp_sec = round(float(prev_snap[1]), 4)

        return meas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        img_bgr: np.ndarray,
        frame_idx: int,
        ts_sec: float,
        camera_profile: str = "sedan_dashcam",
        gps_lat: Optional[float] = None,
        gps_lon: Optional[float] = None,
    ) -> Tuple[List[HazardObservation], PipelinePerformance]:
        """
        Process a single BGR frame. Returns (hazards, performance).
        Never raises; failures are isolated within each sub-step.
        """
        t_frame_start = time.perf_counter()
        frame_id = f"{self.session_id}_f{frame_idx:06d}"
        H, W = img_bgr.shape[:2]
        has_gps = gps_lat is not None and gps_lon is not None

        frame_ctx = FrameContext(
            source_session_id=self.session_id,
            frame_id=frame_id,
            frame_idx=frame_idx,
            timestamp_sec=float(ts_sec),
            frame_wh=(W, H),
            mode=self.cfg.mode,
            camera_profile=camera_profile,
            gps_lat=gps_lat if has_gps else None,
            gps_lon=gps_lon if has_gps else None,
            has_gps=has_gps,
        )

        perf = PipelinePerformance(
            frame_id=frame_id,
            frame_idx=frame_idx,
            mode=self.cfg.mode,
        )

        hazards: List[HazardObservation] = []
        sched_dict: Dict[str, Any] = {}

        # -------- Detection + tracking --------
        t0 = time.perf_counter()
        try:
            results = self._yolo.track(
                img_bgr,
                persist=True,
                tracker=self._tracker_config,
                imgsz=self._det_input_size,
                conf=self._det_conf,
                iou=self._det_iou,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Detection error frame %d: %s", frame_idx, exc)
            perf.detect_track_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            perf.total_frame_ms = round((time.perf_counter() - t_frame_start) * 1000.0, 2)
            return hazards, perf
        perf.detect_track_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Parse detections
        horizon_y = int(H * self.cfg.detector.horizon_cutoff_ratio)
        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        track_ids: List[int] = []
        raw_dets: List[Tuple[Tuple, float, int, int]] = []  # (box, conf, cls_id, tid)

        res = results[0] if results else None
        if res is not None and res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)
            bxs = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for tid, bx, conf, cls in zip(ids, bxs, confs, clss):
                x1, y1, x2, y2 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])
                box = (x1, y1, x2, y2)
                below_horizon = y2 > horizon_y
                boxes_xyxy.append(box)
                track_ids.append(int(tid))
                raw_dets.append((box, float(conf), int(cls), int(tid), below_horizon))

        # -------- Scheduler decision --------
        t1 = time.perf_counter()
        try:
            sched_dict = self._scheduler.decide_frame(
                frame_idx=frame_idx,
                ts_sec=ts_sec,
                boxes_xyxy=boxes_xyxy,
                track_ids=track_ids,
                frame_bgr=img_bgr,
                camera_profile=camera_profile,
            )
        except Exception as exc:
            logger.error("Scheduler error frame %d: %s", frame_idx, exc)
            sched_dict = {
                "run_depth": False,
                "frame_reasons": ["SCHEDULER_ERROR"],
                "per_track_reasons": {},
                "shared_across_tracks": 0,
                "scheduler_ms": 0.0,
            }
        perf.scheduler_ms = round((time.perf_counter() - t1) * 1000.0, 2)
        self._last_sched_decision = sched_dict

        run_depth = bool(sched_dict.get("run_depth", False))
        self._last_depth_attempt_ran = False
        self._last_depth_attempt_ok = False

        # -------- Depth inference (keyframe) --------
        depth_map: Optional[np.ndarray] = None
        depth_timings: Dict[str, float] = {}
        if run_depth and self.cfg.depth.enabled:
            self._last_depth_attempt_ran = True
            t2 = time.perf_counter()
            depth_map, depth_timings = self._run_depth_inference(img_bgr)
            self._last_depth_attempt_ok = depth_map is not None
            elapsed_ms = (time.perf_counter() - t2) * 1000.0
            perf.depth_pre_ms = round(float(depth_timings.get("preprocess_ms", 0.0)), 2)
            perf.depth_inf_ms = round(float(depth_timings.get("inference_ms", elapsed_ms)), 2)
            perf.depth_resize_ms = round(float(depth_timings.get("resize_ms", 0.0)), 2)
            self._scheduler.update_observed_latency(elapsed_ms)

        self._current_depth_map = depth_map
        self._current_img_bgr = img_bgr

        # -------- Optional horizon diagnostics --------
        horizon_diag: Optional[Dict[str, Any]] = None
        if self.cfg.diagnostics.enable_horizon_proxy and self._horizon_tracker is not None:
            t3 = time.perf_counter()
            try:
                horizon_diag = self._horizon_tracker.update(img_bgr, ts_sec)
                if hasattr(horizon_diag, "to_dict"):
                    horizon_diag = horizon_diag.to_dict()
                elif not isinstance(horizon_diag, dict):
                    horizon_diag = None
            except Exception as exc:
                logger.debug("Horizon tracker error: %s", exc)
                horizon_diag = None
            perf.horizon_diag_ms = round((time.perf_counter() - t3) * 1000.0, 2)

        # -------- Per-track extraction + cache update --------
        t4 = time.perf_counter()
        gen_id = self._frame_generation
        for (box, conf, cls_id, tid, below_horizon) in raw_dets:
            det = DetectionObservation(
                track_id=tid,
                track_generation_id=gen_id,
                box_xyxy=box,
                detector_confidence=conf,
                class_id=cls_id,
                model_identity=self.cfg.detector.model_id,
                detector_input_size=self._det_input_size,
                is_below_horizon_cutoff=below_horizon,
            )

            # Update track registry
            existing = self._track_registry.get(tid)
            if existing is None:
                self._track_registry[tid] = TrackObservation(
                    source_session_id=self.session_id,
                    track_id=tid,
                    track_generation_id=gen_id,
                    first_seen_frame_idx=frame_idx,
                    last_seen_frame_idx=frame_idx,
                    observation_count=1,
                    box_xyxy=box,
                    latest_state="NOT_EVALUATED",
                )
            else:
                existing.last_seen_frame_idx = frame_idx
                existing.observation_count += 1
                existing.box_xyxy = box

            if run_depth and depth_map is not None and self.cfg.depth.enabled:
                # --- Keyframe: run extraction ---
                try:
                    extract_out = self._extract_relative_score(depth_map, box, boxes_xyxy, img_bgr)
                except Exception as exc:
                    logger.debug("Extraction error track %d: %s", tid, exc)
                    extract_out = {"is_valid": False, "failure_reasons": [f"extraction_error:{type(exc).__name__}"]}

                valid_extract = bool(extract_out.get("is_valid", extract_out.get("valid", False)))
                core = extract_out.get("core_cavity_stats") or {}
                raw_p75 = core.get("p75_residual", extract_out.get("raw_p75")) if valid_extract else None
                iqr_p75 = (
                    core.get("scale_normalized_p75", extract_out.get("iqr_norm_p75"))
                    if valid_extract
                    else None
                )
                rel_score = raw_p75 if raw_p75 is not None else iqr_p75
                ref_diags_dict = dict(
                    extract_out.get("ref_diagnostics") or extract_out.get("fit_out") or {}
                )
                failure_reasons = tuple(
                    extract_out.get("failure_reasons") or extract_out.get("reasons") or []
                )

                cache_out = self._cache.record_fresh_measurement(
                    track_id=tid,
                    frame_id=frame_id,
                    frame_idx=frame_idx,
                    ts_sec=ts_sec,
                    box_xyxy=box,
                    is_valid=valid_extract,
                    relative_score=rel_score,
                    raw_p75=raw_p75,
                    iqr_norm_p75=iqr_p75,
                    ref_diagnostics=ref_diags_dict,
                    failure_reasons=failure_reasons,
                )
                meas = self._build_measurement(
                    frame_ctx, det, cache_out,
                    extract_out=extract_out, was_keyframe=True,
                )
            else:
                # --- Non-keyframe: read cache ---
                cache_out = self._cache.read_non_keyframe_state(tid, frame_idx, ts_sec)
                meas = self._build_measurement(
                    frame_ctx, det, cache_out,
                    extract_out=None, was_keyframe=False,
                )

            if self._track_registry.get(tid):
                self._track_registry[tid].latest_state = meas.measurement_state

            hz_obs = HazardObservation(
                frame_context=frame_ctx,
                detection=det,
                measurement=meas,
                horizon_diag=horizon_diag,
            )
            self._ensure_tracking_state()
            self._track_obs_history.setdefault(tid, []).append(hz_obs.to_dict())
            if self.cfg.quality.enabled and self._track_registry.get(tid):
                self._track_registry[tid].quality_summary = build_track_quality_summary(
                    self._track_obs_history[tid], cfg=self.cfg.quality
                )
            hazards.append(hz_obs)

        perf.extract_ms = round((time.perf_counter() - t4) * 1000.0, 2)
        perf.total_frame_ms = round((time.perf_counter() - t_frame_start) * 1000.0, 2)
        self._current_depth_map = None
        self._current_img_bgr = None

        return hazards, perf

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _transcode_video_h264(vid_path: Path) -> Tuple[bool, str, str]:
        """
        Attempt browser-compatible H.264 transcode using bundled imageio-ffmpeg.
        Returns (success, codec_name, note).
        """
        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            tmp_h264 = vid_path.with_name(f".{vid_path.stem}_h264_{uuid.uuid4().hex[:6]}.mp4")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(vid_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_h264),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
            if res.returncode == 0 and tmp_h264.exists() and tmp_h264.stat().st_size > 256:
                cap_chk = cv2.VideoCapture(str(tmp_h264))
                ok_open = cap_chk.isOpened()
                ok_read, _ = cap_chk.read() if ok_open else (False, None)
                cap_chk.release()
                if ok_open and ok_read:
                    os.replace(tmp_h264, vid_path)
                    return True, "h264", "libx264_yuv420p_faststart"
            if tmp_h264.exists():
                tmp_h264.unlink(missing_ok=True)
            return False, "mp4v", "h264_transcode_unverified"
        except Exception as exc:
            return False, "mp4v", f"h264_transcode_skipped:{type(exc).__name__}"

    def process_video(
        self,
        source: str,
        output_dir: str = "runs/v2_outputs",
        save_video: bool = False,
        save_json: bool = True,
        save_csv: bool = False,
        diagnostics: bool = False,
        max_frames: Optional[int] = None,
        stem_override: Optional[str] = None,
        on_init: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        transcode_h264: bool = False,
        write_empty_frames_json: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a video file end-to-end.
        Returns a session summary dict.
        """
        source_path = Path(source)
        stem = stem_override or source_path.stem
        mode_tag = self.cfg.mode
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_tracking_state()
        geom_cfg = getattr(self.cfg, "geometry_snapshot", None)
        if geom_cfg is not None and geom_cfg.enabled:
            self._geometry_manager = GeometrySnapshotManager(
                session_id=self.session_id,
                output_dir=out_dir,
                mode=mode_tag,
                stem=stem,
                cfg=geom_cfg,
            )
        else:
            self._geometry_manager = None

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        vid_writer = None
        vid_path: Optional[Path] = None
        vid_tmp_path: Optional[Path] = None

        all_hazards: List[Dict[str, Any]] = []
        all_perfs: List[Dict[str, Any]] = []
        state_counts: Dict[str, int] = {s: 0 for s in MEASUREMENT_STATES}
        actual_depth_calls = 0
        failed_depth_inferences = 0
        warnings_emitted: List[str] = []
        frame_idx = 0
        cancelled = False
        depth_load_err = getattr(self, "_depth_load_error", None)
        model_init_sec = getattr(self, "_model_init_sec", None)

        # Reset CUDA peak stats before video loop for clean job-scoped telemetry
        cuda_stats_available = False
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    cuda_stats_available = True
            except Exception:
                cuda_stats_available = False

        t_pipeline_start = time.perf_counter()
        fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps_src <= 0.0 or fps_src > 240.0:
            fps_src = 25.0
        raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        reliable_total_frames: Optional[int] = raw_total_frames if raw_total_frames > 0 else None
        if reliable_total_frames is not None and max_frames is not None:
            reliable_total_frames = min(reliable_total_frames, int(max_frames))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            if depth_load_err:
                msg = f"Depth model unavailable; explicit fallback active ({depth_load_err})"
                warnings_emitted.append(msg)
                if on_warning is not None:
                    on_warning(msg)

            if save_video:
                vid_path = out_dir / f"phase8a_{mode_tag}_{stem}_annotated.mp4"
                vid_tmp_path = out_dir / f".phase8a_{mode_tag}_{stem}_annotated_{uuid.uuid4().hex[:6]}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid_writer = cv2.VideoWriter(str(vid_tmp_path), fourcc, fps_src, (W, H))

            if on_init is not None:
                on_init({
                    "total_frames": reliable_total_frames,
                    "source_fps": fps_src,
                    "width": W,
                    "height": H,
                })

            while True:
                if should_cancel is not None and should_cancel():
                    cancelled = True
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                ts_sec = float(frame_idx) / max(fps_src, 1.0)
                hazards, perf = self.process_frame(
                    frame, frame_idx, ts_sec,
                    camera_profile="sedan_dashcam",
                )

                if getattr(self, "_last_depth_attempt_ran", False):
                    if getattr(self, "_last_depth_attempt_ok", False):
                        actual_depth_calls += 1
                    else:
                        failed_depth_inferences += 1
                        last_d_err = getattr(self, "_last_depth_error", None)
                        if last_d_err and on_warning is not None:
                            on_warning(f"Depth inference failed on frame {frame_idx}: {last_d_err}")

                for hz in hazards:
                    st = hz.measurement.measurement_state
                    state_counts[st] = state_counts.get(st, 0) + 1
                    if save_json:
                        all_hazards.append(hz.to_dict())

                if save_csv:
                    all_perfs.append(perf.to_dict())

                if save_video and vid_writer is not None:
                    ann = _annotate_frame(
                        frame, hazards, getattr(self, "_last_sched_decision", {}), perf,
                        show_timing_hud=self.cfg.diagnostics.show_timing_hud,
                        show_rejection_reasons=self.cfg.diagnostics.show_rejection_reasons and diagnostics,
                    )
                    vid_writer.write(ann)

                frame_idx += 1

                if on_progress is not None:
                    elapsed_now = max(time.perf_counter() - t_pipeline_start, 1e-6)
                    on_progress({
                        "processed_frames": frame_idx,
                        "total_frames": reliable_total_frames,
                        "detection_observation_count": len(all_hazards) if save_json else sum(state_counts.values()),
                        "unique_candidate_track_count": len(self._track_registry),
                        "depth_invocation_count": actual_depth_calls,
                        "scheduler_request_count": self._scheduler.depth_calls,
                        "failed_depth_count": failed_depth_inferences,
                        "measurement_state_counts": dict(state_counts),
                        "throughput_fps": round(frame_idx / elapsed_now, 2),
                    })
        finally:
            cap.release()
            if vid_writer is not None:
                vid_writer.release()

        elapsed_processing = time.perf_counter() - t_pipeline_start
        end_fps = frame_idx / max(elapsed_processing, 1e-6)

        peak_alloc_mb: Optional[float] = None
        peak_reserved_mb: Optional[float] = None
        if cuda_stats_available:
            try:
                import torch
                peak_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
                peak_reserved_mb = round(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0), 2)
            except Exception:
                pass

        t_artifact_start = time.perf_counter()
        video_codec: Optional[str] = None
        video_browser_compatible: Optional[bool] = None
        video_transcode_note: Optional[str] = None
        if save_video and vid_path is not None and vid_tmp_path is not None and vid_tmp_path.exists():
            if frame_idx > 0:
                os.replace(vid_tmp_path, vid_path)
                video_codec = "mp4v"
                video_browser_compatible = False
                if transcode_h264 and not cancelled:
                    ok_h264, codec_name, note = self._transcode_video_h264(vid_path)
                    video_codec = codec_name
                    video_browser_compatible = ok_h264
                    video_transcode_note = note
                else:
                    # Verify OpenCV readability of generated MP4
                    cap_v = cv2.VideoCapture(str(vid_path))
                    if cap_v.isOpened():
                        cap_v.release()
                    else:
                        video_transcode_note = "opencv_video_unreadable"
            else:
                vid_tmp_path.unlink(missing_ok=True)

        self._ensure_tracking_state()
        total_obs_count = sum(state_counts.values())
        all_obs_for_summary = all_hazards if all_hazards else [
            obs for obs_list in self._track_obs_history.values() for obs in obs_list
        ]
        track_q_summaries = [
            t.quality_summary for t in self._track_registry.values() if t.quality_summary
        ]
        session_quality_summary = (
            summarize_session_quality(all_obs_for_summary, track_q_summaries)
            if self.cfg.quality.enabled
            else None
        )
        summary = {
            "session_id": self.session_id,
            "mode": self.cfg.mode,
            "source": str(source_path),
            "total_frames": frame_idx,
            "source_fps": fps_src,
            "elapsed_sec": round(elapsed_processing, 3),
            "end_to_end_fps": round(end_fps, 2),
            "depth_enabled": self.cfg.depth.enabled,
            "depth_model_id": self._depth_model_id,
            "depth_input_size": self._depth_input_size,
            "scheduler_strategy": self.cfg.scheduler.strategy,
            "scheduler_depth_requests": self._scheduler.depth_calls,
            "depth_calls": actual_depth_calls,
            "failed_depth_inferences": failed_depth_inferences,
            "budget_skips": self._scheduler.budget_skips,
            "depth_invocation_rate": round(
                actual_depth_calls / max(frame_idx, 1), 4
            ),
            "unique_tracks": len(self._track_registry),
            "total_hazard_observations": total_obs_count,
            "measurement_state_counts": dict(state_counts),
            "cancelled": cancelled,
            "fallback_used": bool(depth_load_err),
            "fallback_reason": depth_load_err,
            "model_init_sec": model_init_sec,
            "peak_allocated_gpu_mb": peak_alloc_mb,
            "peak_reserved_gpu_mb": peak_reserved_mb,
            "video_codec": video_codec,
            "video_browser_compatible": video_browser_compatible,
            "video_transcode_note": video_transcode_note,
            "quality_summary": session_quality_summary,
            "geometry_recording_enabled": bool(self._geometry_manager is not None),
            "geometry_snapshot_count": len(self._geometry_manager.snapshots_meta) if self._geometry_manager is not None else 0,
        }

        if self._geometry_manager is not None:
            try:
                self._geometry_manager.write_manifest()
            except Exception as exc:
                logger.warning("Failed to write geometry manifest: %s", exc)

        # --- Write outputs atomically ---
        if save_json and (all_hazards or write_empty_frames_json):
            jpath = out_dir / f"phase8a_{mode_tag}_{stem}_frames.json"
            self._atomic_write_text(jpath, json.dumps(all_hazards, indent=2, allow_nan=False))
            logger.info("Wrote %d hazard records → %s", len(all_hazards), jpath)

        if save_json:
            track_summary = {
                str(tid): obs.to_dict() for tid, obs in self._track_registry.items()
            }
            tp = out_dir / f"phase8a_{mode_tag}_{stem}_tracks.json"
            self._atomic_write_text(tp, json.dumps(track_summary, indent=2, allow_nan=False))

        if save_csv and all_perfs:
            cp = out_dir / f"phase8a_{mode_tag}_{stem}_performance.csv"
            tmp_cp = cp.with_name(f".{cp.name}.{uuid.uuid4().hex[:8]}.tmp")
            with open(tmp_cp, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(all_perfs[0].keys()))
                writer.writeheader()
                writer.writerows(all_perfs)
            os.replace(tmp_cp, cp)

        summary["artifact_generation_sec"] = round(time.perf_counter() - t_artifact_start, 4)

        sp = out_dir / f"phase8a_{mode_tag}_{stem}_session_summary.json"
        self._atomic_write_text(sp, json.dumps(summary, indent=2, allow_nan=False))

        if on_complete is not None and not cancelled:
            on_complete(summary)

        return summary

    def reset_session(self, reason: str = "user_request") -> None:
        """Reset per-session state (track registry, scheduler, cache) for a new video."""
        self._frame_generation += 1
        self._track_registry.clear()
        self._ensure_tracking_state()
        self._track_obs_history.clear()
        self._track_last_fresh_quality.clear()
        self._geometry_manager = None
        self._scheduler.reset_for_new_video(
            video_session_id=f"{self.session_id}_g{self._frame_generation}",
            reason=reason,
        )
        logger.info("Session reset | reason=%s generation=%d", reason, self._frame_generation)
