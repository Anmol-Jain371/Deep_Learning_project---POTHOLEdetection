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
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        self.cfg = cfg
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self._frame_generation: int = 0

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
        logger.info(
            "RoadVisionV2Pipeline ready | session=%s mode=%s device=%s depth=%s",
            self.session_id, cfg.mode, self.device,
            self._depth_model_id if cfg.depth.enabled else "off",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        except Exception as exc:
            logger.error("Failed to load depth model: %s", exc)
            self._depth_model = None

    def _run_depth_inference(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """Run depth inference with failure isolation. Returns (depth_map, timings_ms)."""
        if self._depth_model is None:
            return None, {}
        try:
            import torch
            pred = self._depth_model.predict(img_bgr)
            dmap = pred.depth_map
            if not np.any(np.isfinite(dmap)):
                logger.warning("Depth prediction returned all non-finite values.")
                return None, pred.timings_ms
            return dmap, pred.timings_ms
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("GPU OOM during depth inference; skipping frame.")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                logger.error("Depth inference error: %s", exc)
            return None, {}
        except Exception as exc:
            logger.error("Unexpected depth error: %s", exc)
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
        )

    def _build_ref_diagnostics(self, extract_out: Dict[str, Any]) -> ReferenceFitDiagnostics:
        fo = extract_out.get("fit_out") or {}
        region = extract_out.get("region") or {}
        return ReferenceFitDiagnostics(
            used_ref_samples=fo.get("used_ref_samples"),
            available_ref_samples=region.get("available_ref_samples"),
            populated_quadrants=region.get("populated_quadrants"),
            inlier_ratio=fo.get("inlier_ratio"),
            ref_fit_mad=fo.get("ref_fit_mad"),
            ref_fit_rmse=fo.get("ref_fit_rmse"),
            annular_retention_ratio=region.get("annular_retention_ratio"),
            plane_degenerate=fo.get("plane_degenerate"),
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

        if was_keyframe and extract_out is not None:
            raw_score = extract_out.get("iqr_norm_p75") if is_valid else None
            norm_score = extract_out.get("scale_norm_p75") if is_valid else None
            is_water = bool(extract_out.get("is_water", False))
            ref_diag = self._build_ref_diagnostics(extract_out)
            if extract_out.get("warnings"):
                warnings = list(extract_out["warnings"])
            if not is_valid and not failure_reasons:
                failure_reasons = list(extract_out.get("reasons") or [])

        if state == "NOT_EVALUATED" and not self.cfg.depth.enabled:
            fallback_used = True
            fallback_reason = "depth_disabled"

        return RelativeDepthMeasurement(
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

        run_depth = bool(sched_dict.get("run_depth", False))

        # -------- Depth inference (keyframe) --------
        depth_map: Optional[np.ndarray] = None
        depth_timings: Dict[str, float] = {}
        if run_depth and self.cfg.depth.enabled:
            t2 = time.perf_counter()
            depth_map, depth_timings = self._run_depth_inference(img_bgr)
            elapsed_ms = (time.perf_counter() - t2) * 1000.0
            perf.depth_pre_ms = round(float(depth_timings.get("preprocess_ms", 0.0)), 2)
            perf.depth_inf_ms = round(float(depth_timings.get("inference_ms", elapsed_ms)), 2)
            perf.depth_resize_ms = round(float(depth_timings.get("resize_ms", 0.0)), 2)
            self._scheduler.update_observed_latency(elapsed_ms)

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
                    extract_out = {"valid": False, "reasons": [f"extraction_error:{type(exc).__name__}"]}

                valid_extract = bool(extract_out.get("valid", False))
                rel_score = extract_out.get("iqr_norm_p75") if valid_extract else None
                raw_p75 = extract_out.get("raw_p75") if valid_extract else None
                iqr_p75 = extract_out.get("iqr_norm_p75") if valid_extract else None
                ref_diags_dict = {}
                if "fit_out" in extract_out:
                    ref_diags_dict = dict(extract_out.get("fit_out") or {})
                failure_reasons = tuple(extract_out.get("reasons") or [])

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

            hazards.append(HazardObservation(
                frame_context=frame_ctx,
                detection=det,
                measurement=meas,
                horizon_diag=horizon_diag,
            ))

        perf.extract_ms = round((time.perf_counter() - t4) * 1000.0, 2)
        perf.total_frame_ms = round((time.perf_counter() - t_frame_start) * 1000.0, 2)

        return hazards, perf

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

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output writers
        vid_writer = None
        if save_video:
            vid_path = str(out_dir / f"phase8a_{mode_tag}_{stem}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vid_writer = cv2.VideoWriter(vid_path, fourcc, fps_src, (W, H))

        all_hazards: List[Dict[str, Any]] = []
        all_perfs: List[Dict[str, Any]] = []
        scheduling_log: List[Dict[str, Any]] = []

        frame_idx = 0
        t_pipeline_start = time.perf_counter()
        last_sched_dict: Dict[str, Any] = {}

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            ts_sec = float(frame_idx) / max(fps_src, 1.0)
            hazards, perf = self.process_frame(
                frame, frame_idx, ts_sec,
                camera_profile="sedan_dashcam",
            )

            if save_json:
                for hz in hazards:
                    all_hazards.append(hz.to_dict())
            if save_csv:
                all_perfs.append(perf.to_dict())

            # Annotate video
            if save_video and vid_writer is not None:
                ann = _annotate_frame(
                    frame, hazards, last_sched_dict, perf,
                    show_timing_hud=self.cfg.diagnostics.show_timing_hud,
                    show_rejection_reasons=self.cfg.diagnostics.show_rejection_reasons and diagnostics,
                )
                vid_writer.write(ann)

            frame_idx += 1

        cap.release()
        if vid_writer is not None:
            vid_writer.release()

        elapsed_total = time.perf_counter() - t_pipeline_start
        end_fps = frame_idx / max(elapsed_total, 1e-6)

        summary = {
            "session_id": self.session_id,
            "mode": self.cfg.mode,
            "source": str(source_path),
            "total_frames": frame_idx,
            "source_fps": fps_src,
            "elapsed_sec": round(elapsed_total, 3),
            "end_to_end_fps": round(end_fps, 2),
            "depth_enabled": self.cfg.depth.enabled,
            "depth_model_id": self._depth_model_id,
            "depth_input_size": self._depth_input_size,
            "scheduler_strategy": self.cfg.scheduler.strategy,
            "depth_calls": self._scheduler.depth_calls,
            "budget_skips": self._scheduler.budget_skips,
            "depth_invocation_rate": round(
                self._scheduler.depth_calls / max(frame_idx, 1), 4
            ),
            "unique_tracks": len(self._track_registry),
            "total_hazard_observations": len(all_hazards),
        }

        # --- Write outputs ---
        if save_json and all_hazards:
            jpath = out_dir / f"phase8a_{mode_tag}_{stem}_frames.json"
            jpath.write_text(json.dumps(all_hazards, indent=2), encoding="utf-8")
            logger.info("Wrote %d hazard records → %s", len(all_hazards), jpath)

        if save_json:
            # Per-track summary
            track_summary = {
                str(tid): obs.to_dict() for tid, obs in self._track_registry.items()
            }
            tp = out_dir / f"phase8a_{mode_tag}_{stem}_tracks.json"
            tp.write_text(json.dumps(track_summary, indent=2), encoding="utf-8")

        if save_csv and all_perfs:
            cp = out_dir / f"phase8a_{mode_tag}_{stem}_performance.csv"
            with open(cp, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(all_perfs[0].keys()))
                writer.writeheader()
                writer.writerows(all_perfs)

        # Session summary JSON always written
        sp = out_dir / f"phase8a_{mode_tag}_{stem}_session_summary.json"
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return summary

    def reset_session(self, reason: str = "user_request") -> None:
        """Reset per-session state (track registry, scheduler, cache) for a new video."""
        self._frame_generation += 1
        self._track_registry.clear()
        self._scheduler.reset_for_new_video(
            video_session_id=f"{self.session_id}_g{self._frame_generation}",
            reason=reason,
        )
        logger.info("Session reset | reason=%s generation=%d", reason, self._frame_generation)
