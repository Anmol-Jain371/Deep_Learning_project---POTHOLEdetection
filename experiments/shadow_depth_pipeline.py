"""
========================================================================================
RoadVision AI -- Phase 5C: Isolated Shadow-Mode Relative Depth Pipeline
========================================================================================
Experimental research orchestrator evaluating existing production extraction alongside
Phase 5B Optimized Adaptive RANSAC + Validity-Aware Temporal Filtering in parallel shadow
mode on identical frames, detector proposals, track IDs, camera profiles, timestamps, and
cached MiDaS inverse-depth predictions.

STRICT ISOLATION & SCIENTIFIC CONSTRAINTS:
  - Does NOT modify `roadvision_engine.py`, `server.py`, dashboard files, or checkpoints.
  - Experimental methods (B, C, D) NEVER emit cm, mm, liters, or physical severity labels.
  - Unavailable or expired experimental values remain explicitly `None` (`null` in JSON) —
    never silently clamped to zero.
  - Failures inside any experimental method are caught and returned as `NOT_EVALUATED`
    so neither legacy evaluation nor sibling methods are interrupted.
========================================================================================
"""

import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable

import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from roadvision_engine import (
    AnnularDonutDepthExtractor,
    MetricDepthCalibrator,
    SpecularAquaplaneDetector,
    TemporalEMASmoother,
    load_camera_config,
)
from experiments.optimized_relative_depth import (
    OptimizedDepthConfig,
    extract_optimized_relative_depth_for_box,
)
from experiments.validity_aware_temporal_filter import (
    TemporalFilterConfig,
    ValidityAwareTrackFilter,
)

MEASUREMENT_STATES = ("VALID", "INVALID", "STALE", "EXPIRED", "NOT_EVALUATED")
RELATIVE_POLARITY = "ref_minus_midas_inv_depth_positive_deeper"


@dataclass
class ShadowMeasurementResult:
    """Structured measurement result contract for shadow-mode A/B evaluation."""
    method_id: str
    frame_id: str
    frame_idx: int
    timestamp_sec: float
    track_id: Optional[int]
    box_xyxy: List[float]
    detector_conf: float
    camera_profile: str
    depth_preprocessing: str
    relative_depth_polarity: str
    raw_relative_p75: Optional[float]
    normalized_relative_p75: Optional[float]
    smoothed_relative_score: Optional[float]
    is_valid: bool
    status: str
    failure_reasons: List[str]
    ref_sample_count: Optional[int]
    available_ref_samples: Optional[int]
    ransac_inlier_ratio: Optional[float]
    ref_mad: Optional[float]
    spatial_support_quadrants: Optional[int]
    track_age_frames: int
    track_age_sec: float
    last_valid_timestamp_sec: Optional[float]
    stale_age_frames: Optional[int]
    water_glare_warning: bool
    extraction_latency_ms: float
    filtering_latency_ms: float
    adaptive_budget_used: Optional[int] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SharedMiDaSRunner:
    """
    MiDaS_small wrapper exposing explicit `rgb_corrected` vs `bgr_legacy` preprocessing
    modes with separate preprocessing vs GPU inference timing for Task 3 ablation.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        torch.hub._check_repo_is_trusted = lambda *a, **k: True
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def _sync(self) -> None:
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()

    def warmup(self, h: int = 360, w: int = 640) -> None:
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        _ = self.predict(dummy, preprocess_mode="rgb_corrected")

    def predict(
        self,
        img_bgr: np.ndarray,
        preprocess_mode: str = "rgb_corrected",
    ) -> Dict[str, Any]:
        if preprocess_mode not in ("rgb_corrected", "bgr_legacy"):
            raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

        H, W = img_bgr.shape[:2]
        t_pre0 = time.perf_counter()
        img_for_net = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if preprocess_mode == "rgb_corrected" else img_bgr
        inp = self.transform(img_for_net).to(self.device)
        t_pre1 = time.perf_counter()

        self._sync()
        t_inf0 = time.perf_counter()
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = self.model(inp)
            else:
                pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze()
        self._sync()
        t_inf1 = time.perf_counter()

        depth_map = pred.cpu().numpy().astype(np.float32)
        return {
            "depth_map": depth_map,
            "preprocess_mode": preprocess_mode,
            "preprocess_ms": round((t_pre1 - t_pre0) * 1000.0, 4),
            "inference_ms": round((t_inf1 - t_inf0) * 1000.0, 4),
        }


class LegacyMethodRunner:
    """Encapsulates production `AnnularDonutDepthExtractor` + `TemporalEMASmoother` behavior."""

    def __init__(self, camera_profile: str = "sedan_dashcam"):
        self.camera_profile = camera_profile
        self._cfg_dict = load_camera_config(profile=camera_profile)
        self._cur_hw: Tuple[int, int] = (360, 640)
        self.calibrator = self._build_calibrator(self._cur_hw[0], self._cur_hw[1])
        self.smoother = TemporalEMASmoother(history_len=10)
        self._first_ts: Dict[int, float] = {}
        self._first_frame: Dict[int, int] = {}
        self._last_non_zero_ts: Dict[int, float] = {}

    def _build_calibrator(self, H: int, W: int) -> MetricDepthCalibrator:
        c = self._cfg_dict
        return MetricDepthCalibrator(
            frame_height=H,
            frame_width=W,
            camera_height_m=c.get("camera_height_m", 1.25),
            vfov_deg=c.get("vfov_deg", 60.0),
            horizon_ratio=c.get("horizon_ratio", 0.40),
            k_disp=c.get("k_disp", 0.082),
            profile_name=c.get("profile_name", self.camera_profile),
            depth_exponent=c.get("depth_exponent", 1.35),
            length_exponent=c.get("length_exponent", 1.15),
            max_depth_clip_cm=c.get("max_depth_clip_cm", 35.0),
        )

    def set_camera_profile(self, camera_profile: str) -> None:
        if camera_profile != self.camera_profile:
            self.camera_profile = camera_profile
            self._cfg_dict = load_camera_config(profile=camera_profile)
            self.calibrator = self._build_calibrator(self._cur_hw[0], self._cur_hw[1])

    def ensure_resolution(self, H: int, W: int) -> None:
        if (H, W) != self._cur_hw:
            self._cur_hw = (H, W)
            self.calibrator = self._build_calibrator(H, W)

    def reset_state(self) -> None:
        self.smoother = TemporalEMASmoother(history_len=10)
        self._first_ts.clear()
        self._first_frame.clear()
        self._last_non_zero_ts.clear()

    def evaluate_box(
        self,
        frame_id: str,
        frame_idx: int,
        timestamp_sec: float,
        track_id: int,
        box_xyxy: Tuple[float, float, float, float],
        conf: float,
        is_water: bool,
        depth_map: Optional[np.ndarray],
        preprocess_mode: str,
    ) -> ShadowMeasurementResult:
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        t0 = time.perf_counter()
        if depth_map is None:
            raw_disp = 0.0
            missing_depth = True
        else:
            missing_depth = False
            raw_disp = float(AnnularDonutDepthExtractor.extract_depth_disparity(depth_map, x1, y1, x2, y2, margin=25))
        t_ext = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        # Production smoother converts raw disparity via calibrator -> EMA (`smooth_cm`)
        cal_cm = self.calibrator.disparity_to_depth_cm(raw_disp, y2)
        pw_cm, pl_cm = self.calibrator.calculate_pothole_dimensions_cm(max(1, x2 - x1), max(1, y2 - y1), y2)
        haz = self.smoother.update(
            track_id, cal_cm, y2, is_water, float(conf), width_cm=pw_cm, length_cm=pl_cm, timestamp=timestamp_sec
        )
        t_flt = (time.perf_counter() - t1) * 1000.0

        if track_id not in self._first_frame:
            self._first_frame[track_id] = frame_idx
            self._first_ts[track_id] = float(timestamp_sec)

        legacy_pos = raw_disp > 0.0 and not missing_depth
        if legacy_pos:
            self._last_non_zero_ts[track_id] = float(timestamp_sec)

        return ShadowMeasurementResult(
            method_id="A_legacy_annular_ema",
            frame_id=frame_id,
            frame_idx=frame_idx,
            timestamp_sec=round(float(timestamp_sec), 4),
            track_id=track_id,
            box_xyxy=[float(v) for v in box_xyxy],
            detector_conf=round(float(conf), 4),
            camera_profile=self.camera_profile,
            depth_preprocessing=preprocess_mode,
            relative_depth_polarity="legacy_annular_p85_clamped_ge0",
            raw_relative_p75=round(raw_disp, 6),  # Legacy returns 0.0 clamp explicitly
            normalized_relative_p75=None,
            smoothed_relative_score=round(float(haz["smooth_cm"]), 4),
            is_valid=legacy_pos,
            status="VALID" if legacy_pos else "INVALID",
            failure_reasons=(["missing_depth"] if missing_depth else ([] if legacy_pos else ["legacy_zero_clamped"])),
            ref_sample_count=None,
            available_ref_samples=None,
            ransac_inlier_ratio=None,
            ref_mad=None,
            spatial_support_quadrants=None,
            track_age_frames=frame_idx - self._first_frame[track_id],
            track_age_sec=round(float(timestamp_sec) - self._first_ts[track_id], 4),
            last_valid_timestamp_sec=self._last_non_zero_ts.get(track_id),
            stale_age_frames=0 if legacy_pos else 1,
            water_glare_warning=bool(is_water),
            extraction_latency_ms=round(t_ext, 4),
            filtering_latency_ms=round(t_flt, 4),
            adaptive_budget_used=None,
        )


def should_escalate_budget(out_512: Dict[str, Any]) -> bool:
    """
    Task 7 objective diagnostic trigger (based strictly on reference-fit & support stats,
    NOT target P75 magnitude alone): escalate 512 -> 1024 only when consensus fraction is
    borderline (<0.65) or sampled ref MAD is unstable despite ample full-ring pixels (>1024).
    """
    rd = out_512["ref_diagnostics"]
    n_avail = rd.get("available_ref_samples", 0) or 0
    if n_avail <= 1024 or rd["plane_degenerate"]:
        return False
    return (rd["inlier_ratio"] < 0.65) or (rd["populated_quadrants"] < 3)


class ShadowDepthPipeline:
    """
    Runs Method A (legacy baseline), Method B (Optimized RANSAC + validity EMA),
    Method C (Optimized RANSAC + rolling-median validity EMA), and optional adaptive
    Method D (`512->1024` trigger) on identical shared detector/MiDaS frame inputs.
    """

    def __init__(
        self,
        camera_profile: str = "sedan_dashcam",
        opt_cfg: Optional[OptimizedDepthConfig] = None,
        filter_cfg: Optional[TemporalFilterConfig] = None,
        enable_adaptive_variant: bool = False,
        preprocess_mode: str = "rgb_corrected",
    ):
        self.camera_profile = camera_profile
        self.opt_cfg = opt_cfg or OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
        self.filter_cfg = filter_cfg or TemporalFilterConfig(alpha=0.35, rolling_window=5, max_stale_frames=3, max_missed_frames_reset=8)
        self.enable_adaptive_variant = enable_adaptive_variant
        self.preprocess_mode = preprocess_mode

        self.legacy_runner = LegacyMethodRunner(camera_profile=camera_profile)
        self.filter_b = ValidityAwareTrackFilter(mode="validity_aware_ema", cfg=self.filter_cfg)
        self.filter_c = ValidityAwareTrackFilter(mode="rolling_median_validity_ema", cfg=self.filter_cfg)
        self.filter_d = ValidityAwareTrackFilter(mode="rolling_median_validity_ema", cfg=self.filter_cfg) if enable_adaptive_variant else None

        # Optional test fault injector hooks for failure-isolation verification
        self.fault_hooks: Dict[str, Callable[[], None]] = {}

    def set_camera_profile(self, camera_profile: str) -> None:
        self.camera_profile = camera_profile
        self.legacy_runner.set_camera_profile(camera_profile)

    def reset_all_tracks(self) -> None:
        self.legacy_runner.reset_state()
        self.filter_b.reset_all()
        self.filter_c.reset_all()
        if self.filter_d is not None:
            self.filter_d.reset_all()

    def _build_exp_result(
        self,
        method_id: str,
        frame_id: str,
        frame_idx: int,
        timestamp_sec: float,
        track_id: int,
        box_xyxy: Tuple[float, float, float, float],
        conf: float,
        is_water: bool,
        depth_missing: bool,
        out_ext: Optional[Dict[str, Any]],
        flt_res: Optional[Dict[str, Any]],
        ext_ms: float,
        flt_ms: float,
        budget_used: Optional[int],
        err_msg: Optional[str] = None,
    ) -> ShadowMeasurementResult:
        if err_msg is not None or out_ext is None or flt_res is None:
            return ShadowMeasurementResult(
                method_id=method_id,
                frame_id=frame_id,
                frame_idx=frame_idx,
                timestamp_sec=round(float(timestamp_sec), 4),
                track_id=track_id,
                box_xyxy=[float(v) for v in box_xyxy],
                detector_conf=round(float(conf), 4),
                camera_profile=self.camera_profile,
                depth_preprocessing=self.preprocess_mode,
                relative_depth_polarity=RELATIVE_POLARITY,
                raw_relative_p75=None,
                normalized_relative_p75=None,
                smoothed_relative_score=None,
                is_valid=False,
                status="NOT_EVALUATED",
                failure_reasons=["missing_depth"] if depth_missing else ["experimental_exception"],
                ref_sample_count=None,
                available_ref_samples=None,
                ransac_inlier_ratio=None,
                ref_mad=None,
                spatial_support_quadrants=None,
                track_age_frames=0,
                track_age_sec=0.0,
                last_valid_timestamp_sec=None,
                stale_age_frames=None,
                water_glare_warning=bool(is_water),
                extraction_latency_ms=round(ext_ms, 4),
                filtering_latency_ms=round(flt_ms, 4),
                adaptive_budget_used=budget_used,
                error_message=err_msg,
            )

        is_v = bool(out_ext["is_valid"])
        if is_v:
            status = "VALID"
        elif flt_res["is_expired"]:
            status = "EXPIRED"
        elif flt_res["is_stale"]:
            status = "STALE"
        else:
            status = "INVALID"

        rd = out_ext["ref_diagnostics"]
        st = out_ext["relative_cavity_stats"]
        last_valid_ts = (
            round(float(timestamp_sec) - float(flt_res["time_since_last_valid_sec"]), 4)
            if flt_res["time_since_last_valid_sec"] is not None
            else None
        )

        return ShadowMeasurementResult(
            method_id=method_id,
            frame_id=frame_id,
            frame_idx=frame_idx,
            timestamp_sec=round(float(timestamp_sec), 4),
            track_id=track_id,
            box_xyxy=[float(v) for v in box_xyxy],
            detector_conf=round(float(conf), 4),
            camera_profile=self.camera_profile,
            depth_preprocessing=self.preprocess_mode,
            relative_depth_polarity=RELATIVE_POLARITY,
            raw_relative_p75=st["p75_residual"] if is_v else None,
            normalized_relative_p75=st["scale_normalized_p75"] if is_v else None,
            smoothed_relative_score=flt_res["filtered_value"],
            is_valid=is_v,
            status=status,
            failure_reasons=list(out_ext["failure_reasons"]),
            ref_sample_count=rd["valid_ref_samples"],
            available_ref_samples=rd.get("available_ref_samples"),
            ransac_inlier_ratio=rd["inlier_ratio"],
            ref_mad=rd["ref_fit_mad"],
            spatial_support_quadrants=rd["populated_quadrants"],
            track_age_frames=flt_res["track_age_frames"],
            track_age_sec=flt_res["track_age_sec"],
            last_valid_timestamp_sec=last_valid_ts,
            stale_age_frames=flt_res["frames_since_last_valid"],
            water_glare_warning=bool(out_ext["is_water_suspected"]),
            extraction_latency_ms=round(ext_ms, 4),
            filtering_latency_ms=round(flt_ms, 4),
            adaptive_budget_used=budget_used,
            error_message=None,
        )

    def process_frame(
        self,
        frame_id: str,
        frame_idx: int,
        timestamp_sec: float,
        img_bgr: np.ndarray,
        depth_map: Optional[np.ndarray],
        boxes_xyxy: List[Tuple[float, float, float, float]],
        confs: List[float],
        track_ids: List[int],
        return_opt_arrays: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate Methods A, B, C (and optional D) on a single frame using shared detections,
        track IDs, and depth map. Failures in any experimental path never affect Method A.
        """
        H, W = img_bgr.shape[:2]
        self.filter_b.step_missing_tracks(track_ids, frame_idx)
        self.filter_c.step_missing_tracks(track_ids, frame_idx)
        if self.filter_d is not None:
            self.filter_d.step_missing_tracks(track_ids, frame_idx)

        results_by_method: Dict[str, List[ShadowMeasurementResult]] = {
            "A_legacy_annular_ema": [],
            "B_opt_ransac_valid_ema": [],
            "C_opt_ransac_rollmed_ema": [],
        }
        if self.enable_adaptive_variant:
            results_by_method["D_opt_ransac_adaptive_1024"] = []

        opt_arrays_by_box: List[Optional[Dict[str, Any]]] = []
        frame_timings = {"water_ms": 0.0, "A_ms": 0.0, "B_ms": 0.0, "C_ms": 0.0, "D_ms": 0.0}

        for idx, (box, conf, tid) in enumerate(zip(boxes_xyxy, confs, track_ids)):
            x1, y1, x2, y2 = max(0, int(round(box[0]))), max(0, int(round(box[1]))), min(W, int(round(box[2]))), min(H, int(round(box[3])))
            tw0 = time.perf_counter()
            is_water = SpecularAquaplaneDetector.inspect_patch(img_bgr[y1:y2, x1:x2]) if (x2 > x1 and y2 > y1) else False
            frame_timings["water_ms"] += (time.perf_counter() - tw0) * 1000.0

            # --- Method A: Isolated Legacy Production Baseline ---
            self.legacy_runner.ensure_resolution(H, W)
            ta0 = time.perf_counter()
            res_a = self.legacy_runner.evaluate_box(
                frame_id=frame_id,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                track_id=tid,
                box_xyxy=box,
                conf=conf,
                is_water=is_water,
                depth_map=depth_map,
                preprocess_mode=self.preprocess_mode,
            )
            frame_timings["A_ms"] += (time.perf_counter() - ta0) * 1000.0
            results_by_method["A_legacy_annular_ema"].append(res_a)

            nbrs = [tuple(boxes_xyxy[j]) for j in range(len(boxes_xyxy)) if j != idx]
            depth_missing = depth_map is None

            # --- Shared Optimized RANSAC Extraction for B & C (isolated exception guard) ---
            out_opt = None
            ext_opt_ms = 0.0
            opt_err = None
            try:
                if "opt_extract" in self.fault_hooks:
                    self.fault_hooks["opt_extract"]()
                if depth_missing:
                    raise ValueError("Depth map is None")
                out_opt = extract_optimized_relative_depth_for_box(
                    depth_map,
                    box,
                    neighbor_boxes=nbrs,
                    img_bgr=img_bgr,
                    cfg=self.opt_cfg,
                    return_arrays=return_opt_arrays,
                )
                ext_opt_ms = out_opt["timing_ms"]["extraction_total_ms"]
            except Exception as exc:
                opt_err = f"{type(exc).__name__}: {exc}"

            opt_arrays_by_box.append(out_opt.get("arrays") if (out_opt is not None and return_opt_arrays) else None)

            # --- Method B: Optimized RANSAC + Validity-Aware EMA ---
            res_b_err = opt_err
            flt_b_res = None
            flt_b_ms = 0.0
            if res_b_err is None and out_opt is not None:
                try:
                    if "B_filter" in self.fault_hooks:
                        self.fault_hooks["B_filter"]()
                    tb0 = time.perf_counter()
                    flt_b_res = self.filter_b.update(
                        tid,
                        out_opt["relative_cavity_stats"]["p75_residual"],
                        out_opt["is_valid"],
                        frame_idx,
                        timestamp_sec,
                    )
                    flt_b_ms = (time.perf_counter() - tb0) * 1000.0
                except Exception as exc:
                    res_b_err = f"{type(exc).__name__}: {exc}"
            frame_timings["B_ms"] += ext_opt_ms + flt_b_ms
            results_by_method["B_opt_ransac_valid_ema"].append(
                self._build_exp_result(
                    "B_opt_ransac_valid_ema", frame_id, frame_idx, timestamp_sec, tid, box, conf, is_water,
                    depth_missing, out_opt, flt_b_res, ext_opt_ms, flt_b_ms, self.opt_cfg.max_ref_samples, res_b_err
                )
            )

            # --- Method C: Optimized RANSAC + Rolling Median Validity-Aware EMA ---
            res_c_err = opt_err
            flt_c_res = None
            flt_c_ms = 0.0
            if res_c_err is None and out_opt is not None:
                try:
                    if "C_filter" in self.fault_hooks:
                        self.fault_hooks["C_filter"]()
                    tc0 = time.perf_counter()
                    flt_c_res = self.filter_c.update(
                        tid,
                        out_opt["relative_cavity_stats"]["p75_residual"],
                        out_opt["is_valid"],
                        frame_idx,
                        timestamp_sec,
                    )
                    flt_c_ms = (time.perf_counter() - tc0) * 1000.0
                except Exception as exc:
                    res_c_err = f"{type(exc).__name__}: {exc}"
            frame_timings["C_ms"] += ext_opt_ms + flt_c_ms
            results_by_method["C_opt_ransac_rollmed_ema"].append(
                self._build_exp_result(
                    "C_opt_ransac_rollmed_ema", frame_id, frame_idx, timestamp_sec, tid, box, conf, is_water,
                    depth_missing, out_opt, flt_c_res, ext_opt_ms, flt_c_ms, self.opt_cfg.max_ref_samples, res_c_err
                )
            )

            # --- Optional Method D: Adaptive 512 -> 1024 Trigger on Challenging Reference Fits ---
            if self.enable_adaptive_variant and self.filter_d is not None:
                res_d_err = opt_err
                out_d = out_opt
                ext_d_ms = ext_opt_ms
                used_b = self.opt_cfg.max_ref_samples
                flt_d_res = None
                flt_d_ms = 0.0
                if res_d_err is None and out_opt is not None:
                    try:
                        if should_escalate_budget(out_opt):
                            cfg_1024 = OptimizedDepthConfig(
                                sampling_mode=self.opt_cfg.sampling_mode,
                                max_ref_samples=1024,
                                seed=self.opt_cfg.seed,
                            )
                            out_d = extract_optimized_relative_depth_for_box(
                                depth_map, box, neighbor_boxes=nbrs, img_bgr=img_bgr, cfg=cfg_1024, return_arrays=False
                            )
                            ext_d_ms = ext_opt_ms + out_d["timing_ms"]["extraction_total_ms"]
                            used_b = 1024
                        td0 = time.perf_counter()
                        flt_d_res = self.filter_d.update(
                            tid,
                            out_d["relative_cavity_stats"]["p75_residual"],
                            out_d["is_valid"],
                            frame_idx,
                            timestamp_sec,
                        )
                        flt_d_ms = (time.perf_counter() - td0) * 1000.0
                    except Exception as exc:
                        res_d_err = f"{type(exc).__name__}: {exc}"
                frame_timings["D_ms"] += ext_d_ms + flt_d_ms
                results_by_method["D_opt_ransac_adaptive_1024"].append(
                    self._build_exp_result(
                        "D_opt_ransac_adaptive_1024", frame_id, frame_idx, timestamp_sec, tid, box, conf, is_water,
                        depth_missing, out_d, flt_d_res, ext_d_ms, flt_d_ms, used_b, res_d_err
                    )
                )

        return {
            "results_by_method": results_by_method,
            "opt_arrays_by_box": opt_arrays_by_box,
            "timings_ms": {k: round(v, 4) for k, v in frame_timings.items()},
        }
