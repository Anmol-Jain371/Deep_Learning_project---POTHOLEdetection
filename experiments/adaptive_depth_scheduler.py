"""
========================================================================================
RoadVision AI -- Phase 7A: Adaptive Track-Aware Depth Scheduler & Async Feasibility Worker
========================================================================================
Implements five causal depth-invocation policies over continuous YOLO + ByteTrack streams:
  A. `A_always_on`        : Run depth on every frame regardless of candidate count.
  B. `B_candidate_gated`  : Skip depth on `NO_CANDIDATES`; run whenever >= 1 proposal exists.
  C. `C_fixed_interval`   : Run depth every K active frames (K in {2, 3, 4, 5}).
  D. `D_track_aware`      : Trigger depth ONLY when >= 1 active track exhibits causal need
                            (NEW_TRACK, MEASUREMENT_EXPIRED, BOX_AREA_CHANGE,
                            BOX_POSITION_CHANGE, LOW_BOX_OVERLAP, INVALID_RETRY,
                            PERIODIC_REFRESH, SCENE_CHANGE, CAMERA_CHANGE), sharing one
                            full-frame depth forward pass across all tracks on keyframes.
  E. `E_budget_aware`     : `D_track_aware` plus causal latency/interval budget guard emitting
                            `SKIPPED_BY_BUDGET` instead of exceeding deadline budget, with
                            starvation protection for persistently stale/invalid tracks.

Also includes `BoundedAsyncDepthWorker` (Task 11 optional feasibility probe) using a
1-slot bounded queue with strict job timestamp/generation verification so expired or
out-of-order async depth maps NEVER overwrite newer track state.

STRICT CAUSAL CONSTRAINTS:
  - Decisions use ONLY information available up to `current_ts` (zero future lookahead).
  - Cached reads (`CACHED_VALID`) NEVER masquerade as `FRESH_VALID` observations.
  - Zero production file modifications; zero physical metric depth claims.
========================================================================================
"""

import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.track_depth_cache import (
    TrackDepthCache,
    TrackCacheConfig,
    compute_box_iou,
)

SCHEDULER_STRATEGIES = (
    "A_always_on",
    "B_candidate_gated",
    "C_fixed_interval",
    "D_track_aware",
    "E_budget_aware",
)

TRIGGER_REASONS = (
    "ALWAYS_ON",
    "CANDIDATE_PRESENT",
    "FIXED_INTERVAL",
    "FIXED_INTERVAL_SKIP",
    "NEW_TRACK",
    "PERIODIC_REFRESH",
    "BOX_AREA_CHANGE",
    "BOX_POSITION_CHANGE",
    "LOW_BOX_OVERLAP",
    "INVALID_RETRY",
    "MEASUREMENT_EXPIRED",
    "SCENE_CHANGE",
    "CAMERA_CHANGE",
    "NO_CANDIDATES",
    "SKIPPED_BY_BUDGET",
    "CACHED_HOLD",
)


@dataclass
class AdaptiveSchedulerConfig:
    strategy: str = "D_track_aware"
    fixed_interval_k: int = 3
    refresh_interval_sec: float = 0.16         # Refresh valid track ~6 Hz even if geometry steady
    max_measurement_age_sec: float = 0.45      # Hard measurement expiry horizon
    area_change_threshold: float = 0.30        # >= 30% bbox area shift vs last evaluated box
    centroid_shift_threshold: float = 0.35     # >= 35% of last bbox diag shift
    min_box_iou: float = 0.48                  # < 0.48 IoU vs last evaluated box triggers refresh
    invalid_retry_base_sec: float = 0.12       # Min cooldown before retrying invalid track
    invalid_retry_max_sec: float = 0.36        # Starvation-bounded backoff cap
    min_ref_inlier_hold: float = 0.55          # Low ref quality requests sooner refresh (0.55)
    scene_change_l1_thresh: float = 38.0       # Mean grayscale thumb L1 jump across consecutive frames
    budget_target_ema_ms: float = 38.0         # Used in E_budget_aware (soft ceiling for 25 FPS)
    budget_min_gap_sec: float = 0.08           # Minimum spacing between depth calls under budget pressure
    starvation_override_sec: float = 0.40      # Never delay > 0.40s even under budget load


class AdaptiveDepthScheduler:
    """
    Causal track-aware selective depth inference scheduler with deterministic decision logs.
    """

    def __init__(
        self,
        cfg: Optional[AdaptiveSchedulerConfig] = None,
        cache: Optional[TrackDepthCache] = None,
    ):
        self.cfg = cfg or AdaptiveSchedulerConfig()
        if self.cfg.strategy not in SCHEDULER_STRATEGIES:
            raise ValueError(f"Unknown strategy '{self.cfg.strategy}'; expected {SCHEDULER_STRATEGIES}")
        self.cache = cache or TrackDepthCache(
            video_session_id="default",
            depth_model_id="dav2_small",
            depth_input_size=336,
            inference_precision="fp16_weights",
            cfg=TrackCacheConfig(max_cache_age_sec=self.cfg.max_measurement_age_sec),
        )

        self._prev_thumb: Optional[np.ndarray] = None
        self._active_candidate_step: int = 0
        self._last_depth_ts: Optional[float] = None
        self._ema_frame_ms: float = 15.0
        self.total_frames: int = 0
        self.depth_calls: int = 0
        self.budget_skips: int = 0

    def reset_for_new_video(self, video_session_id: str, reason: str = "new_session") -> None:
        self.cache.video_session_id = str(video_session_id)
        self.cache.reset_session(reason)
        self._prev_thumb = None
        self._active_candidate_step = 0
        self._last_depth_ts = None
        self._ema_frame_ms = 15.0
        self.total_frames = 0
        self.depth_calls = 0
        self.budget_skips = 0

    def update_observed_latency(self, frame_total_ms: float) -> None:
        self._ema_frame_ms = 0.25 * float(frame_total_ms) + 0.75 * self._ema_frame_ms

    def _check_scene_change(self, frame_bgr: Optional[np.ndarray]) -> bool:
        if frame_bgr is None:
            return False
        small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
        if self._prev_thumb is None:
            self._prev_thumb = small
            return False
        diff = float(np.mean(np.abs(small.astype(np.float32) - self._prev_thumb.astype(np.float32))))
        self._prev_thumb = small
        return diff >= self.cfg.scene_change_l1_thresh

    def decide_frame(
        self,
        frame_idx: int,
        ts_sec: float,
        boxes_xyxy: List[Tuple[float, float, float, float]],
        track_ids: List[int],
        frame_bgr: Optional[np.ndarray] = None,
        camera_profile: str = "sedan_dashcam",
    ) -> Dict[str, Any]:
        """
        Evaluate purely causal triggers at `ts_sec` and return:
          - `run_depth`: bool
          - `frame_reasons`: list of trigger reasons causing inference or skip
          - `per_track_reasons`: dict track_id -> reason string
          - `shared_across_tracks`: int (how many tracks benefit from this single depth call)
        """
        t0 = time.perf_counter()
        self.total_frames += 1

        cam_changed = self.cache.set_camera_profile(camera_profile)
        scene_changed = self._check_scene_change(frame_bgr)
        if scene_changed and not cam_changed:
            self.cache.reset_session("SCENE_CHANGE")

        # Purge tracks missing past absence reset horizon
        self.cache.purge_missing_tracks(track_ids, ts_sec)

        # Zero candidates handling
        if not boxes_xyxy:
            if self.cfg.strategy == "A_always_on":
                self.depth_calls += 1
                self._last_depth_ts = float(ts_sec)
                return {
                    "run_depth": True,
                    "frame_reasons": ["ALWAYS_ON"],
                    "per_track_reasons": {},
                    "shared_across_tracks": 0,
                    "scheduler_ms": round((time.perf_counter() - t0) * 1000.0, 4),
                }
            return {
                "run_depth": False,
                "frame_reasons": ["NO_CANDIDATES"],
                "per_track_reasons": {},
                "shared_across_tracks": 0,
                "scheduler_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        # Register observations & track-ID reuse resets FIRST cleanly
        obs_info = {}
        for tid, bx in zip(track_ids, boxes_xyxy):
            ent, is_new, reuse_reason = self.cache.observe_detection(tid, frame_idx, ts_sec, bx)
            obs_info[tid] = (ent, is_new, reuse_reason)

        st = self.cfg.strategy
        if st in ("A_always_on", "B_candidate_gated"):
            tag = "ALWAYS_ON" if st == "A_always_on" else "CANDIDATE_PRESENT"
            self.depth_calls += 1
            self._last_depth_ts = float(ts_sec)
            return {
                "run_depth": True,
                "frame_reasons": [tag],
                "per_track_reasons": {tid: tag for tid in track_ids},
                "shared_across_tracks": len(track_ids),
                "scheduler_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        if st == "C_fixed_interval":
            k = max(1, int(self.cfg.fixed_interval_k))
            run = (self._active_candidate_step % k == 0) or cam_changed or scene_changed
            self._active_candidate_step += 1
            reason = "CAMERA_CHANGE" if cam_changed else ("SCENE_CHANGE" if scene_changed else ("FIXED_INTERVAL" if run else "FIXED_INTERVAL_SKIP"))
            if run:
                self.depth_calls += 1
                self._last_depth_ts = float(ts_sec)
            return {
                "run_depth": run,
                "frame_reasons": [reason],
                "per_track_reasons": {tid: reason for tid in track_ids},
                "shared_across_tracks": len(track_ids) if run else 0,
                "scheduler_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        # Strategies D_track_aware and E_budget_aware: inspect each track causally
        per_track_triggers: Dict[int, str] = {}
        critical_trigger = cam_changed or scene_changed
        starvation_due = False

        for tid, bx in zip(track_ids, boxes_xyxy):
            ent, is_new, reuse_reason = obs_info[tid]
            if cam_changed:
                per_track_triggers[tid] = "CAMERA_CHANGE"
                continue
            if scene_changed:
                per_track_triggers[tid] = "SCENE_CHANGE"
                continue
            if is_new or ent.last_attempted_timestamp_sec is None:
                per_track_triggers[tid] = "NEW_TRACK"
                starvation_due = True
                continue

            age_valid = ent.cache_age_sec(ts_sec)
            since_attempt = max(0.0, float(ts_sec) - float(ent.last_attempted_timestamp_sec))

            # Track has NEVER achieved a valid observation yet -> invalid retry with starvation cap
            if ent.last_valid_timestamp_sec is None:
                backoff = min(self.cfg.invalid_retry_max_sec,
                              self.cfg.invalid_retry_base_sec * (1.5 ** min(3, ent.consecutive_invalid_attempts)))
                if since_attempt >= backoff:
                    per_track_triggers[tid] = "INVALID_RETRY"
                    if since_attempt >= self.cfg.starvation_override_sec:
                        starvation_due = True
                else:
                    per_track_triggers[tid] = "CACHED_HOLD"
                continue

            # Expired timestamp check
            if ent.is_expired_at(ts_sec, self.cache.cfg) or (age_valid is not None and age_valid >= self.cfg.max_measurement_age_sec):
                per_track_triggers[tid] = "MEASUREMENT_EXPIRED"
                starvation_due = True
                continue

            # Previous attempt invalid -> retry after backoff even if cached hold exists
            if ent.last_attempt_valid is False:
                backoff = min(self.cfg.invalid_retry_max_sec,
                              self.cfg.invalid_retry_base_sec * (1.5 ** min(3, ent.consecutive_invalid_attempts)))
                if since_attempt >= backoff:
                    per_track_triggers[tid] = "INVALID_RETRY"
                    continue

            # Bounding-box geometry displacement triggers vs last EVALUATED box
            eb = ent.last_evaluated_box_xyxy
            if eb is not None:
                area_e = max(1.0, (eb[2] - eb[0]) * (eb[3] - eb[1]))
                area_c = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
                if abs(area_c - area_e) / area_e >= self.cfg.area_change_threshold:
                    per_track_triggers[tid] = "BOX_AREA_CHANGE"
                    continue
                iou = compute_box_iou(eb, bx)
                if iou < self.cfg.min_box_iou:
                    per_track_triggers[tid] = "LOW_BOX_OVERLAP"
                    continue
                cx_e, cy_e = 0.5 * (eb[0] + eb[2]), 0.5 * (eb[1] + eb[3])
                cx_c, cy_c = 0.5 * (bx[0] + bx[2]), 0.5 * (bx[1] + bx[3])
                diag_e = max(20.0, math.hypot(eb[2] - eb[0], eb[3] - eb[1]))
                if math.hypot(cx_c - cx_e, cy_c - cy_e) / diag_e >= self.cfg.centroid_shift_threshold:
                    per_track_triggers[tid] = "BOX_POSITION_CHANGE"
                    continue

            # Low previous reference-fit quality or periodic freshness refresh
            prev_inl = float(ent.ref_diagnostics.get("inlier_ratio", 1.0)) if ent.ref_diagnostics else 1.0
            eff_refresh = self.cfg.refresh_interval_sec * (0.75 if prev_inl < self.cfg.min_ref_inlier_hold else 1.0)
            if age_valid is not None and age_valid >= eff_refresh:
                per_track_triggers[tid] = "PERIODIC_REFRESH"
                if age_valid >= self.cfg.starvation_override_sec:
                    starvation_due = True
                continue

            per_track_triggers[tid] = "CACHED_HOLD"

        active_triggers = [r for r in per_track_triggers.values() if r != "CACHED_HOLD"]
        want_depth = len(active_triggers) > 0

        if not want_depth:
            return {
                "run_depth": False,
                "frame_reasons": ["CACHED_HOLD"],
                "per_track_reasons": per_track_triggers,
                "shared_across_tracks": 0,
                "scheduler_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        # Apply latency budget guard ONLY under E_budget_aware when non-critical/non-starving
        if st == "E_budget_aware" and (not critical_trigger) and (not starvation_due):
            dt_last_depth = (float(ts_sec) - self._last_depth_ts) if self._last_depth_ts is not None else 1e9
            if (self._ema_frame_ms > self.cfg.budget_target_ema_ms) or (dt_last_depth < self.cfg.budget_min_gap_sec):
                self.budget_skips += 1
                return {
                    "run_depth": False,
                    "frame_reasons": ["SKIPPED_BY_BUDGET"],
                    "per_track_reasons": {tid: "SKIPPED_BY_BUDGET" for tid in track_ids},
                    "shared_across_tracks": 0,
                    "suppressed_triggers": sorted(set(active_triggers)),
                    "scheduler_ms": round((time.perf_counter() - t0) * 1000.0, 4),
                }

        self.depth_calls += 1
        self._last_depth_ts = float(ts_sec)
        return {
            "run_depth": True,
            "frame_reasons": sorted(set(active_triggers)),
            "per_track_reasons": per_track_triggers,
            "shared_across_tracks": len(track_ids),
            "scheduler_ms": round((time.perf_counter() - t0) * 1000.0, 4),
        }


class BoundedAsyncDepthWorker:
    """
    Task 11 Optional Bounded Asynchronous Depth Worker (`maxsize=1`).
    Enforces strict job-session, timestamp-age, monotonic-order, and generation validation
    so stale/out-of-order depth maps are rejected cleanly and never corrupt track state.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], Any],
        max_result_age_sec: float = 0.25,
    ):
        self.predict_fn = predict_fn
        self.max_result_age_sec = float(max_result_age_sec)
        self._req_q: queue.Queue = queue.Queue(maxsize=1)
        self._res_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop_ev = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.current_session_epoch: int = 1
        self.last_consumed_frame_idx: int = -1
        self.submitted_jobs: int = 0
        self.dropped_queue_full: int = 0
        self.completed_jobs: int = 0
        self.rejected_stale_or_order: int = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_ev.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def invalidate_session(self) -> None:
        """Bump session epoch cleanly so any in-flight async jobs are rejected upon completion."""
        self.current_session_epoch += 1
        self.last_consumed_frame_idx = -1
        while not self._res_q.empty():
            try:
                self._res_q.get_nowait()
            except queue.Empty:
                break

    def submit(
        self,
        frame_idx: int,
        ts_sec: float,
        img_bgr: np.ndarray,
        boxes_by_track: Dict[int, Tuple[float, float, float, float]],
    ) -> bool:
        """Non-blocking submit onto bounded maxsize=1 queue; drops new request if busy."""
        payload = {
            "session_epoch": self.current_session_epoch,
            "frame_idx": int(frame_idx),
            "ts_sec": float(ts_sec),
            "img_bgr": img_bgr.copy(),
            "boxes_by_track": dict(boxes_by_track),
            "submit_wall_t": time.perf_counter(),
        }
        try:
            self._req_q.put_nowait(payload)
            self.submitted_jobs += 1
            return True
        except queue.Full:
            self.dropped_queue_full += 1
            return False

    def _worker_loop(self) -> None:
        while not self._stop_ev.is_set():
            try:
                job = self._req_q.get(timeout=0.02)
            except queue.Empty:
                continue
            try:
                pred = self.predict_fn(job["img_bgr"])
                out = {
                    "session_epoch": job["session_epoch"],
                    "frame_idx": job["frame_idx"],
                    "ts_sec": job["ts_sec"],
                    "boxes_by_track": job["boxes_by_track"],
                    "img_bgr": job["img_bgr"],
                    "depth_map": pred.depth_map if hasattr(pred, "depth_map") else pred["depth_map"],
                    "worker_total_ms": pred.timings_ms["total_ms"] if hasattr(pred, "timings_ms") else 0.0,
                    "complete_wall_t": time.perf_counter(),
                }
                while self._res_q.full():
                    try:
                        _ = self._res_q.get_nowait()
                        self.rejected_stale_or_order += 1
                    except queue.Empty:
                        break
                self._res_q.put_nowait(out)
                self.completed_jobs += 1
            except Exception:
                self.rejected_stale_or_order += 1
            finally:
                self._req_q.task_done()

    def poll_latest_valid(
        self,
        current_frame_idx: int,
        current_ts_sec: float,
        min_frame_idx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Drain ready worker outputs and return ONLY the newest result passing epoch,
        monotonic order (`frame_idx > last_consumed_frame_idx`), and timestamp age checks.
        """
        best = None
        while not self._res_q.empty():
            try:
                res = self._res_q.get_nowait()
            except queue.Empty:
                break
            age = float(current_ts_sec) - float(res["ts_sec"])
            bad_epoch = (res["session_epoch"] != self.current_session_epoch)
            bad_order = (res["frame_idx"] <= self.last_consumed_frame_idx) or (
                min_frame_idx is not None and res["frame_idx"] < min_frame_idx
            )
            expired = (age < 0.0) or (age > self.max_result_age_sec)
            if bad_epoch or bad_order or expired:
                self.rejected_stale_or_order += 1
                continue
            if best is not None:
                self.rejected_stale_or_order += 1
            best = res

        if best is not None:
            self.last_consumed_frame_idx = best["frame_idx"]
        return best
