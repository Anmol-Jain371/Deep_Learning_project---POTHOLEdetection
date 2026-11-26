"""
========================================================================================
RoadVision AI -- Phase 7A: Track-Level Depth Cache & Time-Aware Temporal Filter Adapter
========================================================================================
Maintains per-(video_session, track_id, track_generation_id) relative-depth states across
selective keyframe / track-aware scheduling steps WITHOUT ever:
  1. Counting cached predictions as fresh observations (`FRESH_VALID` vs `CACHED_VALID`).
  2. Feeding cached or invalid values into the temporal filter (`EMA` updates ONLY on `FRESH_VALID`).
  3. Reusing old pixel-level depth maps across moving dashcam frames without spatial transform.
  4. Leaking measurements across track-ID reuse, track disappearance gaps, or camera/scene switches.
  5. Fabricating `0.0` when expired, unmeasured, or invalid (`None` explicitly).

Supports both standard frame-step `ValidityAwareTrackFilter` (`alpha=0.35`) and
continuous-time irregular-interval EMA (`alpha_eff = 1 - exp(-dt / tau_sec)`) for Task 9.
========================================================================================
"""

import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple, Deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.validity_aware_temporal_filter import (
    TemporalFilterConfig,
    ValidityAwareTrackFilter,
)

CACHE_STATES = (
    "FRESH_VALID",
    "FRESH_INVALID",
    "CACHED_VALID",
    "EXPIRED",
    "NOT_EVALUATED",
)


def compute_box_iou(
    box_a: Optional[Tuple[float, float, float, float]],
    box_b: Optional[Tuple[float, float, float, float]],
) -> float:
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1e-6, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return float(inter / max(1e-6, area_a + area_b - inter))


@dataclass
class TrackCacheConfig:
    max_cache_age_sec: float = 0.45         # Expire cached relative score if older than 450 ms
    max_invalid_stale_sec: float = 0.25     # Do not hold prior valid value across prolonged invalid retries
    max_absence_reset_sec: float = 0.60     # Gap > 0.60s treats returning track_id as new generation (anti-reuse)
    max_reuse_centroid_norm: float = 1.75   # Spatial jump > 1.75x bbox diag with zero IoU => ID reuse reset
    tau_sec: float = 0.16                   # Time-constant for irregular-interval continuous-time EMA
    rolling_window: int = 5
    fixed_alpha: float = 0.35


@dataclass
class TrackCacheEntry:
    video_session_id: str
    track_id: int
    track_generation_id: int
    depth_model_id: str
    depth_input_size: int
    inference_precision: str

    last_valid_relative_score: Optional[float] = None
    last_valid_raw_p75: Optional[float] = None
    last_valid_iqr_norm_p75: Optional[float] = None
    last_valid_frame_id: Optional[str] = None
    last_valid_frame_idx: Optional[int] = None
    last_valid_timestamp_sec: Optional[float] = None

    last_attempted_frame_idx: Optional[int] = None
    last_attempted_timestamp_sec: Optional[float] = None
    last_attempt_valid: Optional[bool] = None
    last_failure_reasons: Tuple[str, ...] = ()
    ref_diagnostics: Dict[str, Any] = field(default_factory=dict)

    last_evaluated_box_xyxy: Optional[Tuple[float, float, float, float]] = None
    last_seen_box_xyxy: Optional[Tuple[float, float, float, float]] = None
    last_seen_frame_idx: int = -1
    last_seen_timestamp_sec: float = 0.0

    # Temporal smoothing states strictly separated per track generation & NEVER updated by cache reads
    ema_fixed_score: Optional[float] = None
    ema_timeaware_score: Optional[float] = None
    rollmed_ema_score: Optional[float] = None
    valid_window: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    last_ema_update_ts: Optional[float] = None

    fresh_valid_count: int = 0
    fresh_invalid_count: int = 0
    cached_read_count: int = 0
    consecutive_invalid_attempts: int = 0
    first_seen_frame_idx: int = -1
    first_seen_timestamp_sec: float = 0.0

    def cache_age_sec(self, current_ts: float) -> Optional[float]:
        if self.last_valid_timestamp_sec is None:
            return None
        return max(0.0, float(current_ts) - float(self.last_valid_timestamp_sec))

    def cache_age_frames(self, current_frame_idx: int) -> Optional[int]:
        if self.last_valid_frame_idx is None:
            return None
        return max(0, int(current_frame_idx) - int(self.last_valid_frame_idx))

    def is_expired_at(self, current_ts: float, cfg: TrackCacheConfig) -> bool:
        age = self.cache_age_sec(current_ts)
        if age is None:
            return False
        if age > cfg.max_cache_age_sec:
            return True
        if (
            self.last_attempt_valid is False
            and self.last_valid_timestamp_sec is not None
            and age > cfg.max_invalid_stale_sec
        ):
            return True
        return False


class TrackDepthCache:
    """
    Explicit lifecycle-aware track measurement cache managing FRESH_VALID, FRESH_INVALID,
    CACHED_VALID, EXPIRED, and NOT_EVALUATED states across heterogeneous frame rates.
    """

    def __init__(
        self,
        video_session_id: str,
        depth_model_id: str,
        depth_input_size: int,
        inference_precision: str = "fp16_weights",
        cfg: Optional[TrackCacheConfig] = None,
    ):
        self.video_session_id = str(video_session_id)
        self.depth_model_id = str(depth_model_id)
        self.depth_input_size = int(depth_input_size)
        self.inference_precision = str(inference_precision)
        self.cfg = cfg or TrackCacheConfig()

        self._entries: Dict[int, TrackCacheEntry] = {}
        self._generation_counter: Dict[int, int] = {}
        self._camera_profile: str = "sedan_dashcam"
        self.reset_events: int = 0

        # Underlying ValidityAwareTrackFilter kept synchronized ONLY on fresh observations
        tf_cfg = TemporalFilterConfig(
            alpha=self.cfg.fixed_alpha,
            rolling_window=self.cfg.rolling_window,
            max_stale_frames=1000,   # Expiration strictly governed by timestamp age cleanly
            max_missed_frames_reset=1000,
        )
        self._legacy_safe_ema = ValidityAwareTrackFilter("validity_aware_ema", cfg=tf_cfg)
        self._legacy_safe_roll = ValidityAwareTrackFilter("rolling_median_validity_ema", cfg=tf_cfg)

    def _next_generation(self, track_id: int) -> int:
        g = self._generation_counter.get(track_id, 0) + 1
        self._generation_counter[track_id] = g
        return g

    def _new_entry(
        self,
        track_id: int,
        frame_idx: int,
        ts_sec: float,
        box_xyxy: Tuple[float, float, float, float],
    ) -> TrackCacheEntry:
        gen = self._next_generation(track_id)
        self._legacy_safe_ema.reset_track(track_id)
        self._legacy_safe_roll.reset_track(track_id)
        return TrackCacheEntry(
            video_session_id=self.video_session_id,
            track_id=track_id,
            track_generation_id=gen,
            depth_model_id=self.depth_model_id,
            depth_input_size=self.depth_input_size,
            inference_precision=self.inference_precision,
            last_seen_box_xyxy=tuple(float(v) for v in box_xyxy),
            last_seen_frame_idx=int(frame_idx),
            last_seen_timestamp_sec=float(ts_sec),
            first_seen_frame_idx=int(frame_idx),
            first_seen_timestamp_sec=float(ts_sec),
            valid_window=deque(maxlen=max(1, self.cfg.rolling_window)),
        )

    def reset_session(self, reason: str = "scene_or_camera_change") -> None:
        """Invalidate all cached entries on scene cut or camera change cleanly."""
        self._entries.clear()
        self._legacy_safe_ema.reset_all()
        self._legacy_safe_roll.reset_all()
        self.reset_events += 1

    def set_camera_profile(self, camera_profile: str) -> bool:
        if camera_profile != self._camera_profile:
            self._camera_profile = camera_profile
            self.reset_session("camera_change")
            return True
        return False

    def observe_detection(
        self,
        track_id: int,
        frame_idx: int,
        ts_sec: float,
        box_xyxy: Tuple[float, float, float, float],
    ) -> Tuple[TrackCacheEntry, bool, Optional[str]]:
        """
        Register track presence BEFORE scheduler decision. Detects new tracks and track-ID reuse.
        Returns (entry, is_new_or_reset_track, reset_reason).
        """
        ent = self._entries.get(track_id)
        bx = tuple(float(v) for v in box_xyxy)
        if ent is None:
            ent = self._new_entry(track_id, frame_idx, ts_sec, bx)
            self._entries[track_id] = ent
            return ent, True, "NEW_TRACK"

        # Detect likely ByteTrack ID reuse or long disappearance gap cleanly
        gap_sec = float(ts_sec) - ent.last_seen_timestamp_sec
        if gap_sec > self.cfg.max_absence_reset_sec:
            ent = self._new_entry(track_id, frame_idx, ts_sec, bx)
            self._entries[track_id] = ent
            return ent, True, "TRACK_ID_REUSED_AFTER_GAP"

        if gap_sec > 0.20 and ent.last_seen_box_xyxy is not None:
            prev = ent.last_seen_box_xyxy
            iou = compute_box_iou(prev, bx)
            cx0, cy0 = 0.5 * (prev[0] + prev[2]), 0.5 * (prev[1] + prev[3])
            cx1, cy1 = 0.5 * (bx[0] + bx[2]), 0.5 * (bx[1] + bx[3])
            diag = max(20.0, math.hypot(prev[2] - prev[0], prev[3] - prev[1]))
            jump_norm = math.hypot(cx1 - cx0, cy1 - cy0) / diag
            if iou == 0.0 and jump_norm > self.cfg.max_reuse_centroid_norm:
                ent = self._new_entry(track_id, frame_idx, ts_sec, bx)
                self._entries[track_id] = ent
                return ent, True, "TRACK_ID_REUSED_SPATIAL_JUMP"

        ent.last_seen_box_xyxy = bx
        ent.last_seen_frame_idx = int(frame_idx)
        ent.last_seen_timestamp_sec = float(ts_sec)
        return ent, False, None

    def get_entry(self, track_id: int) -> Optional[TrackCacheEntry]:
        return self._entries.get(track_id)

    def purge_missing_tracks(self, active_track_ids: list, current_ts: float) -> int:
        active = set(active_track_ids)
        removed = 0
        for tid in list(self._entries.keys()):
            if tid not in active:
                gap = float(current_ts) - self._entries[tid].last_seen_timestamp_sec
                if gap > self.cfg.max_absence_reset_sec:
                    del self._entries[tid]
                    self._legacy_safe_ema.reset_track(tid)
                    self._legacy_safe_roll.reset_track(tid)
                    removed += 1
        return removed

    def record_fresh_measurement(
        self,
        track_id: int,
        frame_id: str,
        frame_idx: int,
        ts_sec: float,
        box_xyxy: Tuple[float, float, float, float],
        is_valid: bool,
        relative_score: Optional[float],
        raw_p75: Optional[float],
        iqr_norm_p75: Optional[float],
        ref_diagnostics: Dict[str, Any],
        failure_reasons: Tuple[str, ...] = (),
    ) -> Dict[str, Any]:
        """
        Update cache & temporal smoothing ONLY from an actual fresh depth invocation.
        """
        ent = self._entries.get(track_id)
        if ent is None:
            ent, _, _ = self.observe_detection(track_id, frame_idx, ts_sec, box_xyxy)

        bx = tuple(float(v) for v in box_xyxy)
        ent.last_attempted_frame_idx = int(frame_idx)
        ent.last_attempted_timestamp_sec = float(ts_sec)
        ent.last_evaluated_box_xyxy = bx
        ent.ref_diagnostics = dict(ref_diagnostics)

        valid_clean = bool(is_valid and relative_score is not None and np.isfinite(relative_score))
        if valid_clean:
            val = float(relative_score)
            ent.last_valid_relative_score = val
            ent.last_valid_raw_p75 = float(raw_p75) if raw_p75 is not None else None
            ent.last_valid_iqr_norm_p75 = float(iqr_norm_p75) if iqr_norm_p75 is not None else None
            ent.last_valid_frame_id = str(frame_id)
            ent.last_valid_frame_idx = int(frame_idx)
            ent.last_valid_timestamp_sec = float(ts_sec)
            ent.last_attempt_valid = True
            ent.last_failure_reasons = ()
            ent.fresh_valid_count += 1
            ent.consecutive_invalid_attempts = 0

            # Fixed-step ValidityAwareEMA & Rolling-Median EMA (strictly on FRESH_VALID)
            out_fixed = self._legacy_safe_ema.update(track_id, val, True, frame_idx, ts_sec)
            out_roll = self._legacy_safe_roll.update(track_id, val, True, frame_idx, ts_sec)
            ent.ema_fixed_score = out_fixed["filtered_value"]
            ent.rollmed_ema_score = out_roll["filtered_value"]

            # Continuous-time irregular-interval EMA: alpha_eff = 1 - exp(-dt / tau_sec)
            ent.valid_window.append(val)
            if ent.ema_timeaware_score is None or ent.last_ema_update_ts is None:
                ent.ema_timeaware_score = val
            else:
                dt = max(1e-3, float(ts_sec) - float(ent.last_ema_update_ts))
                a_eff = 1.0 - math.exp(-dt / max(1e-3, self.cfg.tau_sec))
                ent.ema_timeaware_score = round(a_eff * val + (1.0 - a_eff) * ent.ema_timeaware_score, 6)
            ent.last_ema_update_ts = float(ts_sec)

            return self._build_output(ent, ts_sec, frame_idx, state="FRESH_VALID", is_fresh=True, is_cached=False)

        else:
            ent.last_attempt_valid = False
            ent.last_failure_reasons = tuple(failure_reasons)
            ent.fresh_invalid_count += 1
            ent.consecutive_invalid_attempts += 1
            # Strictly never modify temporal numeric state on FRESH_INVALID!
            return self._build_output(ent, ts_sec, frame_idx, state="FRESH_INVALID", is_fresh=True, is_cached=False)

    def read_non_keyframe_state(
        self,
        track_id: int,
        frame_idx: int,
        ts_sec: float,
    ) -> Dict[str, Any]:
        """
        Return read-only cached state for a skipped frame without modifying EMA or counting
        the observation as fresh. Explicitly transitions to EXPIRED when max age exceeded.
        """
        ent = self._entries.get(track_id)
        if ent is None or ent.last_valid_relative_score is None:
            return {
                "state": "NOT_EVALUATED",
                "is_fresh": False,
                "is_cached": False,
                "is_expired": False,
                "is_valid_available": False,
                "track_generation_id": ent.track_generation_id if ent else None,
                "relative_score": None,
                "ema_fixed_score": None,
                "ema_timeaware_score": None,
                "rollmed_ema_score": None,
                "cache_age_sec": None,
                "cache_age_frames": None,
                "fresh_valid_count": ent.fresh_valid_count if ent else 0,
                "cached_read_count": ent.cached_read_count if ent else 0,
            }

        if ent.is_expired_at(ts_sec, self.cfg):
            return self._build_output(ent, ts_sec, frame_idx, state="EXPIRED", is_fresh=False, is_cached=False)

        ent.cached_read_count += 1
        return self._build_output(ent, ts_sec, frame_idx, state="CACHED_VALID", is_fresh=False, is_cached=True)

    def _build_output(
        self,
        ent: TrackCacheEntry,
        ts_sec: float,
        frame_idx: int,
        state: str,
        is_fresh: bool,
        is_cached: bool,
    ) -> Dict[str, Any]:
        age_sec = ent.cache_age_sec(ts_sec)
        age_frames = ent.cache_age_frames(frame_idx)
        expired = (state == "EXPIRED")
        # Never expose old score when FRESH_INVALID or EXPIRED or NOT_EVALUATED!
        # Only FRESH_VALID and CACHED_VALID expose valid measurement values.
        avail = state in ("FRESH_VALID", "CACHED_VALID")
        return {
            "state": state,
            "is_fresh": is_fresh,
            "is_cached": is_cached,
            "is_expired": expired,
            "is_valid_available": avail,
            "track_generation_id": ent.track_generation_id,
            "relative_score": ent.last_valid_relative_score if avail else None,
            "raw_p75": ent.last_valid_raw_p75 if avail else None,
            "iqr_norm_p75": ent.last_valid_iqr_norm_p75 if avail else None,
            "ema_fixed_score": ent.ema_fixed_score if avail else None,
            "ema_timeaware_score": ent.ema_timeaware_score if avail else None,
            "rollmed_ema_score": ent.rollmed_ema_score if avail else None,
            "cache_age_sec": round(age_sec, 4) if age_sec is not None else None,
            "cache_age_frames": age_frames,
            "fresh_valid_count": ent.fresh_valid_count,
            "cached_read_count": ent.cached_read_count,
            "last_failure_reasons": list(ent.last_failure_reasons),
            "ref_diagnostics": ent.ref_diagnostics if avail else {},
            "meta": {
                "video_session_id": ent.video_session_id,
                "depth_model_id": ent.depth_model_id,
                "depth_input_size": ent.depth_input_size,
                "inference_precision": ent.inference_precision,
            },
        }
