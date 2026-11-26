"""
========================================================================================
RoadVision AI -- Phase 5B: Validity-Aware Temporal Filtering for Relative Depth Tracks
========================================================================================
Independent experimental temporal filtering module comparing four per-track smoothing
policies strictly in relative inverse-depth units (never cm; never production `roadvision_engine.py`):
  1. raw                         : unsmoothed valid residual (`None` on invalid frames)
  2. legacy_ema_zero_clamp       : reproduces production bug clamping invalid/0 to 0.0 in EMA
  3. validity_aware_ema          : updates EMA ONLY on `is_valid=True`; tracks staleness & expires
  4. rolling_median_validity_ema : causal rolling median of recent valid observations + validity EMA

Strict requirements enforced:
  - Invalid measurements (`is_valid=False` or `value is None` / non-finite) NEVER update numerical state.
  - Missing/expired estimates are explicitly represented as `None` (never fabricated 0.0).
  - Track age (`track_age_frames`, `track_age_sec`) and time since last valid measurement are tracked.
  - Estimates held across invalid gaps mark `is_stale=True` and expire/reset cleanly when gap exceeds
    `max_stale_frames` or `max_missed_frames_reset`.
========================================================================================
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Deque
import numpy as np

FILTER_MODES = (
    "raw",
    "legacy_ema_zero_clamp",
    "validity_aware_ema",
    "rolling_median_validity_ema",
)


@dataclass
class TemporalFilterConfig:
    alpha: float = 0.35
    rolling_window: int = 5
    max_stale_frames: int = 3          # After 3 consecutive invalid/unmeasured steps, mark expired -> None
    max_missed_frames_reset: int = 8   # Hard track state reset if absent/invalid for >= 8 frames
    max_stale_sec: Optional[float] = None


@dataclass
class TrackState:
    track_id: int
    first_seen_frame: int
    first_seen_ts: float
    last_seen_frame: int
    last_seen_ts: float
    ema_value: Optional[float] = None
    last_valid_frame: Optional[int] = None
    last_valid_ts: Optional[float] = None
    consecutive_invalid: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    valid_window: Deque[float] = field(default_factory=lambda: deque(maxlen=5))


class ValidityAwareTrackFilter:
    """Per-track validity-aware temporal filter operating strictly in relative-depth units."""

    def __init__(self, mode: str = "validity_aware_ema", cfg: Optional[TemporalFilterConfig] = None):
        if mode not in FILTER_MODES:
            raise ValueError(f"Unknown filter mode '{mode}'; expected one of {FILTER_MODES}")
        self.mode = mode
        self.cfg = cfg or TemporalFilterConfig()
        self._tracks: Dict[int, TrackState] = {}

    def reset_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def reset_all(self) -> None:
        self._tracks.clear()

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)

    def step_missing_tracks(self, active_track_ids: List[int], frame_idx: int) -> List[int]:
        """Reset tracks that have not been seen by the tracker for >= max_missed_frames_reset."""
        expired = []
        active_set = set(active_track_ids)
        for tid, st in list(self._tracks.items()):
            if tid not in active_set and (frame_idx - st.last_seen_frame) >= self.cfg.max_missed_frames_reset:
                expired.append(tid)
                del self._tracks[tid]
        return expired

    def update(
        self,
        track_id: int,
        raw_value: Optional[float],
        is_valid: bool,
        frame_idx: int,
        timestamp_sec: float,
    ) -> Dict[str, Any]:
        """Process one detection observation for `track_id` and return explicit state & estimate."""
        val_clean: Optional[float] = None
        if is_valid and raw_value is not None and np.isfinite(raw_value):
            val_clean = float(raw_value)
            valid_obs = True
        else:
            valid_obs = False

        st = self._tracks.get(track_id)
        if st is None:
            st = TrackState(
                track_id=track_id,
                first_seen_frame=frame_idx,
                first_seen_ts=float(timestamp_sec),
                last_seen_frame=frame_idx,
                last_seen_ts=float(timestamp_sec),
                valid_window=deque(maxlen=max(1, self.cfg.rolling_window)),
            )
            self._tracks[track_id] = st
        elif (frame_idx - st.last_seen_frame) >= self.cfg.max_missed_frames_reset:
            # Auto-reset expired reappearing track ID so stale prior episode state does not leak
            st = TrackState(
                track_id=track_id,
                first_seen_frame=frame_idx,
                first_seen_ts=float(timestamp_sec),
                last_seen_frame=frame_idx,
                last_seen_ts=float(timestamp_sec),
                valid_window=deque(maxlen=max(1, self.cfg.rolling_window)),
            )
            self._tracks[track_id] = st

        st.last_seen_frame = frame_idx
        st.last_seen_ts = float(timestamp_sec)
        track_age_frames = frame_idx - st.first_seen_frame
        track_age_sec = float(timestamp_sec) - st.first_seen_ts

        did_update = False
        is_stale = False
        is_expired_stale = False
        filtered_out: Optional[float] = None

        if self.mode == "raw":
            if valid_obs:
                st.valid_count += 1
                st.last_valid_frame = frame_idx
                st.last_valid_ts = float(timestamp_sec)
                st.consecutive_invalid = 0
                st.ema_value = val_clean
                filtered_out = val_clean
                did_update = True
            else:
                st.invalid_count += 1
                st.consecutive_invalid += 1
                is_stale = st.last_valid_frame is not None
                filtered_out = None

        elif self.mode == "legacy_ema_zero_clamp":
            # Reproduces historical production bug: invalid / suppressed observations feed 0.0 into EMA
            input_val = val_clean if valid_obs else 0.0
            if st.ema_value is None:
                st.ema_value = input_val
            else:
                st.ema_value = self.cfg.alpha * input_val + (1.0 - self.cfg.alpha) * st.ema_value
            if valid_obs:
                st.valid_count += 1
                st.last_valid_frame = frame_idx
                st.last_valid_ts = float(timestamp_sec)
                st.consecutive_invalid = 0
                did_update = True
            else:
                st.invalid_count += 1
                st.consecutive_invalid += 1
                is_stale = True
            filtered_out = round(float(st.ema_value), 6)

        else:
            # Validity-aware modes (`validity_aware_ema` & `rolling_median_validity_ema`)
            if valid_obs:
                st.valid_count += 1
                st.last_valid_frame = frame_idx
                st.last_valid_ts = float(timestamp_sec)
                st.consecutive_invalid = 0
                st.valid_window.append(val_clean)

                target_val = (
                    float(np.median(list(st.valid_window)))
                    if self.mode == "rolling_median_validity_ema"
                    else val_clean
                )
                if st.ema_value is None:
                    st.ema_value = target_val
                else:
                    st.ema_value = self.cfg.alpha * target_val + (1.0 - self.cfg.alpha) * st.ema_value
                did_update = True
                filtered_out = round(float(st.ema_value), 6)
            else:
                # INVALID observation MUST NOT update numerical state (`ema_value` and `valid_window` untouched)
                st.invalid_count += 1
                st.consecutive_invalid += 1
                if st.last_valid_frame is None:
                    # Never observed validly -> explicitly missing
                    filtered_out = None
                else:
                    gap_frames = frame_idx - st.last_valid_frame
                    gap_sec = float(timestamp_sec) - float(st.last_valid_ts)
                    too_old_sec = self.cfg.max_stale_sec is not None and gap_sec > self.cfg.max_stale_sec
                    if gap_frames > self.cfg.max_stale_frames or too_old_sec:
                        # Expired beyond staleness horizon -> stop holding estimate (`None`) & reset filter value
                        is_stale = True
                        is_expired_stale = True
                        filtered_out = None
                        if gap_frames >= self.cfg.max_missed_frames_reset:
                            st.ema_value = None
                            st.valid_window.clear()
                    else:
                        # Within bounded staleness window -> mark stale explicitly
                        is_stale = True
                        filtered_out = round(float(st.ema_value), 6) if st.ema_value is not None else None

        frames_since_valid = (frame_idx - st.last_valid_frame) if st.last_valid_frame is not None else None
        time_since_valid = (
            round(float(timestamp_sec) - float(st.last_valid_ts), 4) if st.last_valid_ts is not None else None
        )

        return {
            "track_id": track_id,
            "mode": self.mode,
            "frame_idx": frame_idx,
            "timestamp_sec": round(float(timestamp_sec), 4),
            "raw_value": round(val_clean, 6) if val_clean is not None else None,
            "is_valid_input": valid_obs,
            "filtered_value": filtered_out,
            "did_update": did_update,
            "is_stale": is_stale,
            "is_expired": is_expired_stale,
            "track_age_frames": track_age_frames,
            "track_age_sec": round(track_age_sec, 4),
            "frames_since_last_valid": frames_since_valid,
            "time_since_last_valid_sec": time_since_valid,
            "consecutive_invalid_count": st.consecutive_invalid,
            "valid_count": st.valid_count,
            "invalid_count": st.invalid_count,
        }


def build_synthetic_temporal_sequences() -> Dict[str, List[Dict[str, Any]]]:
    """Build deterministic controlled sequences to evaluate stability, lag, dropouts, staleness, and resets."""
    rng = np.random.default_rng(77)
    fps = 15.0
    dt = 1.0 / fps
    seqs: Dict[str, List[Dict[str, Any]]] = {}

    # 1. Stable constant signal (true=10.0) + noise sigma=1.2
    s1 = []
    for i in range(40):
        s1.append({"frame": i, "ts": i * dt, "true": 10.0, "obs": 10.0 + float(rng.normal(0.0, 1.2)), "valid": True})
    seqs["stable_noisy"] = s1

    # 2. Abrupt step change (6.0 -> 18.0 at frame 15) + Gaussian noise to measure response lag & overshoot
    s2 = []
    for i in range(45):
        true_v = 6.0 if i < 15 else 18.0
        s2.append({"frame": i, "ts": i * dt, "true": true_v, "obs": true_v + float(rng.normal(0.0, 0.8)), "valid": True})
    seqs["abrupt_step"] = s2

    # 3. Intermittent invalid dropouts & corrupted values during invalid steps (12.0 + burst invalid glitches = 85.0 or 0.0)
    s3 = []
    invalid_idx = {6, 7, 14, 15, 22, 23, 24, 32}
    for i in range(40):
        is_v = i not in invalid_idx
        obs = (12.0 + float(rng.normal(0.0, 0.9))) if is_v else (95.0 if i % 2 == 0 else -15.0)
        s3.append({"frame": i, "ts": i * dt, "true": 12.0, "obs": obs, "valid": is_v})
    seqs["invalid_dropouts_with_glitches"] = s3

    # 4. Prolonged occlusion causing staleness expiry (frames 0..9 valid at 14.0, frames 10..20 invalid, 21..30 valid)
    s4 = []
    for i in range(32):
        is_v = not (10 <= i <= 20)
        obs = (14.0 + float(rng.normal(0.0, 0.5))) if is_v else None
        s4.append({"frame": i, "ts": i * dt, "true": 14.0, "obs": obs, "valid": is_v})
    seqs["prolonged_invalid_expiry"] = s4

    # 5. Track ID gap reset (episode 1 at 5.0 frames 0..9; absent frames 10..21; reappears as track ID at 20.0 frames 22..31)
    s5 = []
    for i in range(10):
        s5.append({"frame": i, "ts": i * dt, "true": 5.0, "obs": 5.0 + float(rng.normal(0.0, 0.4)), "valid": True})
    for i in range(22, 32):
        s5.append({"frame": i, "ts": i * dt, "true": 20.0, "obs": 20.0 + float(rng.normal(0.0, 0.4)), "valid": True})
    seqs["track_gap_reset"] = s5

    return seqs


def evaluate_temporal_filter_suite(cfg: Optional[TemporalFilterConfig] = None) -> Dict[str, Any]:
    """Evaluate all 4 filter modes across all 5 controlled synthetic sequences."""
    cfg = cfg or TemporalFilterConfig()
    seqs = build_synthetic_temporal_sequences()
    summary: Dict[str, Any] = {}

    for seq_name, items in seqs.items():
        seq_res: Dict[str, Any] = {}
        for mode in FILTER_MODES:
            flt = ValidityAwareTrackFilter(mode=mode, cfg=cfg)
            trace = []
            for item in items:
                out = flt.update(
                    track_id=1,
                    raw_value=item["obs"],
                    is_valid=item["valid"],
                    frame_idx=item["frame"],
                    timestamp_sec=item["ts"],
                )
                out["true_rel"] = item["true"]
                trace.append(out)

            # Metrics on emitted non-None values
            emitted = [t for t in trace if t["filtered_value"] is not None]
            emitted_valid_only = [t for t in trace if t["filtered_value"] is not None and t["is_valid_input"]]
            vals_all = [t["filtered_value"] for t in emitted]
            errs_all = [abs(t["filtered_value"] - t["true_rel"]) for t in emitted]
            errs_valid = [abs(t["filtered_value"] - t["true_rel"]) for t in emitted_valid_only]

            # Step-response lag (90% settling time after frame 15 for abrupt_step)
            lag_frames = None
            if seq_name == "abrupt_step":
                target90 = 6.0 + 0.90 * (18.0 - 6.0)  # 16.8
                for t in trace:
                    if t["frame_idx"] >= 15 and t["filtered_value"] is not None and t["filtered_value"] >= target90:
                        lag_frames = t["frame_idx"] - 15
                        break

            # Reset leakage check (on frame 22 right after track reset, distance from episode-2 true 20.0)
            reset_first_err = None
            if seq_name == "track_gap_reset":
                f22 = [t for t in trace if t["frame_idx"] == 22]
                if f22 and f22[0]["filtered_value"] is not None:
                    reset_first_err = round(abs(f22[0]["filtered_value"] - 20.0), 4)

            seq_res[mode] = {
                "emitted_steps": len(emitted),
                "total_steps": len(trace),
                "stale_steps": sum(1 for t in trace if t["is_stale"] and not t["is_expired"]),
                "expired_steps": sum(1 for t in trace if t["is_expired"]),
                "mae_emitted": round(float(np.mean(errs_all)), 4) if errs_all else None,
                "mae_valid_steps": round(float(np.mean(errs_valid)), 4) if errs_valid else None,
                "std_emitted": round(float(np.std(vals_all, ddof=1)), 4) if len(vals_all) > 1 else 0.0,
                "mean_abs_delta": round(float(np.mean(np.abs(np.diff(vals_all)))), 4) if len(vals_all) > 1 else 0.0,
                "step_lag_90pct_frames": lag_frames,
                "post_reset_first_abs_err": reset_first_err,
            }
        summary[seq_name] = seq_res
    return summary
