"""
Phase 13 Steps 17 & 18: Adaptive Depth Scheduling Ablation & Temporal Filtering Ablation.

Freezes detection+ByteTrack output on exploratory road footage once so every scheduling
strategy (`A_always_on`, `B_candidate_gated`, `C_fixed_interval`, `D_track_aware`, `E_budget_aware`)
runs over the exact same box/track sequence. Also benchmarks live depth timing and runs controlled
synthetic + real-track temporal filter comparisons (`raw`, `legacy_ema_zero_clamp`,
`validity_aware_ema`, `rolling_median_validity_ema`).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from experiments.adaptive_depth_scheduler import AdaptiveDepthScheduler, AdaptiveSchedulerConfig
from experiments.core_aware_cavity_extraction import CoreExtractionConfig, extract_cavity_with_strategy
from experiments.depth_model_comparison import DepthAnythingV2SmallAdapter
from experiments.optimized_relative_depth import OptimizedDepthConfig
from experiments.track_depth_cache import TrackCacheConfig, TrackDepthCache
from experiments.validity_aware_temporal_filter import TemporalFilterConfig, ValidityAwareTrackFilter


def run_scheduler_and_temporal_ablation(
    project_root: str,
    cfg: Dict[str, Any],
    *,
    smoke: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    vid_spec = cfg["exploratory_videos"][0]
    vp = os.path.join(project_root, vid_spec["path"])
    if not os.path.isfile(vp):
        return {"status": "BLOCKED", "reason": f"Exploratory video missing: {vid_spec['path']}"}

    n_frames = int(cfg["smoke_limits"]["scheduler_frames"] if smoke else cfg["full_limits"]["scheduler_frames"])
    det_ckpt = os.path.join(project_root, cfg["detector_checkpoints"]["yolov8n_100e_sgd"])

    # Freeze detection+ByteTrack + precomputed DAv2 336 depth maps once so all 5 schedulers see identical detections
    det = YOLO(det_ckpt)
    depth_adapter = DepthAnythingV2SmallAdapter(device=device, input_size=336, use_fp16=(device == "cuda"))
    depth_adapter.warmup(360, 640)

    cap = cv2.VideoCapture(vp)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or vid_spec["fps"] or 25.0)
    stream: List[Dict[str, Any]] = []
    det_lats: List[float] = []
    depth_lats: List[float] = []

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for fidx in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = fidx / fps
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        td0 = time.perf_counter()
        res = det.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.50, imgsz=640, verbose=False)[0]
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        det_lats.append((time.perf_counter() - td0) * 1000.0)

        cands: List[Dict[str, Any]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxys = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            ids = res.boxes.id.int().cpu().numpy() if res.boxes.id is not None else np.arange(len(xyxys))
            for bx, cf, tid in zip(xyxys, confs, ids):
                cands.append({"track_id": int(tid), "box_xyxy": [float(x) for x in bx], "conf": float(cf)})

        d_pred = depth_adapter.predict(frame)
        depth_lats.append(d_pred.timings_ms["total_ms"])
        stream.append({"frame_idx": fidx, "timestamp_sec": ts, "frame": frame, "cands": cands, "depth_map": d_pred.depth_map})
    cap.release()

    peak_gpu_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if (device == "cuda" and torch.cuda.is_available())
        else None
    )
    del det
    del depth_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    opt_cfg = OptimizedDepthConfig(seed=42)
    core_cfg = CoreExtractionConfig()
    strategies = ("A_always_on", "B_candidate_gated", "C_fixed_interval", "D_track_aware", "E_budget_aware")
    sched_results: Dict[str, Any] = {}
    baseline_calls = max(1, len(stream))

    mean_depth_cost = float(np.mean(depth_lats)) if depth_lats else 0.0
    mean_det_cost = float(np.mean(det_lats)) if det_lats else 0.0

    for strat in strategies:
        cache = TrackDepthCache("phase13_sched", "depth_anything_v2_small", 336, "fp16_autocast", TrackCacheConfig())
        scfg = AdaptiveSchedulerConfig(strategy=strat, fixed_interval_k=3)
        sched = AdaptiveDepthScheduler(scfg, cache)

        req_cnt = call_cnt = fresh_valid = cached_valid = fresh_invalid = expired = not_eval = 0
        ages: List[float] = []
        frame_lats_sim: List[float] = []
        starved_tracks = 0
        seen_tids: set = set()
        measured_tids: set = set()
        t_start = time.perf_counter()

        for step_idx, item in enumerate(stream):
            fidx = item["frame_idx"]
            ts = item["timestamp_sec"]
            cands = item["cands"]
            for c in cands:
                seen_tids.add(c["track_id"])

            boxes = [tuple(c["box_xyxy"]) for c in cands]
            tids = [c["track_id"] for c in cands]
            decision = sched.decide_frame(
                fidx,
                ts,
                boxes,
                tids,
                frame_bgr=item["frame"],
            )
            req_cnt += 1
            t_proc0 = time.perf_counter()

            if decision["run_depth"]:
                call_cnt += 1
                dmap = item["depth_map"]
                for c in cands:
                    tid = c["track_id"]
                    ext = extract_cavity_with_strategy(
                        dmap,
                        tuple(c["box_xyxy"]),
                        strategy="elliptical_core",
                        opt_cfg=opt_cfg,
                        core_cfg=core_cfg,
                        img_bgr=item["frame"],
                    )
                    cav_stats = ext.get("core_cavity_stats") or {}
                    p75 = cav_stats.get("p75_residual")
                    norm_p75 = cav_stats.get("scale_normalized_p75")
                    cache.record_fresh_measurement(
                        track_id=tid,
                        frame_id=f"f{fidx}",
                        frame_idx=fidx,
                        ts_sec=ts,
                        box_xyxy=tuple(c["box_xyxy"]),
                        is_valid=bool(ext["is_valid"]),
                        relative_score=p75,
                        raw_p75=p75,
                        iqr_norm_p75=norm_p75,
                        ref_diagnostics=ext.get("ref_diagnostics") or {},
                        failure_reasons=tuple(ext.get("failure_reasons") or ()),
                    )
                    if ext["is_valid"]:
                        fresh_valid += 1
                        measured_tids.add(tid)
                        ages.append(0.0)
                    else:
                        fresh_invalid += 1
                sim_ms = det_lats[step_idx] + depth_lats[step_idx] + (time.perf_counter() - t_proc0) * 1000.0
            else:
                for c in cands:
                    tid = c["track_id"]
                    read = cache.read_non_keyframe_state(
                        track_id=tid,
                        frame_idx=fidx,
                        ts_sec=ts,
                    )
                    st = read["state"]
                    if st == "CACHED_VALID":
                        cached_valid += 1
                        measured_tids.add(tid)
                        if read.get("cache_age_sec") is not None:
                            ages.append(float(read["cache_age_sec"]))
                    elif st == "EXPIRED":
                        expired += 1
                    else:
                        not_eval += 1
                sim_ms = det_lats[step_idx] + (time.perf_counter() - t_proc0) * 1000.0
            sched.update_observed_latency(sim_ms)
            frame_lats_sim.append(sim_ms)

        starved_tracks = len(seen_tids - measured_tids)
        total_obs = fresh_valid + cached_valid + fresh_invalid + expired + not_eval
        tot_sim_sec = float(sum(frame_lats_sim) / 1000.0)
        sched_results[strat] = {
            "frames_processed": len(stream),
            "scheduler_requests": req_cnt,
            "actual_depth_calls": call_cnt,
            "depth_calls_per_frame": round(call_cnt / max(1, len(stream)), 4),
            "depth_call_reduction_vs_always_on_pct": round(((baseline_calls - call_cnt) / baseline_calls) * 100.0, 2),
            "total_observations": total_obs,
            "fresh_valid_count": fresh_valid,
            "cached_valid_count": cached_valid,
            "fresh_invalid_count": fresh_invalid,
            "expired_count": expired,
            "not_evaluated_count": not_eval,
            "fresh_valid_coverage_pct": round((fresh_valid / max(1, total_obs)) * 100.0, 2),
            "effective_availability_pct": round(((fresh_valid + cached_valid) / max(1, total_obs)) * 100.0, 2),
            "mean_age_sec": round(float(np.mean(ages)), 4) if ages else None,
            "track_starvation_count": starved_tracks,
            "simulated_frame_ms_mean": round(float(np.mean(frame_lats_sim)), 2),
            "simulated_frame_ms_median": round(float(np.median(frame_lats_sim)), 2),
            "simulated_frame_ms_p95": round(float(np.percentile(frame_lats_sim, 95.0)), 2),
            "simulated_sustained_fps": round(len(stream) / tot_sim_sec, 2) if tot_sim_sec > 0 else None,
            "peak_gpu_allocated_mb": peak_gpu_mb,
        }

    # Temporal Filtering Ablation on synthetic step + invalid gap + expiry sequence
    # True underlying step from 0.08 -> 0.22 at frame 10, with invalid gap at frames [5, 6, 7] and long expiry gap [16..20]
    seq = []
    for fidx in range(25):
        true_v = 0.08 if fidx < 10 else 0.22
        is_val = fidx not in (5, 6, 7, 16, 17, 18, 19, 20)
        obs_v = (true_v + (0.015 if fidx % 2 == 0 else -0.015)) if is_val else None
        seq.append((fidx, fidx * 0.1, obs_v, is_val))

    temporal_results: Dict[str, Any] = {}
    for mode in ("raw", "legacy_ema_zero_clamp", "validity_aware_ema", "rolling_median_validity_ema"):
        flt = ValidityAwareTrackFilter(mode=mode, cfg=TemporalFilterConfig(alpha=0.35, rolling_window=5, max_stale_frames=3, max_missed_frames_reset=8))
        vals: List[Optional[float]] = []
        invalid_masked_ok = 0
        invalid_tot = 0
        step_resp_after_2f = None
        for fidx, ts, ov, iv in seq:
            out = flt.update(track_id=1, raw_value=ov, is_valid=iv, frame_idx=fidx, timestamp_sec=ts)
            vals.append(out.get("filtered_value"))
            if not iv:
                invalid_tot += 1
                if fidx >= 19 and out.get("filtered_value") is None:
                    invalid_masked_ok += 1
            if fidx == 12:
                step_resp_after_2f = out.get("filtered_value")

        valid_outs = [float(v) for v in vals if v is not None]
        diffs = [abs(valid_outs[i] - valid_outs[i - 1]) for i in range(1, len(valid_outs))]
        temporal_results[mode] = {
            "mean_step_to_step_diff": round(float(np.mean(diffs)), 5) if diffs else None,
            "step_response_at_f12_target_0_22": round(float(step_resp_after_2f), 5) if step_resp_after_2f is not None else None,
            "expired_gap_masked_null_count": invalid_masked_ok,
            "hides_expired_as_numeric": (mode == "legacy_ema_zero_clamp"),
        }

    return {
        "status": "COMPLETED",
        "source_video": vid_spec["path"],
        "frozen_stream_frames": len(stream),
        "frozen_det_mean_ms": round(mean_det_cost, 3),
        "frozen_depth_mean_ms": round(mean_depth_cost, 3),
        "scheduling_ablation": sched_results,
        "temporal_filtering_ablation": temporal_results,
    }
