"""
experiments/benchmark_phase8a_integration.py
============================================
Phase 8A Integration Benchmark — RoadVision AI 2.0 Modular Pipeline

Evaluates all 3 operating modes (live, balanced, survey) across representative
video clips covering rural roads, urban streets, and survey footage.

Outputs timing metrics, depth scheduling statistics, and measurement coverage
to runs/benchmark/phase8a_integration_results.json.

DISCLAIMER: All evaluations are exploratory. Relative cavity scores are uncalibrated
inverse-depth residuals. No physical metric cm values are computed or reported.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_v2.config import load_v2_config
from roadvision_v2.pipeline import RoadVisionV2Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_phase8a")

# ---------------------------------------------------------------------------
# Video clips for benchmarking
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_VIDEOS = [
    {
        "id": "ruralRoad_202f",
        "path": str(_ROOT / "uploads" / "upload_1794847200_ruralRoad_potHoles.mp4"),
        "max_frames": 202,
        "description": "Rural road with potholes, 25fps, 640x360",
        "label": "exploratory",
    },
    {
        "id": "urban_rangoon_200f",
        "path": str(_ROOT / "Watch Video_ Terrible potholes in Rangoon Road.mp4"),
        "max_frames": 200,
        "description": "Urban potholes, Rangoon Road, 25fps",
        "label": "exploratory",
    },
    {
        "id": "rawSurvey_160f",
        "path": str(_ROOT / "roadvision-dashboard" / "public" / "videos" / "raw_survey.mp4"),
        "max_frames": 160,
        "description": "Survey footage, variable FPS, 1280x720",
        "label": "exploratory",
    },
]

_MODES = ["live", "balanced", "survey"]


def _run_benchmark_pass(
    mode: str,
    video: Dict[str, Any],
    output_base: Path,
) -> Dict[str, Any]:
    """Run one (mode, video) pass and return benchmark result dict."""
    vid_path = video["path"]
    if not os.path.exists(vid_path):
        logger.warning("Video not found, skipping: %s", vid_path)
        return {"skipped": True, "reason": f"file_not_found: {vid_path}"}

    cfg = load_v2_config(
        config_path=str(_ROOT / "config" / "roadvision_v2.yaml"),
        mode=mode,
        check_assets=True,
    )

    pipeline = RoadVisionV2Pipeline(cfg=cfg)

    out_dir = output_base / f"{mode}_{video['id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-frame timing collection ---
    frame_times: List[float] = []
    depth_inf_times: List[float] = []
    detect_times: List[float] = []
    states_hist: Dict[str, int] = {}
    n_hazards_total = 0

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return {"skipped": True, "reason": "cannot_open_video"}

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t_total_start = time.perf_counter()
    frame_idx = 0
    max_frames = video.get("max_frames")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        ts_sec = float(frame_idx) / max(fps_src, 1.0)
        t_f = time.perf_counter()
        hazards, perf = pipeline.process_frame(frame, frame_idx, ts_sec)
        frame_elapsed = (time.perf_counter() - t_f) * 1000.0

        frame_times.append(perf.total_frame_ms)
        depth_inf_times.append(perf.depth_inf_ms)
        detect_times.append(perf.detect_track_ms)

        for hz in hazards:
            st = hz.measurement.measurement_state
            states_hist[st] = states_hist.get(st, 0) + 1
            n_hazards_total += 1

        frame_idx += 1

    cap.release()
    t_total = time.perf_counter() - t_total_start

    # --- Compute statistics ---
    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {}
        a = np.array(arr, dtype=np.float64)
        return {
            "mean": round(float(np.mean(a)), 3),
            "median": round(float(np.median(a)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "max": round(float(np.max(a)), 3),
            "min": round(float(np.min(a)), 3),
        }

    deadline_ms = 1000.0 / max(fps_src, 1.0)
    deadline_misses = sum(1 for t in frame_times if t > deadline_ms)
    end_fps = frame_idx / max(t_total, 1e-6)

    fresh_valid = states_hist.get("FRESH_VALID", 0)
    cached_valid = states_hist.get("CACHED_VALID", 0)
    fresh_invalid = states_hist.get("FRESH_INVALID", 0)
    not_evaluated = states_hist.get("NOT_EVALUATED", 0)
    expired = states_hist.get("EXPIRED", 0)

    result = {
        "mode": mode,
        "video_id": video["id"],
        "video_description": video["description"],
        "label": video.get("label", "exploratory"),
        "source_fps": fps_src,
        "source_wh": [W, H],
        "frames_processed": frame_idx,
        "elapsed_total_sec": round(t_total, 3),
        "end_to_end_fps": round(end_fps, 2),
        "frame_ms": _stats(frame_times),
        "depth_inf_ms": _stats([t for t in depth_inf_times if t > 0.0]),
        "detect_track_ms": _stats(detect_times),
        "deadline_ms_target": round(deadline_ms, 2),
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": round(deadline_misses / max(frame_idx, 1), 4),
        "depth_enabled": cfg.depth.enabled,
        "depth_model_id": pipeline._depth_model_id,
        "depth_input_size": pipeline._depth_input_size,
        "scheduler_strategy": cfg.scheduler.strategy,
        "depth_calls": pipeline._scheduler.depth_calls,
        "budget_skips": pipeline._scheduler.budget_skips,
        "depth_invocation_rate": round(pipeline._scheduler.depth_calls / max(frame_idx, 1), 4),
        "total_hazard_observations": n_hazards_total,
        "unique_tracks": len(pipeline._track_registry),
        "measurement_state_histogram": states_hist,
        "fresh_valid_observations": fresh_valid,
        "cached_valid_observations": cached_valid,
        "fresh_invalid_observations": fresh_invalid,
        "not_evaluated_observations": not_evaluated,
        "expired_observations": expired,
        "fresh_valid_coverage_pct": round(100.0 * fresh_valid / max(n_hazards_total, 1), 2),
        "cached_availability_pct": round(100.0 * cached_valid / max(n_hazards_total, 1), 2),
    }

    # Try to get GPU peak memory
    try:
        import torch
        if torch.cuda.is_available():
            result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    logger.info(
        "[%s | %s] done | %d frames @ %.2f fps | depth_calls=%d (%.0f%%) | deadline_misses=%d",
        mode, video["id"], frame_idx, end_fps,
        pipeline._scheduler.depth_calls,
        pipeline._scheduler.depth_calls / max(frame_idx, 1) * 100,
        deadline_misses,
    )
    return result


def main() -> None:
    output_base = _ROOT / "runs" / "benchmark"
    output_base.mkdir(parents=True, exist_ok=True)
    viz_base = output_base / "phase8a_viz"
    viz_base.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for mode in _MODES:
        for video in _VIDEOS:
            logger.info("=" * 60)
            logger.info("Benchmarking: mode=%s video=%s", mode, video["id"])
            try:
                result = _run_benchmark_pass(mode, video, viz_base)
                all_results.append(result)
            except Exception as exc:
                logger.error("Failed [%s | %s]: %s", mode, video["id"], exc)
                all_results.append({
                    "mode": mode,
                    "video_id": video["id"],
                    "error": str(exc),
                    "skipped": True,
                })

    # Write results
    out_path = output_base / "phase8a_integration_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info("Results written → %s", out_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("Phase 8A Integration Benchmark Summary (exploratory)")
    print("=" * 80)
    hdr = f"{'Mode':<10} {'Video':<22} {'FPS':>7} {'Depth%':>8} {'Deadline':>10} {'FV%':>7}"
    print(hdr)
    print("-" * 80)
    for r in all_results:
        if r.get("skipped"):
            print(f"  {'SKIPPED':} {r.get('video_id','?'):<22} — {r.get('reason','')}")
            continue
        print(
            f"{r['mode']:<10} {r['video_id']:<22} "
            f"{r['end_to_end_fps']:>7.2f} "
            f"{r['depth_invocation_rate']*100:>7.1f}% "
            f"{r['deadline_misses']:>8}ms/f "
            f"{r['fresh_valid_coverage_pct']:>6.1f}%"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
