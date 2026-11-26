"""
========================================================================================
RoadVision AI -- Phase 7A: Research-Only Adaptive Depth Scheduler Video Demonstration
========================================================================================
Generates annotated research videos displaying:
  - Detector bounding boxes & ByteTrack IDs
  - Depth model & resolution (`DAv2 Small 336 fp16_weights`)
  - Depth keyframe indicator (`KEYFRAME` vs `SKIP`) & trigger reason
  - Fresh vs Cached / Invalid / Expired measurement state & age (ms)
  - Strictly RELATIVE cavity score & Time-Aware EMA (ZERO cm/mm/physical claims!)
  - Per-frame processing latency (ms) & cumulative depth inference savings (%)

Isolated from production `roadvision_engine.py`, `server.py`, and dashboard.
========================================================================================
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional

import cv2
import torch
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.roi_guided_segmentation import DEFAULT_DET_WEIGHTS
from experiments.optimized_relative_depth import OptimizedDepthConfig, build_and_sample_annular_region, fit_ransac_optimized
from experiments.core_aware_cavity_extraction import CoreExtractionConfig, extract_cavity_with_strategy
from experiments.track_depth_cache import TrackCacheConfig, TrackDepthCache
from experiments.adaptive_depth_scheduler import AdaptiveSchedulerConfig, AdaptiveDepthScheduler
from experiments.benchmark_adaptive_depth_scheduler import FROZEN_PHASE7A_CONFIG, FastDAv2Runner, BASE_DIR, VIZ_DIR


def render_scheduler_demo_video(
    video_rel_path: str,
    session_id: str,
    out_mp4_path: str,
    preview_png_path: Optional[str] = None,
    preview_frame_idx: int = 45,
    max_frames: int = 202,
    strategy: str = "D_track_aware",
    input_size: int = 336,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(DEFAULT_DET_WEIGHTS)
    runner = FastDAv2Runner(input_size=input_size, device=dev)
    det.predict(np.zeros((360, 640, 3), dtype=np.uint8), imgsz=640, conf=0.25, verbose=False)
    runner.warmup(360, 640)

    opt_cfg = OptimizedDepthConfig(**FROZEN_PHASE7A_CONFIG["ransac"])
    core_cfg = CoreExtractionConfig()
    cache = TrackDepthCache(session_id, "dav2_small", input_size, "fp16_weights", TrackCacheConfig(**FROZEN_PHASE7A_CONFIG["cache"]))
    scfg = AdaptiveSchedulerConfig(
        strategy=strategy,
        refresh_interval_sec=FROZEN_PHASE7A_CONFIG["scheduler"]["refresh_interval_sec"],
        max_measurement_age_sec=FROZEN_PHASE7A_CONFIG["scheduler"]["max_measurement_age_sec"],
        area_change_threshold=FROZEN_PHASE7A_CONFIG["scheduler"]["area_change_threshold"],
        centroid_shift_threshold=FROZEN_PHASE7A_CONFIG["scheduler"]["centroid_shift_threshold"],
        min_box_iou=FROZEN_PHASE7A_CONFIG["scheduler"]["min_box_iou"],
        invalid_retry_base_sec=FROZEN_PHASE7A_CONFIG["scheduler"]["invalid_retry_base_sec"],
        invalid_retry_max_sec=FROZEN_PHASE7A_CONFIG["scheduler"]["invalid_retry_max_sec"],
        min_ref_inlier_hold=FROZEN_PHASE7A_CONFIG["scheduler"]["min_ref_inlier_hold"],
        scene_change_l1_thresh=FROZEN_PHASE7A_CONFIG["scheduler"]["scene_change_l1_thresh"],
    )
    sched = AdaptiveDepthScheduler(scfg, cache)

    cap = cv2.VideoCapture(os.path.join(BASE_DIR, video_rel_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    state_colors = {
        "FRESH_VALID": (34, 197, 94),
        "CACHED_VALID": (235, 180, 30),
        "FRESH_INVALID": (60, 60, 239),
        "EXPIRED": (0, 120, 255),
        "NOT_EVALUATED": (160, 160, 160),
    }

    frames_proc = 0
    for fidx in range(max_frames):
        tf0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = fidx / fps
        res = det.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.50, imgsz=640, verbose=False)[0]
        boxes, tids = [], []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy().tolist()]
            tids = [int(t) for t in res.boxes.id.int().cpu().tolist()] if res.boxes.id is not None else [i + 1 for i in range(len(boxes))]

        dec = sched.decide_frame(fidx, ts, boxes, tids, frame_bgr=frame)
        track_outs = {}
        if dec["run_depth"]:
            pr = runner.predict(frame)
            dmap = pr.depth_map
            for idx_b, (bx, tid) in enumerate(zip(boxes, tids)):
                nbrs = [boxes[j] for j in range(len(boxes)) if j != idx_b]
                reg = build_and_sample_annular_region(dmap, bx, neighbor_boxes=nbrs, cfg=opt_cfg)
                fit = fit_ransac_optimized(dmap, reg, cfg=opt_cfg)
                o = extract_cavity_with_strategy(
                    dmap, bx, strategy="elliptical_core", neighbor_boxes=nbrs, img_bgr=frame,
                    opt_cfg=opt_cfg, core_cfg=core_cfg, precomputed_region=reg, precomputed_fit=fit,
                )
                track_outs[tid] = cache.record_fresh_measurement(
                    tid, f"{session_id}_{fidx}", fidx, ts, bx, o["is_valid"],
                    o["core_cavity_stats"]["scale_normalized_p75"],
                    o["core_cavity_stats"]["p75_residual"], None,
                    o["ref_diagnostics"], tuple(o["failure_reasons"]),
                )
        else:
            for tid in tids:
                track_outs[tid] = cache.read_non_keyframe_state(tid, fidx, ts)

        ftot_ms = (time.perf_counter() - tf0) * 1000.0
        sched.update_observed_latency(ftot_ms)
        frames_proc += 1

        vis = frame.copy()
        saved_pct = 100.0 * (1.0 - sched.depth_calls / max(1, frames_proc))
        # Header HUD overlay (research-only, strictly relative units)
        cv2.rectangle(vis, (0, 0), (W, 52), (18, 24, 34), -1)
        kf_txt = "KEYFRAME: DEPTH RUN" if dec["run_depth"] else "SKIP: CACHED / GATED"
        kf_col = (50, 220, 90) if dec["run_depth"] else (240, 190, 50)
        cv2.putText(vis, f"DAv2-S {input_size} FP16w | {strategy} | {kf_txt}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, kf_col, 1, cv2.LINE_AA)
        trig_s = "|".join(dec["frame_reasons"])[:38]
        cv2.putText(
            vis,
            f"Trig:{trig_s} | Lat:{ftot_ms:.1f}ms | Calls:{sched.depth_calls}/{frames_proc} (Saved {saved_pct:.1f}%) | RelUnits",
            (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 230, 240), 1, cv2.LINE_AA
        )

        for bx, tid in zip(boxes, tids):
            out = track_outs[tid]
            st_s = out["state"]
            col = state_colors.get(st_s, (200, 200, 200))
            x1, y1, x2, y2 = [int(round(v)) for v in bx]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            age_ms = int(round(out["cache_age_sec"] * 1000.0)) if out["cache_age_sec"] is not None else -1
            rel_txt = f"Rel:{out['relative_score']:.2f}" if out["relative_score"] is not None else st_s
            lbl = f"ID:{tid} {st_s} {rel_txt} age={age_ms}ms"
            cv2.putText(vis, lbl, (x1, max(66, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv2.LINE_AA)

        writer.write(vis)
        if preview_png_path and fidx == preview_frame_idx:
            cv2.imwrite(preview_png_path, vis)

    cap.release()
    writer.release()
    return {
        "session_id": session_id,
        "frames_rendered": frames_proc,
        "depth_calls": sched.depth_calls,
        "saved_pct": round(100.0 * (1.0 - sched.depth_calls / max(1, frames_proc)), 2),
        "out_mp4": out_mp4_path,
        "preview_png": preview_png_path,
    }


def main():
    demos = [
        render_scheduler_demo_video(
            "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
            "ruralRoad_demo",
            os.path.join(VIZ_DIR, "phase7a_demo_ruralRoad_dav2_336_track_aware.mp4"),
            preview_png_path=os.path.join(VIZ_DIR, "phase7a_demo_preview_rural_f0045.png"),
            preview_frame_idx=45,
            max_frames=202,
        ),
        render_scheduler_demo_video(
            "roadvision-dashboard/public/videos/raw_survey.mp4",
            "rawSurvey_demo_10fps",
            os.path.join(VIZ_DIR, "phase7a_demo_rawSurvey_dav2_336_track_aware.mp4"),
            preview_png_path=os.path.join(VIZ_DIR, "phase7a_demo_preview_survey_f0021.png"),
            preview_frame_idx=21,
            max_frames=160,
        ),
    ]
    print(json.dumps(demos, indent=2))


if __name__ == "__main__":
    main()
