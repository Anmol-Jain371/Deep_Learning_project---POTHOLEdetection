"""
========================================================================================
🔬 RoadVision AI — ByteTrack Tuning Experiment #2B: fuse_score = False
========================================================================================
Ablation Goal:
  Evaluate pure spatial IoU association by disabling score fusion (fuse_score: false).
  Isolates spatial continuity from confidence scores on Model #2 (median conf ~ 0.509).

Model:
  runs/detect/pothole_combined_v2/weights/best.pt (Model #2 Frozen)

Detector Hyperparameters (Constant):
  conf: 0.14
  iou: 0.45
  imgsz: 640

Video:
  Watch Video_ Terrible potholes in Rangoon Road.mp4 (1,005 frames @ 25 FPS)

Trackers Compared:
  - Control Tracker: road_tracker.yaml (fuse_score: true, match_thresh: 0.55, new_track: 0.16)
  - Exp #2B Tracker: road_tracker_exp2b.yaml (fuse_score: false, match_thresh: 0.55, new_track: 0.16)
========================================================================================
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def evaluate_tracker(
    model_path: str,
    video_path: str,
    tracker_cfg: str,
    output_video_path: str = None,
    tag: str = "TRACKER",
    box_color: tuple = (0, 255, 0),
    conf: float = 0.14,
    iou: float = 0.45,
    imgsz: int = 640
) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(tracker_cfg):
        raise FileNotFoundError(f"Tracker YAML not found: {tracker_cfg}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    horizon_y = int(H * 0.40)

    out = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    track_history: Dict[int, List[int]] = {}
    track_confs: Dict[int, List[float]] = {}
    all_confs: List[float] = []
    frames_with_detections = 0
    total_raw_detections = 0
    frame_dets: List[List[Dict[str, Any]]] = []

    frame_idx = 0
    t0 = time.time()

    print(f"\nRunning Tracker Evaluation: [{tag}]")
    print(f"Config: {tracker_cfg}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        try:
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )
        except Exception:
            results = None

        current_dets = []
        annotated_frame = frame.copy() if out is not None else None

        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            c_scores = results[0].boxes.conf.cpu().numpy()

            if getattr(results[0].boxes, 'id', None) is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = [i + 1 for i in range(len(boxes))]

            valid_in_frame = 0
            for box, tid, score in zip(boxes, track_ids, c_scores):
                x1, y1, x2, y2 = map(int, box)
                if y2 <= horizon_y:
                    continue

                valid_in_frame += 1
                total_raw_detections += 1
                all_confs.append(float(score))

                if tid not in track_history:
                    track_history[tid] = []
                    track_confs[tid] = []
                track_history[tid].append(frame_idx)
                track_confs[tid].append(float(score))

                current_dets.append({
                    "box": (x1, y1, x2, y2),
                    "tid": tid,
                    "conf": float(score)
                })

                if annotated_frame is not None:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    label = f"ID {tid} | {score:.2f}"
                    cv2.putText(
                        annotated_frame, label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2
                    )

            if valid_in_frame > 0:
                frames_with_detections += 1

        if annotated_frame is not None:
            cv2.rectangle(annotated_frame, (0, 0), (W, 36), (15, 15, 15), -1)
            cv2.putText(
                annotated_frame, f"{tag}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, box_color, 2
            )
            stats_str = f"Frame: {frame_idx}/{total_frames} | Dets: {len(current_dets)} | Tracks: {len(track_history)}"
            cv2.putText(
                annotated_frame, stats_str, (W - 360, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1
            )
            out.write(annotated_frame)

        frame_dets.append(current_dets)

        if frame_idx % 100 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            sys.stdout.write(f"\r  Frame {frame_idx}/{total_frames} ({frame_idx/max(0.1, elapsed):.1f} FPS)...")
            sys.stdout.flush()

    cap.release()
    if out is not None:
        out.release()

    total_time = time.time() - t0
    avg_fps = frame_idx / max(0.1, total_time)
    print(f"\nDone in {total_time:.1f}s ({avg_fps:.1f} avg FPS).")

    # Metrics computation
    total_lifespan = 0
    total_dropouts = 0
    for tid, frames_seen in track_history.items():
        if len(frames_seen) >= 2:
            lifespan = frames_seen[-1] - frames_seen[0] + 1
            observed = len(frames_seen)
            dropouts = lifespan - observed
            total_lifespan += lifespan
            total_dropouts += dropouts

    flicker_pct = (total_dropouts / max(1, total_lifespan)) * 100.0 if total_lifespan > 0 else 0.0

    verified_2 = sum(1 for f_list in track_history.values() if len(f_list) >= 2)
    persistent_5 = sum(1 for f_list in track_history.values() if len(f_list) >= 5)
    single_frame_blips = sum(1 for f_list in track_history.values() if len(f_list) == 1)

    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0
    median_conf = float(np.median(all_confs)) if all_confs else 0.0

    return {
        "tag": tag,
        "config_file": tracker_cfg,
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "detection_coverage_pct": (frames_with_detections / max(1, total_frames)) * 100.0,
        "total_raw_detections": total_raw_detections,
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "total_ids": len(track_history),
        "single_frame_blips": single_frame_blips,
        "verified_2": verified_2,
        "persistent_5": persistent_5,
        "flicker_pct": flicker_pct,
        "avg_fps": avg_fps,
        "total_time_s": total_time,
        "frame_dets": frame_dets
    }


def render_split_comparison(video_path, output_path, ctrl_res, tuned_res):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    combined_w = W * 2
    combined_h = H

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))

    c_dets = ctrl_res["frame_dets"]
    t_dets = tuned_res["frame_dets"]
    total = min(len(c_dets), len(t_dets))

    print(f"\nRendering Tracker Split-Screen -> {output_path}...")
    frame_idx = 0
    while cap.isOpened() and frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        f_left = frame.copy()
        f_right = frame.copy()

        # Left (Control - Orange)
        for det in c_dets[frame_idx]:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(f_left, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(f_left, f"ID {det['tid']}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # Right (Exp 2B - Green/Cyan)
        for det in t_dets[frame_idx]:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(f_right, (x1, y1), (x2, y2), (0, 255, 128), 2)
            cv2.putText(f_right, f"ID {det['tid']}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)

        # Header Banners
        cv2.rectangle(f_left, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(f_left, "CONTROL: fuse_score=True (m=0.55)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2)

        cv2.rectangle(f_right, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(f_right, "EXP #2B: fuse_score=False (Pure IoU)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 2)

        combined = np.hstack([f_left, f_right])
        out.write(combined)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Split-Screen Video saved: {output_path}")


def main():
    model_path = r"runs\detect\pothole_combined_v2\weights\best.pt"
    video_path = "Watch Video_ Terrible potholes in Rangoon Road.mp4"

    ctrl_cfg = "road_tracker.yaml"
    exp2b_cfg = "road_tracker_exp2b.yaml"

    ctrl_vid = r"runs\benchmark\tracker_control_annotated.mp4"
    exp2b_vid = r"runs\benchmark\tracker_exp2b_annotated.mp4"
    split_vid = r"runs\benchmark\tracker_control_vs_exp2b_split.mp4"

    print("=" * 80)
    print("ROADVISION AI: BYTETRACK EXPERIMENT #2B (fuse_score: False)")
    print(f"Model      : {model_path} (Model #2 Frozen)")
    print(f"Video      : {video_path}")
    print(f"Control    : {ctrl_cfg} (fuse_score=True)")
    print(f"Exp #2B    : {exp2b_cfg} (fuse_score=False)")
    print("=" * 80)

    # 1. Run Control Tracker
    ctrl_res = evaluate_tracker(
        model_path=model_path,
        video_path=video_path,
        tracker_cfg=ctrl_cfg,
        output_video_path=ctrl_vid,
        tag="CONTROL TRACKER (fuse=True)",
        box_color=(0, 165, 255)
    )

    # 2. Run Exp #2B Tracker
    exp2b_res = evaluate_tracker(
        model_path=model_path,
        video_path=video_path,
        tracker_cfg=exp2b_cfg,
        output_video_path=exp2b_vid,
        tag="EXP #2B (fuse=False)",
        box_color=(0, 255, 128)
    )

    # 3. Render Split Comparison
    render_split_comparison(video_path, split_vid, ctrl_res, exp2b_res)

    # 4. Save JSON results
    results_json_path = r"runs\benchmark\tracker_exp2b_results.json"
    export_data = {
        "control": {k: v for k, v in ctrl_res.items() if k != "frame_dets"},
        "exp2b": {k: v for k, v in exp2b_res.items() if k != "frame_dets"}
    }
    with open(results_json_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nSaved JSON results -> {results_json_path}")

    # 5. Formatted Summary Report
    print("\n" + "=" * 80)
    print("BYTETRACK EXPERIMENT #2B: RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Current Tracker':<16} | {'Exp #2B (fuse=False)':<16} | {'Delta':<12}")
    print("-" * 80)

    delta_ids = exp2b_res['total_ids'] - ctrl_res['total_ids']
    delta_1f = exp2b_res['single_frame_blips'] - ctrl_res['single_frame_blips']
    delta_2f = exp2b_res['verified_2'] - ctrl_res['verified_2']
    delta_5f = exp2b_res['persistent_5'] - ctrl_res['persistent_5']
    delta_cov = exp2b_res['detection_coverage_pct'] - ctrl_res['detection_coverage_pct']
    delta_flk = exp2b_res['flicker_pct'] - ctrl_res['flicker_pct']
    delta_conf = (exp2b_res['mean_confidence'] - ctrl_res['mean_confidence']) * 100.0

    print(f"{'Total Track IDs':<28} | {ctrl_res['total_ids']:<16} | {exp2b_res['total_ids']:<16} | {delta_ids:+d}")
    print(f"{'1-Frame Transient Tracks':<28} | {ctrl_res['single_frame_blips']:<16} | {exp2b_res['single_frame_blips']:<16} | {delta_1f:+d}")
    print(f"{'>= 2-Frame Tracks (Verified)':<28} | {ctrl_res['verified_2']:<16} | {exp2b_res['verified_2']:<16} | {delta_2f:+d}")
    print(f"{'>= 5-Frame Tracks (Persistent)':<28} | {ctrl_res['persistent_5']:<16} | {exp2b_res['persistent_5']:<16} | {delta_5f:+d}")
    print(f"{'Detection Frame Coverage':<28} | {ctrl_res['detection_coverage_pct']:.1f}% ({ctrl_res['frames_with_detections']}){'':<4} | {exp2b_res['detection_coverage_pct']:.1f}% ({exp2b_res['frames_with_detections']}){'':<4} | {delta_cov:+.1f}%")
    print(f"{'Detection Flicker Rate':<28} | {ctrl_res['flicker_pct']:.1f}%{'':<10} | {exp2b_res['flicker_pct']:.1f}%{'':<10} | {delta_flk:+.1f}%")
    print(f"{'Mean Confidence':<28} | {ctrl_res['mean_confidence']*100:.1f}%{'':<10} | {exp2b_res['mean_confidence']*100:.1f}%{'':<10} | {delta_conf:+.1f}%")
    print(f"{'Processing FPS':<28} | {ctrl_res['avg_fps']:.1f} FPS{'':<8} | {exp2b_res['avg_fps']:.1f} FPS{'':<8} | {exp2b_res['avg_fps']-ctrl_res['avg_fps']:+.1f} FPS")
    print("=" * 80)

if __name__ == "__main__":
    main()
