"""
========================================================================================
🔬 RoadVision AI — 3-Model Empirical Benchmark Suite
========================================================================================
Compares on unseen Rangoon Road video (1,005 frames @ 25 FPS):
  1. Baseline: pothole_best.pt
  2. Exp #1:   runs/detect/pothole_v8_upgraded/weights/best.pt
  3. Exp #2:   runs/detect/pothole_combined_v2/weights/best.pt

Evaluates:
  - Raw detections
  - Detection frame coverage (% and count)
  - Mean & median confidence
  - Peak confidence
  - ByteTrack unique IDs
  - >= 2-frame verified hazards
  - >= 5-frame persistent hazards
  - Fleeting 1-frame detections
  - Flicker dropout rate
  - Processing FPS & elapsed time

Generates video artifacts:
  - runs/benchmark/baseline_annotated.mp4
  - runs/benchmark/finetuned_v2_annotated.mp4
  - runs/benchmark/baseline_vs_v2_split_screen.mp4
========================================================================================
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def analyze_and_render_model(
    model_path: str,
    video_path: str,
    output_video_path: str,
    model_tag: str,
    box_color: tuple = (0, 255, 0),
    tracker_cfg: str = "road_tracker.yaml",
    conf: float = 0.14,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Runs model + ByteTrack, gathers telemetry metrics, and renders individual annotated video."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    horizon_y = int(H * 0.40)

    # Initialize VideoWriter if output requested
    out = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    track_history: Dict[int, List[int]] = {}     # track_id -> list of frame indices
    track_confs: Dict[int, List[float]] = {}       # track_id -> list of confidences
    all_confs: List[float] = []
    frames_with_detections = 0
    total_raw_detections = 0
    frame_detections: List[List[Dict[str, Any]]] = []  # per frame: list of boxes

    frame_idx = 0
    t0 = time.time()

    print(f"\nEvaluating: [{model_tag}] ({os.path.basename(model_path)})")
    print(f"Video: {os.path.basename(video_path)} ({total_frames} frames @ {fps:.1f} FPS, {W}x{H})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        try:
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_cfg if os.path.exists(tracker_cfg) else "bytetrack.yaml",
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False
            )
        except Exception:
            results = model.predict(
                frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False
            )

        current_frame_dets = []
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
                    continue  # Ignore detections above road horizon

                valid_in_frame += 1
                total_raw_detections += 1
                all_confs.append(float(score))

                if tid not in track_history:
                    track_history[tid] = []
                    track_confs[tid] = []
                track_history[tid].append(frame_idx)
                track_confs[tid].append(float(score))

                current_frame_dets.append({
                    "box": (x1, y1, x2, y2),
                    "tid": tid,
                    "conf": float(score)
                })

                if annotated_frame is not None:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    label = f"ID {tid} | {score:.2f}"
                    cv2.putText(
                        annotated_frame, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2
                    )

            if valid_in_frame > 0:
                frames_with_detections += 1

        if annotated_frame is not None:
            # HUD Overlay Banner
            cv2.rectangle(annotated_frame, (0, 0), (W, 36), (15, 15, 15), -1)
            cv2.putText(
                annotated_frame, f"{model_tag}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, box_color, 2
            )
            stats_str = f"Frame: {frame_idx}/{total_frames} | Dets: {len(current_frame_dets)} | Tracks: {len(track_history)}"
            cv2.putText(
                annotated_frame, stats_str, (W - 380, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1
            )
            out.write(annotated_frame)

        frame_detections.append(current_frame_dets)

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            sys.stdout.write(f"\r  Frame {frame_idx}/{total_frames} ({frame_idx/max(0.1, elapsed):.1f} FPS)...")
            sys.stdout.flush()

    cap.release()
    if out is not None:
        out.release()
    total_time = time.time() - t0
    avg_fps = frame_idx / max(0.1, total_time)
    print(f"\nDone in {total_time:.1f}s ({avg_fps:.1f} avg FPS).")

    # Compute Flickering & Dropout Rate
    total_track_lifespan = 0
    total_dropouts = 0
    for tid, frames_seen in track_history.items():
        if len(frames_seen) >= 2:
            lifespan = frames_seen[-1] - frames_seen[0] + 1
            observed = len(frames_seen)
            dropouts = lifespan - observed
            total_track_lifespan += lifespan
            total_dropouts += dropouts

    flicker_pct = (total_dropouts / max(1, total_track_lifespan)) * 100.0 if total_track_lifespan > 0 else 0.0

    # Verified hazards: hazards observed across at least 2 frames and >= 5 frames
    verified_hazards_2 = sum(1 for f_list in track_history.values() if len(f_list) >= 2)
    persistent_hazards_5 = sum(1 for f_list in track_history.values() if len(f_list) >= 5)
    single_frame_fleeting = sum(1 for f_list in track_history.values() if len(f_list) == 1)

    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0
    median_conf = float(np.median(all_confs)) if all_confs else 0.0
    max_conf = float(np.max(all_confs)) if all_confs else 0.0

    return {
        "model_tag": model_tag,
        "model_name": os.path.basename(model_path),
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "detection_coverage_pct": (frames_with_detections / max(1, total_frames)) * 100.0,
        "total_raw_detections": total_raw_detections,
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "max_confidence": max_conf,
        "unique_bytetrack_ids": len(track_history),
        "verified_hazards_2": verified_hazards_2,
        "persistent_hazards_5": persistent_hazards_5,
        "single_frame_fleeting": single_frame_fleeting,
        "flicker_pct": flicker_pct,
        "total_time_s": total_time,
        "avg_fps": avg_fps,
        "output_video": output_video_path,
        "frame_detections": frame_detections
    }


def generate_split_screen_video(
    video_path: str,
    output_path: str,
    left_res: Dict[str, Any],
    right_res: Dict[str, Any],
    left_tag: str = "BASELINE: pothole_best.pt",
    right_tag: str = "EXP #2: pothole_combined_v2"
):
    """Creates a split-screen video comparing Left and Right model detections."""
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

    left_dets = left_res["frame_detections"]
    right_dets = right_res["frame_detections"]
    total_frames = min(len(left_dets), len(right_dets))

    print(f"\nRendering Split-Screen Video -> {output_path}...")

    frame_idx = 0
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        f_left = frame.copy()
        f_right = frame.copy()

        # Left (Amber/Orange)
        for det in left_dets[frame_idx]:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(f_left, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = f"ID {det['tid']} | {det['conf']:.2f}"
            cv2.putText(f_left, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # Right (Vibrant Cyan/Green)
        for det in right_dets[frame_idx]:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(f_right, (x1, y1), (x2, y2), (0, 255, 128), 2)
            label = f"ID {det['tid']} | {det['conf']:.2f}"
            cv2.putText(f_right, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)

        # Top Banners
        cv2.rectangle(f_left, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(f_left, left_tag, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 165, 255), 2)
        cv2.putText(f_left, f"Active: {len(left_dets[frame_idx])}", (W - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        cv2.rectangle(f_right, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(f_right, right_tag, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 128), 2)
        cv2.putText(f_right, f"Active: {len(right_dets[frame_idx])}", (W - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        combined = np.hstack([f_left, f_right])
        out.write(combined)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Comparison Video saved: {output_path}")


def main():
    video_path = "Watch Video_ Terrible potholes in Rangoon Road.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    base_model = "pothole_best.pt"
    exp1_model = r"runs\detect\pothole_v8_upgraded\weights\best.pt"
    exp2_model = r"runs\detect\pothole_combined_v2\weights\best.pt"

    base_vid = r"runs\benchmark\baseline_annotated.mp4"
    exp1_vid = r"runs\benchmark\exp1_annotated.mp4"
    exp2_vid = r"runs\benchmark\finetuned_v2_annotated.mp4"
    split_vid = r"runs\benchmark\baseline_vs_v2_split_screen.mp4"

    print("=" * 80)
    print("ROADVISION AI: 3-MODEL EMPIRICAL BENCHMARK")
    print(f"Video      : {video_path}")
    print(f"1. Baseline: {base_model}")
    print(f"2. Exp #1  : {exp1_model}")
    print(f"3. Exp #2  : {exp2_model}")
    print("=" * 80)

    # 1. Evaluate Baseline
    print("\n--- [1/3] Benchmarking Baseline Model ---")
    base_res = analyze_and_render_model(
        model_path=base_model,
        video_path=video_path,
        output_video_path=base_vid,
        model_tag="BASELINE: pothole_best.pt",
        box_color=(0, 165, 255)
    )

    # 2. Evaluate Exp #1
    print("\n--- [2/3] Benchmarking Exp #1 Model ---")
    exp1_res = analyze_and_render_model(
        model_path=exp1_model,
        video_path=video_path,
        output_video_path=exp1_vid,
        model_tag="EXP #1: pothole_v8_upgraded",
        box_color=(0, 255, 0)
    )

    # 3. Evaluate Exp #2 (Combined V2)
    print("\n--- [3/3] Benchmarking Exp #2 Model (Dataset Combined V2) ---")
    exp2_res = analyze_and_render_model(
        model_path=exp2_model,
        video_path=video_path,
        output_video_path=exp2_vid,
        model_tag="EXP #2: pothole_combined_v2",
        box_color=(0, 255, 128)
    )

    # 4. Generate Split-Screen Comparison: Baseline (Left) vs Exp #2 (Right)
    generate_split_screen_video(
        video_path=video_path,
        output_path=split_vid,
        left_res=base_res,
        right_res=exp2_res,
        left_tag="BASELINE: pothole_best.pt",
        right_tag="EXP #2: pothole_combined_v2"
    )

    # 5. Summary Table
    print("\n" + "=" * 80)
    print("3-MODEL BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Baseline':<14} | {'Exp #1':<14} | {'Exp #2 (V2)':<14}")
    print("-" * 80)
    print(f"{'Raw Detections':<28} | {base_res['total_raw_detections']:<14} | {exp1_res['total_raw_detections']:<14} | {exp2_res['total_raw_detections']:<14}")
    print(f"{'Detection Frame Coverage':<28} | {base_res['detection_coverage_pct']:.1f}% ({base_res['frames_with_detections']}){'':<4} | {exp1_res['detection_coverage_pct']:.1f}% ({exp1_res['frames_with_detections']}){'':<4} | {exp2_res['detection_coverage_pct']:.1f}% ({exp2_res['frames_with_detections']}){'':<4}")
    print(f"{'Mean Confidence':<28} | {base_res['mean_confidence']*100:.1f}%{'':<9} | {exp1_res['mean_confidence']*100:.1f}%{'':<9} | {exp2_res['mean_confidence']*100:.1f}%{'':<9}")
    print(f"{'Median Confidence':<28} | {base_res['median_confidence']*100:.1f}%{'':<9} | {exp1_res['median_confidence']*100:.1f}%{'':<9} | {exp2_res['median_confidence']*100:.1f}%{'':<9}")
    print(f"{'ByteTrack Total IDs':<28} | {base_res['unique_bytetrack_ids']:<14} | {exp1_res['unique_bytetrack_ids']:<14} | {exp2_res['unique_bytetrack_ids']:<14}")
    print(f"{'>= 2-Frame Hazards':<28} | {base_res['verified_hazards_2']:<14} | {exp1_res['verified_hazards_2']:<14} | {exp2_res['verified_hazards_2']:<14}")
    print(f"{'>= 5-Frame Hazards':<28} | {base_res['persistent_hazards_5']:<14} | {exp1_res['persistent_hazards_5']:<14} | {exp2_res['persistent_hazards_5']:<14}")
    print(f"{'Single-Frame Fleeting':<28} | {base_res['single_frame_fleeting']:<14} | {exp1_res['single_frame_fleeting']:<14} | {exp2_res['single_frame_fleeting']:<14}")
    print(f"{'Flicker Dropout Rate':<28} | {base_res['flicker_pct']:.1f}%{'':<9} | {exp1_res['flicker_pct']:.1f}%{'':<9} | {exp2_res['flicker_pct']:.1f}%{'':<9}")
    print(f"{'Processing FPS':<28} | {base_res['avg_fps']:.1f} FPS{'':<6} | {exp1_res['avg_fps']:.1f} FPS{'':<6} | {exp2_res['avg_fps']:.1f} FPS{'':<6}")
    print("=" * 80)

if __name__ == "__main__":
    main()
