"""
========================================================================================
🔬 RoadVision AI — Final Benchmark: Model #2 vs RDD2022+Local Final Model
========================================================================================
Compares:
  - Model #2 (Domain-Adapted Baseline): runs/detect/pothole_combined_v2/weights/best.pt
  - Model #3 (RDD2022 India Clean Model): pothole_rdd_india_best.pt (or runs/detect/pothole_rdd_india/weights/best.pt)

On 100% Blind External Test Benchmark:
  - Video: Watch Video_ Terrible potholes in Rangoon Road.mp4 (1,005 frames @ 25 FPS)
  - Tracker: road_tracker.yaml (ByteTrack with fuse_score: False, match_thresh: 0.55)
  - Parameters: conf=0.14, iou=0.45, imgsz=640
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

def evaluate_model(
    model_path: str,
    video_path: str,
    tracker_cfg: str,
    output_video_path: str = None,
    tag: str = "MODEL",
    box_color: tuple = (0, 255, 0),
    conf: float = 0.14,
    iou: float = 0.45,
    imgsz: int = 640
) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
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

    print(f"\nEvaluating: [{tag}]")
    print(f"Weights : {model_path}")
    print(f"Tracker : {tracker_cfg}")

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
        "model_file": model_path,
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


def render_split_comparison(video_path, output_path, m2_res, rdd_res):
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

    m2_dets = m2_res["frame_dets"]
    rdd_dets = rdd_res["frame_dets"]
    total = min(len(m2_dets), len(rdd_dets))

    print(f"\nRendering Model Split-Screen -> {output_path}...")
    frame_idx = 0
    while cap.isOpened() and frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        f_left = frame.copy()
        f_right = frame.copy()

        # Left (Model #2 - Orange)
        for det in m2_dets[frame_idx]:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(f_left, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(f_left, f"ID {det['tid']} | {det['conf']:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # Right (RDD2022 Final Model - Cyan/Green)
        for det in rdd_dets[frame_idx]:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(f_right, (x1, y1), (x2, y2), (0, 255, 128), 2)
            cv2.putText(f_right, f"ID {det['tid']} | {det['conf']:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)

        # Header Banners
        cv2.rectangle(f_left, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(f_left, "MODEL #2: pothole_combined_v2", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2)

        cv2.rectangle(f_right, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(f_right, "FINAL MODEL: RDD2022 India + Local V1", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 2)

        combined = np.hstack([f_left, f_right])
        out.write(combined)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Comparison Video saved: {output_path}")


def main():
    video_path = "Watch Video_ Terrible potholes in Rangoon Road.mp4"
    tracker_cfg = "road_tracker.yaml"

    m2_weights = r"runs\detect\pothole_combined_v2\weights\best.pt"
    
    # Candidate RDD weights paths
    rdd_candidates = [
        "pothole_rdd_india_best.pt",
        r"runs\detect\pothole_rdd_india\weights\best.pt",
        r"runs\detect\pothole_rdd_india_best.pt"
    ]
    rdd_weights = None
    for cand in rdd_candidates:
        if os.path.exists(cand):
            rdd_weights = cand
            break

    if not rdd_weights:
        print("=" * 80)
        print("STATUS: Waiting for trained weights 'pothole_rdd_india_best.pt'.")
        print("Please place the downloaded weights in the project root or in runs/detect/pothole_rdd_india/weights/best.pt.")
        print("=" * 80)
        sys.exit(1)

    split_vid = r"runs\benchmark\model2_vs_rdd_split_screen.mp4"
    results_json = r"runs\benchmark\model2_vs_rdd_benchmark_results.json"

    print("=" * 80)
    print("ROADVISION AI: BLIND HELD-OUT BENCHMARK COMPARISON")
    print(f"Video   : {video_path} (100% UNTOUCHED TEST DATA)")
    print(f"Tracker : {tracker_cfg} (Production ByteTrack)")
    print(f"Model A : {m2_weights} (Model #2 Baseline)")
    print(f"Model B : {rdd_weights} (RDD2022 India Final Model)")
    print("=" * 80)

    # 1. Run Model #2
    m2_res = evaluate_model(
        model_path=m2_weights,
        video_path=video_path,
        tracker_cfg=tracker_cfg,
        output_video_path=r"runs\benchmark\model2_annotated.mp4",
        tag="MODEL #2: pothole_combined_v2",
        box_color=(0, 165, 255)
    )

    # 2. Run RDD2022 Model
    rdd_res = evaluate_model(
        model_path=rdd_weights,
        video_path=video_path,
        tracker_cfg=tracker_cfg,
        output_video_path=r"runs\benchmark\rdd_annotated.mp4",
        tag="FINAL: RDD2022 India + Local V1",
        box_color=(0, 255, 128)
    )

    # 3. Split Screen Comparison
    render_split_comparison(video_path, split_vid, m2_res, rdd_res)

    # 4. Save JSON
    export_data = {
        "model_2": {k: v for k, v in m2_res.items() if k != "frame_dets"},
        "model_rdd": {k: v for k, v in rdd_res.items() if k != "frame_dets"}
    }
    with open(results_json, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nSaved benchmark telemetry -> {results_json}")

    # 5. Formatted Summary Report
    print("\n" + "=" * 80)
    print("BLIND TEST BENCHMARK RESULTS: MODEL #2 vs RDD2022 INDIA FINAL")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Model #2 (Baseline)':<20} | {'RDD2022 Final Model':<20} | {'Delta':<12}")
    print("-" * 80)

    delta_ids = rdd_res['total_ids'] - m2_res['total_ids']
    delta_1f = rdd_res['single_frame_blips'] - m2_res['single_frame_blips']
    delta_2f = rdd_res['verified_2'] - m2_res['verified_2']
    delta_5f = rdd_res['persistent_5'] - m2_res['persistent_5']
    delta_cov = rdd_res['detection_coverage_pct'] - m2_res['detection_coverage_pct']
    delta_flk = rdd_res['flicker_pct'] - m2_res['flicker_pct']
    delta_conf = (rdd_res['mean_confidence'] - m2_res['mean_confidence']) * 100.0

    print(f"{'Total Track IDs':<28} | {m2_res['total_ids']:<20} | {rdd_res['total_ids']:<20} | {delta_ids:+d}")
    print(f"{'1-Frame Transient Tracks':<28} | {m2_res['single_frame_blips']:<20} | {rdd_res['single_frame_blips']:<20} | {delta_1f:+d}")
    print(f"{'>= 2-Frame Tracks (Verified)':<28} | {m2_res['verified_2']:<20} | {rdd_res['verified_2']:<20} | {delta_2f:+d}")
    print(f"{'>= 5-Frame Tracks (Persistent)':<28} | {m2_res['persistent_5']:<20} | {rdd_res['persistent_5']:<20} | {delta_5f:+d}")
    print(f"{'Detection Frame Coverage':<28} | {m2_res['detection_coverage_pct']:.1f}% ({m2_res['frames_with_detections']}){'':<8} | {rdd_res['detection_coverage_pct']:.1f}% ({rdd_res['frames_with_detections']}){'':<8} | {delta_cov:+.1f}%")
    print(f"{'Detection Flicker Rate':<28} | {m2_res['flicker_pct']:.1f}%{'':<14} | {rdd_res['flicker_pct']:.1f}%{'':<14} | {delta_flk:+.1f}%")
    print(f"{'Mean Confidence':<28} | {m2_res['mean_confidence']*100:.1f}%{'':<14} | {rdd_res['mean_confidence']*100:.1f}%{'':<14} | {delta_conf:+.1f}%")
    print(f"{'Processing FPS':<28} | {m2_res['avg_fps']:.1f} FPS{'':<12} | {rdd_res['avg_fps']:.1f} FPS{'':<12} | {rdd_res['avg_fps']-m2_res['avg_fps']:+.1f} FPS")
    print("=" * 80)

if __name__ == "__main__":
    main()
