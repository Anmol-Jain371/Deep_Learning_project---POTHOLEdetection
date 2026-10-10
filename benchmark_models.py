"""
========================================================================================
🔬 RoadVision AI — Model Benchmark & Validation Suite
========================================================================================
Compares:
  - Baseline Model: pothole_best.pt
  - Fine-Tuned Model: runs/detect/pothole_v8_upgraded/weights/best.pt

Evaluates on unseen phone video:
  1. Baseline vs fine-tuned detection metrics (Total raw box count)
  2. Mean/median/peak confidence
  3. Frames with detections (% coverage)
  4. False-positive / ephemeral detections (single-frame flashes)
  5. Detection flickering / dropouts (% frame dropout rate within tracks)
  6. ByteTrack ID count and fragmentation
  7. Persistent / unique hazard count (>= 2 frames & >= 5 frames)
  8. Processing time and FPS
  9. Annotated output video for BOTH models & Split-Screen Video
 10. Final conclusion stating which model performed better and why
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

    # Initialize VideoWriter
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

    print(f"\nEvaluating & Annotating: [{model_tag}] ({os.path.basename(model_path)})")
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
        annotated_frame = frame.copy()

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

                # Draw bounding box & label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"ID {tid} | {score:.2f}"
                cv2.putText(
                    annotated_frame, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2
                )

            if valid_in_frame > 0:
                frames_with_detections += 1

        # HUD Overlay Banner
        cv2.rectangle(annotated_frame, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(
            annotated_frame, f"{model_tag}", (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2
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
    out.release()
    total_time = time.time() - t0
    avg_fps = frame_idx / max(0.1, total_time)
    print(f"\nDone in {total_time:.1f}s ({avg_fps:.1f} avg FPS). Saved: {output_video_path}")

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


def generate_side_by_side_video(
    video_path: str,
    output_path: str,
    base_res: Dict[str, Any],
    fine_res: Dict[str, Any]
):
    """Creates a split-screen video comparing Baseline (Left) and Fine-Tuned (Right)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Keep native dimensions: stitch horizontally
    combined_w = W * 2
    combined_h = H

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))

    base_dets = base_res["frame_detections"]
    fine_dets = fine_res["frame_detections"]
    total_frames = min(len(base_dets), len(fine_dets))

    print(f"\n🎬 Rendering Split-Screen Comparison Video -> {output_path}...")

    frame_idx = 0
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        f_base = frame.copy()
        f_fine = frame.copy()

        # Draw baseline detections (Orange/Amber)
        for det in base_dets[frame_idx]:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(f_base, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = f"ID {det['tid']} | {det['conf']:.2f}"
            cv2.putText(f_base, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # Draw fine-tuned detections (Vibrant Green)
        for det in fine_dets[frame_idx]:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(f_fine, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {det['tid']} | {det['conf']:.2f}"
            cv2.putText(f_fine, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Add top banners
        cv2.rectangle(f_base, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(f_base, "BASELINE: pothole_best.pt", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 165, 255), 2)
        cv2.putText(f_base, f"Active: {len(base_dets[frame_idx])}", (W - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        cv2.rectangle(f_fine, (0, 0), (W, 36), (15, 15, 15), -1)
        cv2.putText(f_fine, "FINE-TUNED: best.pt (Epoch 5)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        cv2.putText(f_fine, f"Active: {len(fine_dets[frame_idx])}", (W - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Stitch horizontally
        combined = np.hstack([f_base, f_fine])
        out.write(combined)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Comparison Video saved successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RoadVision AI Model Comparative Benchmarking")
    parser.add_argument("--video", type=str, required=True, help="Path to unseen test video (.mp4)")
    parser.add_argument("--baseline", type=str, default="pothole_best.pt", help="Baseline weights")
    parser.add_argument("--finetuned", type=str, default=r"runs\detect\pothole_v8_upgraded\weights\best.pt", help="Fine-tuned weights")
    parser.add_argument("--tracker", type=str, default="road_tracker.yaml", help="Tracker config YAML")
    parser.add_argument("--conf", type=float, default=0.14, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or '0'")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[!] Error: Test video not found: {args.video}")
        sys.exit(1)

    os.makedirs(r"runs\benchmark", exist_ok=True)
    base_out_vid = r"runs\benchmark\baseline_annotated.mp4"
    fine_out_vid = r"runs\benchmark\finetuned_annotated.mp4"
    split_out_vid = r"runs\benchmark\split_screen_comparison.mp4"

    print("=" * 80)
    print("🚀 ROADVISION AI: BASELINE vs. FINE-TUNED BENCHMARK SUITE")
    print(f"Unseen Video : {args.video}")
    print(f"Baseline     : {args.baseline}")
    print(f"Fine-Tuned   : {args.finetuned}")
    print(f"Tracker      : {args.tracker}")
    print(f"Confidence   : {args.conf}")
    print(f"Resolution   : {args.imgsz}x{args.imgsz}")
    print("=" * 80)

    # 1. Run & Annotate Baseline
    base_res = analyze_and_render_model(
        model_path=args.baseline,
        video_path=args.video,
        output_video_path=base_out_vid,
        model_tag="BASELINE: pothole_best.pt",
        box_color=(0, 165, 255),
        tracker_cfg=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device
    )

    # 2. Run & Annotate Fine-Tuned
    fine_res = analyze_and_render_model(
        model_path=args.finetuned,
        video_path=args.video,
        output_video_path=fine_out_vid,
        model_tag="FINE-TUNED: best.pt (Epoch 5)",
        box_color=(0, 255, 0),
        tracker_cfg=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device
    )

    # 3. Generate Split-Screen Comparison Video
    generate_side_by_side_video(args.video, split_out_vid, base_res, fine_res)

    # 4. Formatted Markdown Report
    print("\n" + "=" * 80)
    print("📊 BENCHMARK METRICS SUMMARY")
    print("=" * 80)

    delta_dets = fine_res['total_raw_detections'] - base_res['total_raw_detections']
    delta_cov = fine_res['detection_coverage_pct'] - base_res['detection_coverage_pct']
    delta_mean_c = (fine_res['mean_confidence'] - base_res['mean_confidence']) * 100.0
    delta_med_c = (fine_res['median_confidence'] - base_res['median_confidence']) * 100.0
    delta_flicker = fine_res['flicker_pct'] - base_res['flicker_pct']
    delta_ids = fine_res['unique_bytetrack_ids'] - base_res['unique_bytetrack_ids']
    delta_v2 = fine_res['verified_hazards_2'] - base_res['verified_hazards_2']
    delta_p5 = fine_res['persistent_hazards_5'] - base_res['persistent_hazards_5']
    delta_single = fine_res['single_frame_fleeting'] - base_res['single_frame_fleeting']

    print(f"Total Detections     : Base={base_res['total_raw_detections']} | Fine={fine_res['total_raw_detections']} (Delta: {delta_dets:+d})")
    print(f"Detection Coverage   : Base={base_res['detection_coverage_pct']:.1f}% | Fine={fine_res['detection_coverage_pct']:.1f}% (Delta: {delta_cov:+.1f}%)")
    print(f"Mean Confidence      : Base={base_res['mean_confidence']*100:.1f}% | Fine={fine_res['mean_confidence']*100:.1f}% (Delta: {delta_mean_c:+.1f}%)")
    print(f"Median Confidence    : Base={base_res['median_confidence']*100:.1f}% | Fine={fine_res['median_confidence']*100:.1f}% (Delta: {delta_med_c:+.1f}%)")
    print(f"Flicker Dropout Rate : Base={base_res['flicker_pct']:.1f}% | Fine={fine_res['flicker_pct']:.1f}% (Delta: {delta_flicker:+.1f}%)")
    print(f"ByteTrack Track IDs  : Base={base_res['unique_bytetrack_ids']} | Fine={fine_res['unique_bytetrack_ids']} (Delta: {delta_ids:+d})")
    print(f"Verified (>=2 frames): Base={base_res['verified_hazards_2']} | Fine={fine_res['verified_hazards_2']} (Delta: {delta_v2:+d})")
    print(f"Persistent (>=5 frms): Base={base_res['persistent_hazards_5']} | Fine={fine_res['persistent_hazards_5']} (Delta: {delta_p5:+d})")
    print(f"Fleeting 1-frame     : Base={base_res['single_frame_fleeting']} | Fine={fine_res['single_frame_fleeting']} (Delta: {delta_single:+d})")
    print(f"Processing FPS       : Base={base_res['avg_fps']:.1f} FPS | Fine={fine_res['avg_fps']:.1f} FPS")
    print("=" * 80)

if __name__ == "__main__":
    main()
