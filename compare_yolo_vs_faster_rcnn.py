"""
========================================================================================
🚗 RoadVision AI — Head-to-Head Benchmark: YOLOv8s vs Faster R-CNN
========================================================================================
Compares:
  - YOLOv8s (One-Stage Detector)
  - Faster R-CNN ResNet-50-FPN (Two-Stage Detector)
Evaluates on real test video footage (e.g. Rangoon Road / Dashcam clips):
  1. Inference Latency (ms) and Throughput (FPS)
  2. Total Detections & Confidence Distribution
  3. Frame Coverage %
  4. Memory and Parameter Footprint Comparison
Outputs a structured JSON summary and terminal comparative table.
========================================================================================
"""

import os
import sys
import time
import json
import argparse
import cv2
import torch
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from faster_rcnn_model import create_faster_rcnn_model
import torchvision.transforms.functional as TF


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8s vs Faster R-CNN")
    parser.add_argument("--video", type=str, default="Watch Video_ Terrible potholes in Rangoon Road.mp4", help="Test video path")
    parser.add_argument("--yolo-weights", type=str, default="runs/detect/pothole_combined_v2/weights/best.pt", help="Path to YOLO checkpoint")
    parser.add_argument("--rcnn-weights", type=str, default="runs/faster_rcnn/faster_rcnn_pothole_best.pth", help="Path to Faster R-CNN checkpoint")
    parser.add_argument("--frames", type=int, default=100, help="Max frames to benchmark")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output-json", type=str, default="runs/benchmark/yolo_vs_faster_rcnn.json", help="Path to output JSON")
    return parser.parse_args()


def benchmark_yolo(weights_path, video_path, max_frames=100, conf_thresh=0.25, device="cpu"):
    print(f"\n⚡ Benchmarking YOLOv8s on [{device.upper()}]...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found at: {weights_path}")

    model = YOLO(weights_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = 0
    total_time = 0.0
    detections = 0
    confidences = []
    frames_with_detections = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        t0 = time.time()
        res = model.predict(frame, conf=conf_thresh, verbose=False, device=device)
        total_time += (time.time() - t0)

        boxes = res[0].boxes
        if len(boxes) > 0:
            frames_with_detections += 1
            detections += len(boxes)
            confidences.extend(boxes.conf.cpu().numpy().tolist())

    cap.release()
    avg_fps = frame_count / max(1e-5, total_time)
    avg_latency_ms = (total_time / max(1, frame_count)) * 1000.0

    return {
        "model": "YOLOv8s (One-Stage)",
        "frames_tested": frame_count,
        "total_detections": detections,
        "frames_with_detections": frames_with_detections,
        "coverage_pct": (frames_with_detections / max(1, frame_count)) * 100.0,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(avg_fps, 2),
        "params_m": 11.13,
        "weights_size_mb": round(os.path.getsize(weights_path) / (1024 * 1024), 2)
    }


def benchmark_faster_rcnn(weights_path, video_path, max_frames=100, conf_thresh=0.25, device="cpu"):
    print(f"\n⚡ Benchmarking Faster R-CNN on [{device.upper()}]...")
    
    # Initialize model
    model = create_faster_rcnn_model(num_classes=2, pretrained=False)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint: {weights_path}")
    else:
        print(f"⚠️ Checkpoint not found at {weights_path}. Benchmarking architecture with base initialization.")

    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = 0
    total_time = 0.0
    detections = 0
    confidences = []
    frames_with_detections = 0

    with torch.no_grad():
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Convert BGR (OpenCV) -> RGB -> Tensor
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = TF.to_tensor(Image.fromarray(img_rgb)).to(device)

            t0 = time.time()
            preds = model([img_tensor])[0]
            total_time += (time.time() - t0)

            # Filter predictions
            scores = preds["scores"].cpu().numpy()
            valid_mask = scores >= conf_thresh
            valid_scores = scores[valid_mask]

            if len(valid_scores) > 0:
                frames_with_detections += 1
                detections += len(valid_scores)
                confidences.extend(valid_scores.tolist())

    cap.release()
    avg_fps = frame_count / max(1e-5, total_time)
    avg_latency_ms = (total_time / max(1, frame_count)) * 1000.0
    total_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    file_size_mb = round(os.path.getsize(weights_path) / (1024 * 1024), 2) if os.path.exists(weights_path) else 170.0

    return {
        "model": "Faster R-CNN ResNet-50-FPN (Two-Stage)",
        "frames_tested": frame_count,
        "total_detections": detections,
        "frames_with_detections": frames_with_detections,
        "coverage_pct": (frames_with_detections / max(1, frame_count)) * 100.0,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(avg_fps, 2),
        "params_m": round(total_params_m, 2),
        "weights_size_mb": file_size_mb
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    print("=" * 80)
    print("ROADVISION AI: ONE-STAGE (YOLOv8s) VS TWO-STAGE (FASTER R-CNN) BENCHMARK")
    print(f"Test Video : {args.video}")
    print(f"Test Frames: {args.frames}")
    print(f"Device     : {device.upper()}")
    print("=" * 80)

    yolo_res = benchmark_yolo(args.yolo_weights, args.video, max_frames=args.frames, conf_thresh=args.conf, device=device)
    rcnn_res = benchmark_faster_rcnn(args.rcnn_weights, args.video, max_frames=args.frames, conf_thresh=args.conf, device=device)

    # Display comparison
    print("\n" + "=" * 80)
    print("🏆 COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 80)
    header = f"{'Metric':<30} | {'YOLOv8s (One-Stage)':<22} | {'Faster R-CNN (Two-Stage)':<24}"
    print(header)
    print("-" * len(header))
    print(f"{'Architecture Paradigm':<30} | {'Single-shot grid':<22} | {'RPN + RoIAlign':<24}")
    print(f"{'Parameters (Millions)':<30} | {str(yolo_res['params_m']) + ' M':<22} | {str(rcnn_res['params_m']) + ' M':<24}")
    print(f"{'Model Size (MB)':<30} | {str(yolo_res['weights_size_mb']) + ' MB':<22} | {str(rcnn_res['weights_size_mb']) + ' MB':<24}")
    print(f"{'Average Latency':<30} | {str(yolo_res['avg_latency_ms']) + ' ms':<22} | {str(rcnn_res['avg_latency_ms']) + ' ms':<24}")
    print(f"{'Throughput (FPS)':<30} | {str(yolo_res['fps']) + ' FPS':<22} | {str(rcnn_res['fps']) + ' FPS':<24}")
    print(f"{'Detection Coverage %':<30} | {str(round(yolo_res['coverage_pct'], 1)) + '%' :<22} | {str(round(rcnn_res['coverage_pct'], 1)) + '%' :<24}")
    print(f"{'Total Defects Detected':<30} | {str(yolo_res['total_detections']):<22} | {str(rcnn_res['total_detections']):<24}")
    print("=" * 80)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"yolov8s": yolo_res, "faster_rcnn": rcnn_res}, f, indent=2)

    print(f"📁 Benchmark output logged to: {args.output_json}\n")


if __name__ == "__main__":
    main()

