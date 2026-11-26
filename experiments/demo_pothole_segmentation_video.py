"""
========================================================================================
RoadVision AI -- Phase 4A.3: Research-Only Pothole Segmentation Video Demo
========================================================================================
Standalone research script to overlay YOLOv8n-seg pothole masks, bounding boxes, and
confidence scores on a local road video while preserving original resolution and FPS.

STRICT SCOPE:
  - Research-only demonstration; NOT integrated with server.py, roadvision_engine.py,
    ByteTrack tracker, MiDaS depth, or planar calibration.
  - Reports per-frame raw detection counts and inference latency ONLY. Never claims
    unique physical pothole counts across frames without validated tracking.
========================================================================================
"""

import os
import json
import time
import argparse
from typing import Optional

import cv2
import torch
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "runs", "segment", "pothole440_yolov8n_seg_v1", "weights", "best.pt")


def annotate_video(
    video_path: str,
    output_path: str,
    weights_path: str = DEFAULT_WEIGHTS,
    conf: float = 0.50,
    iou_nms: float = 0.70,
    imgsz: int = 640,
    max_frames: Optional[int] = None,
) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    # Warmup pass on dummy frame for accurate CUDA latency statistics
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    _ = model.predict(source=dummy, imgsz=imgsz, conf=conf, iou=iou_nms, retina_masks=True, device=device, verbose=False)

    latencies_ms = []
    per_frame_counts = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou_nms,
                retina_masks=True,
                device=device,
                verbose=False,
            )

            overlay = frame.copy()
            n_det = 0
            r0 = results[0]

            if r0.boxes is not None and len(r0.boxes) > 0:
                boxes = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                n_det = len(boxes)

                if r0.masks is not None and r0.masks.data is not None and len(r0.masks.data) > 0:
                    m_np = r0.masks.data.cpu().numpy()
                    union_mask = np.zeros((height, width), dtype=bool)
                    for i in range(m_np.shape[0]):
                        m_i = m_np[i]
                        if m_i.shape[0] != height or m_i.shape[1] != width:
                            m_i = cv2.resize(m_i, (width, height), interpolation=cv2.INTER_LINEAR)
                        union_mask |= (m_i >= 0.5)

                    color_layer = overlay.copy()
                    color_layer[union_mask] = (0, 165, 255)  # Amber-orange BGR
                    overlay = cv2.addWeighted(overlay, 0.62, color_layer, 0.38, 0)

                    cnts, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, cnts, -1, (0, 220, 255), 2)

                for (x1, y1, x2, y2), sc in zip(boxes, confs):
                    p1 = (max(0, int(x1)), max(0, int(y1)))
                    p2 = (min(width - 1, int(x2)), min(height - 1, int(y2)))
                    cv2.rectangle(overlay, p1, p2, (0, 255, 128), 2)
                    label = f"pothole {sc:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    ly1 = max(th + 4, p1[1])
                    cv2.rectangle(overlay, (p1[0], ly1 - th - 4), (p1[0] + tw + 4, ly1 + 2), (0, 180, 90), -1)
                    cv2.putText(overlay, label, (p1[0] + 2, ly1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            per_frame_counts.append(n_det)

            hud = f"YOLOv8n-seg (Pothole440) | Frame {frame_idx} | Raw Frame Detections: {n_det} | {dt_ms:.1f} ms"
            cv2.rectangle(overlay, (6, 6), (min(width - 6, 620), 30), (15, 15, 20), -1)
            cv2.putText(overlay, hud, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

            writer.write(overlay)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    arr_ms = np.array(latencies_ms, dtype=float)
    arr_det = np.array(per_frame_counts, dtype=int)
    summary = {
        "input_video": os.path.abspath(video_path),
        "output_video": os.path.abspath(output_path),
        "weights_path": os.path.abspath(weights_path),
        "conf_threshold": conf,
        "iou_nms": iou_nms,
        "imgsz": imgsz,
        "device": str(device),
        "preserved_dimensions": f"{width}x{height}",
        "preserved_fps": round(float(fps), 3),
        "source_total_frames": total_src_frames,
        "frames_processed": frame_idx,
        "frames_with_detections": int((arr_det > 0).sum()) if frame_idx > 0 else 0,
        "total_per_frame_raw_detections_sum": int(arr_det.sum()) if frame_idx > 0 else 0,
        "mean_raw_detections_per_frame": round(float(arr_det.mean()), 3) if frame_idx > 0 else 0.0,
        "max_raw_detections_in_frame": int(arr_det.max()) if frame_idx > 0 else 0,
        "mean_frame_latency_ms": round(float(arr_ms.mean()), 2) if frame_idx > 0 else 0.0,
        "median_frame_latency_ms": round(float(np.median(arr_ms)), 2) if frame_idx > 0 else 0.0,
        "p95_frame_latency_ms": round(float(np.percentile(arr_ms, 95)), 2) if frame_idx > 0 else 0.0,
        "mean_throughput_fps": round(1000.0 / float(arr_ms.mean()), 2) if frame_idx > 0 and arr_ms.mean() > 0 else 0.0,
        "note": "Per-frame raw detection sum across frames is NOT a unique physical pothole count (no tracking enabled).",
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Research-Only Pothole Segmentation Video Demo")
    parser.add_argument("--video", type=str, required=True, help="Input local video path")
    parser.add_argument("--output", type=str, required=True, help="Output annotated MP4 path")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou-nms", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    args = parser.parse_args()

    summary = annotate_video(
        video_path=args.video,
        output_path=args.output,
        weights_path=args.weights,
        conf=args.conf,
        iou_nms=args.iou_nms,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
    )
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
