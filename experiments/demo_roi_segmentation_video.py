"""
========================================================================================
RoadVision AI -- Phase 4A.4: Research-Only ROI-Guided Segmentation Video Demonstration
========================================================================================
Standalone experimental video renderer showing:
  - Detector bounding boxes + confidence
  - Accepted ROI-guided segmentation masks (emerald green overlay)
  - Bounding-box fallback when quality assessment rejects ROI masks (amber box + reason)
  - Per-frame stage & combined latency HUD

STRICT SCOPE:
  - Research-only offline visualization; NOT integrated into server.py or roadvision_engine.py.
  - Reports raw per-frame counts only (NOT unique physical potholes across frames).
========================================================================================
"""

import os
import sys
import json
import argparse
from collections import Counter
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.roi_guided_segmentation import (
    ROISegConfig,
    ROIGuidedSegmenter,
    DEFAULT_DET_WEIGHTS,
    DEFAULT_SEG_WEIGHTS,
)


def render_roi_video(
    video_path: str,
    output_path: str,
    batch_roi: bool = False,
    det_conf: float = 0.25,
    seg_conf: float = 0.25,
    max_frames: Optional[int] = None,
) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cfg = ROISegConfig(det_conf=det_conf, seg_conf=seg_conf, batch_roi=batch_roi)
    segmenter = ROIGuidedSegmenter(DEFAULT_DET_WEIGHTS, DEFAULT_SEG_WEIGHTS, config=cfg)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    segmenter.warmup(height, width)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    latencies = []
    tot_props = tot_acc = tot_fb = 0
    rej_reasons = Counter()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            res = segmenter.process_frame(frame)
            s = res["summary"]
            t = res["timing_ms"]
            tot_props += s["num_proposals"]
            tot_acc += s["accepted_masks"]
            tot_fb += s["bbox_fallbacks"]
            latencies.append(t["combined_total_ms"])

            vis = frame.copy()
            if res["union_accepted_mask"].any():
                tint = vis.copy()
                tint[res["union_accepted_mask"]] = (120, 200, 30)  # BGR emerald
                vis = cv2.addWeighted(vis, 0.62, tint, 0.38, 0)
                cnts, _ = cv2.findContours(res["union_accepted_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (140, 255, 80), 2)

            for p in res["proposals"]:
                x1, y1, x2, y2 = [int(round(v)) for v in p["det_box_xyxy"]]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
                if p["accepted"]:
                    col = (80, 230, 80)  # BGR green
                    lbl = f"ROI MASK d:{p['det_conf']:.2f} s:{p['seg_conf']:.2f}"
                else:
                    col = (0, 180, 255)  # BGR amber
                    rtag = p["rejection_reasons"][0].replace("_mask_prediction", "").replace("_", "-")[:16]
                    lbl = f"BBOX FB d:{p['det_conf']:.2f} ({rtag})"
                    for r in p["rejection_reasons"]:
                        rej_reasons[r] += 1

                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                ly = max(th + 4, y1)
                cv2.rectangle(vis, (x1, ly - th - 4), (min(width - 1, x1 + tw + 4), ly + 2), (20, 20, 25), -1)
                cv2.putText(vis, lbl, (x1 + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

            hud = (
                f"ROI-Seg+Fallback | f={frame_idx} | Props:{s['num_proposals']} "
                f"Mask:{s['accepted_masks']} FB:{s['bbox_fallbacks']} | "
                f"Det:{t['detector_ms']:.1f}ms ROI:{t['seg_roi_ms']:.1f}ms Tot:{t['combined_total_ms']:.1f}ms"
            )
            cv2.rectangle(vis, (4, 4), (min(width - 4, 634), 26), (12, 12, 16), -1)
            cv2.putText(vis, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (240, 240, 240), 1, cv2.LINE_AA)

            writer.write(vis)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    arr = np.array(latencies, dtype=float)
    return {
        "input_video": os.path.abspath(video_path),
        "output_video": os.path.abspath(output_path),
        "batch_roi": batch_roi,
        "preserved_dimensions": f"{width}x{height}",
        "preserved_fps": round(float(fps), 3),
        "source_total_frames": total_src,
        "frames_processed": frame_idx,
        "raw_proposals_sum": tot_props,
        "accepted_roi_masks_sum": tot_acc,
        "bbox_fallbacks_sum": tot_fb,
        "acceptance_rate_pct": round(100.0 * tot_acc / max(1, tot_props), 2),
        "rejection_reason_counts": dict(rej_reasons),
        "mean_latency_ms": round(float(arr.mean()), 2) if len(arr) else 0.0,
        "median_latency_ms": round(float(np.median(arr)), 2) if len(arr) else 0.0,
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 2) if len(arr) else 0.0,
        "mean_fps": round(1000.0 / float(arr.mean()), 2) if len(arr) and arr.mean() > 0 else 0.0,
        "note": "Raw per-frame proposal/mask sums are NOT unique physical pothole counts (no tracking enabled).",
    }


def main():
    parser = argparse.ArgumentParser(description="Research-Only ROI-Guided Segmentation Video Demo")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-roi", action="store_true")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--seg-conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    args = parser.parse_args()

    summary = render_roi_video(
        video_path=args.video,
        output_path=args.output,
        batch_roi=args.batch_roi,
        det_conf=args.det_conf,
        seg_conf=args.seg_conf,
        max_frames=args.max_frames,
    )
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
