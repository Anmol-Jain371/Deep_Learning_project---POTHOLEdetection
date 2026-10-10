"""
========================================================================================
RoadVision AI — Scientific Detection Benchmark & Evaluation Engine
========================================================================================
Evaluates the locked production detector (Model #2: pothole_combined_v2) on the
held-out Rangoon Road benchmark dataset.

Scientific Integrity Rules:
  1. Never fabricates ground truth labels or accuracy metrics.
  2. If ground-truth annotations are missing or empty, halts with ANNOTATION PENDING.
  3. Clearly distinguishes detector object-level metrics (Precision, Recall, F1, AP50)
     from tracker continuity metrics (ID count, blips, flicker rate).
========================================================================================
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def calculate_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes in format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def load_yolo_annotations(txt_path, img_w, img_h):
    """
    Loads YOLO format annotations (class_id, x_center, y_center, width, height)
    normalized, and returns list of [x1, y1, x2, y2] pixel coordinates.
    """
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * img_w
                    yc = float(parts[2]) * img_h
                    bw = float(parts[3]) * img_w
                    bh = float(parts[4]) * img_h
                    x1 = max(0, xc - bw / 2.0)
                    y1 = max(0, yc - bh / 2.0)
                    x2 = min(img_w, xc + bw / 2.0)
                    y2 = min(img_h, yc + bh / 2.0)
                    boxes.append([x1, y1, x2, y2])
                except ValueError:
                    continue
    return boxes

def main():
    parser = argparse.ArgumentParser(description="RoadVision Detection Benchmark Evaluator")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.14, help="Inference confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.50, help="IoU matching threshold for AP50")
    args = parser.parse_args()

    eval_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(eval_dir)

    # Resolve model weights
    default_prod = os.path.join(project_root, "runs", "detect", "pothole_combined_v2", "weights", "best.pt")
    weights_path = args.weights or default_prod
    if not os.path.exists(weights_path):
        weights_path = os.path.join(project_root, "pothole_best.pt")

    frames_dir = os.path.join(eval_dir, "frames")
    labels_dir = os.path.join(eval_dir, "ground_truth_labels")
    manifest_path = os.path.join(eval_dir, "evaluation_manifest.csv")

    print("\n" + "=" * 80)
    print("ROADVISION AI -- HELD-OUT DETECTION EVALUATION ENGINE")
    print("=" * 80)
    print(f"Model Checkpoint : {weights_path}")
    print(f"Confidence Thresh: {args.conf}")
    print(f"IoU Match Thresh : {args.iou_thresh}")
    print(f"Evaluation Frames: {frames_dir}")
    print(f"Ground Truth Dir : {labels_dir}")

    # Check available frames
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        print("[ERROR] No sampled evaluation frames found. Run 'sample_evaluation_frames.py' first.")
        return

    # Check ground-truth annotation status
    total_frames = len(frame_files)
    annotated_frames = 0
    total_gt_boxes = 0
    ground_truth_map = {}

    for fpath in frame_files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        lpath = os.path.join(labels_dir, f"{fname}.txt")
        if os.path.exists(lpath) and os.path.getsize(lpath) > 0:
            boxes = load_yolo_annotations(lpath, 640, 360)
            if len(boxes) > 0:
                annotated_frames += 1
                total_gt_boxes += len(boxes)
                ground_truth_map[fname] = boxes

    print("\n[GROUND TRUTH INVENTORY]")
    print(f"  * Sampled Evaluation Frames        : {total_frames}")
    print(f"  * Manually Annotated Frames        : {annotated_frames} / {total_frames}")
    print(f"  * Total Ground-Truth Pothole Boxes : {total_gt_boxes}")

    if annotated_frames == 0:
        print("\n" + "=" * 80)
        print("[STATUS] SCIENTIFIC EVALUATION: ANNOTATION PENDING")
        print("=" * 80)
        print("SCIENTIFIC INTEGRITY POLICY:")
        print("  Ground-truth bounding boxes must be manually drawn by human inspectors.")
        print("  Model predictions are NEVER automatically treated as ground truth.")
        print("  No synthetic or fabricated precision/recall metrics will be generated.\n")
        print("HOW TO PROVIDE GROUND TRUTH ANNOTATIONS:")
        print(f"  1. Open frames in '{frames_dir}' using any standard labeling tool (CVAT, LabelImg, etc.).")
        print(f"  2. Save YOLO annotations to '{labels_dir}/<frame_id>.txt'.")
        print("     Format: <class_id> <x_center> <y_center> <width> <height> (normalized 0.0-1.0)")
        print(f"  3. Re-run 'python evaluation/evaluate_detection_benchmark.py' to produce certified metrics.")
        print("=" * 80 + "\n")
        return

    # If annotations exist, run genuine evaluation
    print(f"\n[RUNNING FROZEN INFERENCE ON {annotated_frames} ANNOTATED FRAMES]...")
    model = YOLO(weights_path)

    tp_count = 0
    fp_count = 0
    fn_count = 0

    for fname, gt_boxes in ground_truth_map.items():
        fpath = os.path.join(frames_dir, f"{fname}.jpg")
        img = cv2.imread(fpath)
        h, w = img.shape[:2]

        results = model.predict(source=img, conf=args.conf, verbose=False)
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()

        matched_gt = set()
        matched_pred = set()

        # Greedy bipartite matching
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= args.iou_thresh and best_g_idx >= 0:
                tp_count += 1
                matched_gt.add(best_g_idx)
                matched_pred.add(p_idx)
            else:
                fp_count += 1

        fn_count += len(gt_boxes) - len(matched_gt)

    precision = tp_count / max(1, tp_count + fp_count)
    recall = tp_count / max(1, tp_count + fn_count)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)

    results_data = {
        "status": "VALIDATED",
        "model_weights": weights_path,
        "confidence_threshold": args.conf,
        "iou_matching_threshold": args.iou_thresh,
        "annotated_frames_evaluated": annotated_frames,
        "ground_truth_instances": total_gt_boxes,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

    out_json = os.path.join(eval_dir, "benchmark_eval_results.json")
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(results_data, jf, indent=2)

    print("\n" + "=" * 80)
    print("DETECTION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  * True Positives (TP)  : {tp_count}")
    print(f"  * False Positives (FP) : {fp_count}")
    print(f"  * False Negatives (FN) : {fn_count}")
    print(f"  * Precision            : {precision * 100:.2f}%")
    print(f"  * Recall               : {recall * 100:.2f}%")
    print(f"  * F1-Score             : {f1 * 100:.2f}%")
    print(f"  * Results saved to     : {out_json}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
