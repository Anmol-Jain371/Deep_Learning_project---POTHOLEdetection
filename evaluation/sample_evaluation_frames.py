"""
========================================================================================
RoadVision AI — Evaluation Frame Sampling & Annotation Manifest Generator
========================================================================================
Purpose:
  Extracts a representative, stratified sample of 40 evaluation frames from the
  held-out Rangoon Road dashcam video ('Watch Video_ Terrible potholes in Rangoon Road.mp4').
  Generates an evaluation manifest CSV and empty YOLO annotation template files.

Scientific Integrity Rules:
  1. Frames are strictly for evaluation benchmarking — NEVER added to training data.
  2. Bounding boxes are NEVER automatically synthesized from model predictions.
  3. Ground truth status remains explicitly 'PENDING' until manually labeled.
========================================================================================
"""

import os
import sys
import cv2
import csv

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(base_dir)
    video_path = os.path.join(project_root, "Watch Video_ Terrible potholes in Rangoon Road.mp4")

    if not os.path.exists(video_path):
        print(f"[ERROR] Evaluation video not found at: {video_path}")
        return

    frames_dir = os.path.join(base_dir, "frames")
    labels_dir = os.path.join(base_dir, "ground_truth_labels")
    manifest_path = os.path.join(base_dir, "evaluation_manifest.csv")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample 40 frames uniformly across the video (step = 25 frames ~ 1.0 second intervals)
    sample_indices = list(range(0, min(total_frames, 1000), 25))

    manifest_rows = []
    print(f"[INFO] Sampling {len(sample_indices)} evaluation frames from {os.path.basename(video_path)} ({total_frames} total frames)...")

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp_sec = round(idx / fps, 2)
        frame_name = f"eval_frame_{idx:04d}"
        img_filename = f"{frame_name}.jpg"
        img_path = os.path.join(frames_dir, img_filename)
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Create corresponding empty YOLO ground-truth annotation file if it doesn't already exist
        txt_path = os.path.join(labels_dir, f"{frame_name}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as tf:
                # Intentionally empty: human annotator adds "<class_id> <x_center> <y_center> <width> <height>"
                pass

        # Categorize frame context
        if idx < 100:
            notes = "Approach section - tree shadows & clean road surface"
        elif 100 <= idx < 400:
            notes = "Mid-distance potholes & heavy asphalt wear"
        elif 400 <= idx < 750:
            notes = "Severe cluster - deep craters & water puddles"
        else:
            notes = "Exit section - varied road patchiness & roadside gravel"

        manifest_rows.append({
            "frame_id": frame_name,
            "source_video": "Watch Video_ Terrible potholes in Rangoon Road.mp4",
            "frame_index": idx,
            "timestamp_sec": timestamp_sec,
            "image_width": w,
            "image_height": h,
            "annotation_status": "PENDING_MANUAL_LABEL",
            "num_ground_truth_potholes": "",
            "difficulty_lighting_notes": notes
        })

    cap.release()

    # Write manifest CSV
    fieldnames = [
        "frame_id", "source_video", "frame_index", "timestamp_sec",
        "image_width", "image_height", "annotation_status",
        "num_ground_truth_potholes", "difficulty_lighting_notes"
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[OK] Successfully extracted {len(manifest_rows)} evaluation frames to: {frames_dir}")
    print(f"[OK] Created ground-truth label templates in: {labels_dir}")
    print(f"[OK] Generated evaluation manifest at: {manifest_path}")

if __name__ == "__main__":
    main()
