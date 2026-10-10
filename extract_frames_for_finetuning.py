"""
========================================================================================
📸 Frame Extractor for Local Domain Fine-Tuning
========================================================================================
Usage:
    python extract_frames_for_finetuning.py --video my_walking_survey.mp4 --out dataset_local --fps 2

Extracts 1-2 clean frames per second from your phone video into dataset_local/images
ready to be uploaded to Roboflow or annotated for YOLOv8 fine-tuning.
========================================================================================
"""

import os
import argparse
import cv2

def main():
    parser = argparse.ArgumentParser(description="Extract frames from phone road video for fine-tuning")
    parser.add_argument("--video", type=str, default="my_walking_survey.mp4", help="Path to input phone video")
    parser.add_argument("--out", type=str, default="local_dataset", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames to extract per second of video")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[!] Video file not found: {args.video}")
        print("Please record a 2-3 minute road video on your phone and place it in this folder.")
        return

    images_dir = os.path.join(args.out, "images")
    os.makedirs(images_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(video_fps / args.fps))

    print(f"Opening {args.video} ({total_frames} frames @ {video_fps:.1f} FPS)...")
    print(f"Extracting every {step} frames (~{args.fps} frames/sec)...")

    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            out_name = f"frame_{saved_count:04d}.jpg"
            out_path = os.path.join(images_dir, out_name)
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"\n[OK] Extraction Complete! {saved_count} frames saved to: {images_dir}")
    print("Next step: Annotate potholes on these images (Roboflow or CVAT) and export in YOLOv8 format.")

if __name__ == "__main__":
    main()
