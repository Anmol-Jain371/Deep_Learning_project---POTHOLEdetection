"""
========================================================================================
🚀 RoadVision AI — Advanced Pothole Model Trainer & Fine-Tuning Pipeline
========================================================================================
Usage:
    # 1. Fine-tune existing pothole_best.pt on new dataset:
    python train_advanced_pothole_model.py --data path/to/data.yaml --epochs 25

    # 2. Train an upgraded YOLOv8 Medium (YOLOv8m) backbone from scratch:
    python train_advanced_pothole_model.py --data path/to/data.yaml --model yolov8m.pt --epochs 40
========================================================================================
"""

import os
import argparse
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train / Fine-tune RoadVision AI Pothole Detector")
    parser.add_argument("--data", type=str, default="dataset_combined_v2/data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="pothole_best.pt", help="Base model weights (pothole_best.pt)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input resolution (1280 preserves fine road cracks and small cavities)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (use 4-8 for 1280p on 8-16GB VRAM)")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="CUDA device index or 'cpu'")
    parser.add_argument("--name", type=str, default="pothole_combined_v2", help="Run name (output saved to runs/detect/<name>)")
    args = parser.parse_args()

    print("=" * 70)
    print("ROADVISION AI: ADVANCED POTHOLE MODEL RETRAINING PIPELINE")
    print(f"Base Weights : {args.model}")
    print(f"Dataset Config: {args.data}")
    print(f"Target Res   : {args.imgsz}x{args.imgsz} (High-Definition Multi-Scale)")
    print(f"Epochs       : {args.epochs}")
    print(f"Device       : {args.device}")
    print("=" * 70)

    if not os.path.exists(args.data):
        print(f"\n[!] Dataset configuration file not found: {args.data}")
        print("To train on a Kaggle or Roboflow dataset:")
        print("  1. Download the dataset formatted as 'YOLOv8'")
        print("  2. Point --data to the unzipped data.yaml file.")
        return

    # Load base model
    model = YOLO(args.model)

    # Launch multi-scale, heavily augmented training for domain invariance
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        optimizer="AdamW",   # Adaptive learning with decoupled weight decay
        lr0=0.002,           # Gentle fine-tuning learning rate (prevents weight shock)
        lrf=0.01,            # Final learning rate fraction (0.002 * 0.01)
        close_mosaic=10,     # Disables mosaic during the final 10 epochs
        # Heavy data augmentations to prevent overfitting and master all road types:
        scale=0.5,           # Multi-scale feature learning (0.5x to 1.5x)
        mosaic=1.0,          # 4-image mosaic stitching for small object recognition
        mixup=0.15,          # Blends images to recognize shadowed and occluded potholes
        perspective=0.0005,  # Simulates varied road inclinations and camera tilts
        hsv_h=0.015,         # Hue variations
        hsv_s=0.7,           # Saturation variations (wet vs dry asphalt)
        hsv_v=0.4,           # Value/brightness jitter (harsh sunlight vs deep shadows)
        fliplr=0.5,          # Horizontal flip
        patience=10,         # Early stopping if mAP plateaus
        save=True
    )

    print("\n[OK] Training Complete!")
    best_weight = os.path.join("runs", "detect", args.name, "weights", "best.pt")
    if os.path.exists(best_weight):
        print(f"Best Upgraded Weights Saved to: {best_weight}")
        print("Note: pothole_best.pt remains untouched for baseline comparison.")

if __name__ == "__main__":
    main()
