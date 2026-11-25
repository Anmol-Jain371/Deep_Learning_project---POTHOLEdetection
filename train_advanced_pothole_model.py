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
    parser.add_argument("--name", type=str, default="pothole_combined_v2_50e", help="Run name (output saved to runs/detect/<name>)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project output folder")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine learning rate scheduler for smooth convergence")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 70)
    print("ROADVISION AI: ADVANCED POTHOLE MODEL RETRAINING PIPELINE")
    print(f"Base Weights : {args.model}")
    print(f"Dataset Config: {args.data}")
    print(f"Target Res   : {args.imgsz}x{args.imgsz} (High-Definition Multi-Scale)")
    print(f"Epochs       : {args.epochs} (cos_lr={args.cos_lr}, patience={args.patience})")
    print(f"Batch / LR0  : {args.batch} / {args.lr0}")
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
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        amp=True,
        cos_lr=args.cos_lr,  # Smooth cosine decay toward minimum LR
        optimizer="AdamW",   # Adaptive learning with decoupled weight decay
        lr0=args.lr0,        # Gentle fine-tuning learning rate (prevents weight shock)
        lrf=args.lrf,        # Final learning rate fraction
        box=8.5,             # Enhanced bounding-box regression loss for irregular cavity boundaries
        dfl=1.8,             # Distribution Focal Loss weight for crisp border alignment
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
        patience=args.patience, # Early stopping if mAP plateaus
        save=True
    )

    print("\n[OK] Training Complete!")
    best_weight = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_weight):
        print(f"Best Upgraded Weights Saved to: {best_weight}")
        print("Note: pothole_best.pt remains untouched for baseline comparison.")

        # Run strict evaluation on held-out test split if defined in data.yaml
        try:
            import yaml
            with open(args.data, "r", encoding="utf-8") as df:
                dcfg = yaml.safe_load(df)
            if isinstance(dcfg, dict) and "test" in dcfg:
                print("\n[EVAL] Evaluating best.pt on strictly held-out TEST split...")
                best_model = YOLO(best_weight)
                test_metrics = best_model.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device)
                print(
                    f"[TEST] mAP50={test_metrics.box.map50:.4f}, "
                    f"mAP50-95={test_metrics.box.map:.4f}, "
                    f"Precision={test_metrics.box.mp:.4f}, "
                    f"Recall={test_metrics.box.mr:.4f}"
                )
        except Exception as e:
            print(f"[!] Post-training test split evaluation skipped: {e}")

if __name__ == "__main__":
    main()
