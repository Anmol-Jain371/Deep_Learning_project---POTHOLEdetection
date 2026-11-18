"""
========================================================================================
🚗 RoadVision AI — Faster R-CNN Training Pipeline
========================================================================================
Trains Faster R-CNN ResNet-50-FPN on RDD2022 India D40 Pothole dataset.
Features:
  - Automatic train/val split with reproducibility seed
  - Mixed precision training (torch.cuda.amp) if CUDA is available
  - Learning rate warmup & Cosine Annealing scheduler
  - Periodic checkpoint saving (best loss & latest)
  - Validation metrics logging
========================================================================================
"""

import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from rdd_faster_rcnn_dataset import RDDPotholeDataset, collate_fn
from faster_rcnn_model import create_faster_rcnn_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on Pothole Dataset")
    parser.add_argument("--data-dir", type=str, default="RDD2022/India/train", help="Path to India/train dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay for optimizer")
    parser.add_argument("--output-dir", type=str, default="runs/faster_rcnn", help="Output directory for weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)
    t0 = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        total_loss += losses.item()

        if (i + 1) % 20 == 0 or (i + 1) == num_batches:
            elapsed = time.time() - t0
            loss_components = " ".join([f"{k[:4]}:{v.item():.3f}" for k, v in loss_dict.items()])
            print(f"  Epoch [{epoch+1}] Step [{i+1:03d}/{num_batches:03d}] | Loss: {losses.item():.4f} ({loss_components}) | Time: {elapsed:.1f}s")

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def validate(model, data_loader, device):
    """
    Computes validation loss in train mode without gradient computation.
    """
    model.train()
    total_val_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()

    return total_val_loss / max(1, num_batches)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 RoadVision AI — Training Faster R-CNN ResNet-50-FPN on [{str(device).upper()}]")
    print(f"📦 Dataset path: {args.data_dir}")
    print(f"📊 Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")

    # 1. Dataset & Train/Val Split
    full_dataset = RDDPotholeDataset(root_dir=args.data_dir, filter_empty=True)
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"📈 Split: {train_size} training images | {val_size} validation images")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 2. Model Creation
    print("⏳ Initializing Faster R-CNN (ResNet-50-FPN-v2 backbone)...")
    model = create_faster_rcnn_model(num_classes=2, pretrained=True)
    model.to(device)
    print("✅ Model initialized successfully.")

    # 3. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "faster_rcnn_pothole_best.pth")
    last_ckpt_path = os.path.join(args.output_dir, "faster_rcnn_pothole_last.pth")

    print("\n🏁 Starting Training Loop...")
    for epoch in range(args.epochs):
        t_start = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        val_loss = validate(model, val_loader, device)
        lr_scheduler.step()

        epoch_time = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"✨ Epoch [{epoch+1}/{args.epochs}] Finished in {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Save latest
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        }, last_ckpt_path)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, best_ckpt_path)
            print(f"  🏆 New Best Model Saved (Val Loss: {best_val_loss:.4f}) -> {best_ckpt_path}")

    print("\n=======================================================")
    print("🎉 Faster R-CNN Training Completed!")
    print(f"🏆 Best Checkpoint: {best_ckpt_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

