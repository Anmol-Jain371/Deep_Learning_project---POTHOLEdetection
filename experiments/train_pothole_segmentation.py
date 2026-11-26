"""
========================================================================================
RoadVision AI -- Phase 4A.2: Reproducible YOLOv8n-seg Training Pipeline
========================================================================================
Trains a segmentation baseline from standard COCO-pretrained `yolov8n-seg.pt`
exclusively on `dataset_pothole440_seg_grouped/data.yaml` (train split for gradient
updates; val split ONLY for checkpoint selection & early stopping; test split NEVER
accessed).

Features:
  - Full CLI configurability (model, data, epochs, imgsz, batch, optimizer, lr0,
    cos_lr, patience, device, seed, workers, project/name, resume).
  - Conservative online augmentations tailored for offline-augmented Pothole440.
  - Pre-training GPU VRAM probe starting at batch=4 (and optional auto-increase to 8
    only when reserved VRAM stays comfortably < 70% of GPU capacity).
  - Safe CUDA OutOfMemoryError recovery stepping down batch size.
  - Telemetry JSON logging (duration, peak VRAM, best epoch, losses, box/mask metrics).
========================================================================================
"""

import os
import sys
import csv
import json
import time
import shutil
import hashlib
import argparse
from typing import Dict, Any, Optional

import torch
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "data.yaml")
DEFAULT_PROJECT = os.path.join(BASE_DIR, "runs", "segment")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_vram(model_path: str, data_yaml: str, imgsz: int, batch_size: int, optimizer: str, lr0: float, workers: int, seed: int, device: str) -> Dict[str, Any]:
    """Runs a brief 1-epoch mini-fraction probe to measure peak CUDA VRAM safely."""
    if not torch.cuda.is_available() or str(device) == "cpu":
        return {"batch_size": batch_size, "ok": True, "device": "cpu", "peak_alloc_mb": 0.0, "peak_reserved_mb": 0.0}

    dev_idx = int(device) if str(device).isdigit() else 0
    torch.cuda.set_device(dev_idx)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev_idx)
    total_mb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 2)

    probe_proj = os.path.join(BASE_DIR, "scratch_seg_probe")
    probe_name = f"bs{batch_size}"
    try:
        m = YOLO(model_path)
        m.train(
            data=data_yaml,
            epochs=1,
            fraction=0.10,
            imgsz=imgsz,
            batch=batch_size,
            optimizer=optimizer,
            lr0=lr0,
            cos_lr=True,
            amp=True,
            workers=workers,
            seed=seed,
            device=dev_idx,
            project=probe_proj,
            name=probe_name,
            exist_ok=True,
            plots=False,
            val=False,
            verbose=False,
        )
        alloc_mb = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2)
        res_mb = torch.cuda.max_memory_reserved(dev_idx) / (1024 ** 2)
        del m
        torch.cuda.empty_cache()
        return {
            "batch_size": batch_size,
            "ok": True,
            "peak_alloc_mb": round(alloc_mb, 2),
            "peak_reserved_mb": round(res_mb, 2),
            "total_vram_mb": round(total_mb, 2),
            "reserved_pct": round(res_mb / total_mb * 100.0, 2),
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "ok": False, "error": str(e)}
    finally:
        if os.path.exists(probe_proj):
            shutil.rmtree(probe_proj, ignore_errors=True)


def parse_results_csv(csv_path: str) -> Dict[str, Any]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Ultralytics default segmentation fitness:
    # 0.1 * (mAP50_B + mAP50_M) + 0.9 * (mAP50-95_B + mAP50-95_M)
    def seg_fitness(r):
        return (
            0.1 * (float(r["metrics/mAP50(B)"]) + float(r["metrics/mAP50(M)"]))
            + 0.9 * (float(r["metrics/mAP50-95(B)"]) + float(r["metrics/mAP50-95(M)"]))
        )

    best_row = max(rows, key=seg_fitness)
    best_mask50_row = max(rows, key=lambda r: float(r["metrics/mAP50(M)"]))
    last_row = rows[-1]

    def extract_row(r):
        return {
            "epoch": int(float(r["epoch"])),
            "train_box_loss": round(float(r["train/box_loss"]), 4),
            "train_seg_loss": round(float(r["train/seg_loss"]), 4),
            "train_cls_loss": round(float(r["train/cls_loss"]), 4),
            "train_dfl_loss": round(float(r["train/dfl_loss"]), 4),
            "val_box_loss": round(float(r["val/box_loss"]), 4),
            "val_seg_loss": round(float(r["val/seg_loss"]), 4),
            "val_cls_loss": round(float(r["val/cls_loss"]), 4),
            "val_dfl_loss": round(float(r["val/dfl_loss"]), 4),
            "box_precision": round(float(r["metrics/precision(B)"]), 4),
            "box_recall": round(float(r["metrics/recall(B)"]), 4),
            "box_mAP50": round(float(r["metrics/mAP50(B)"]), 4),
            "box_mAP50_95": round(float(r["metrics/mAP50-95(B)"]), 4),
            "mask_precision": round(float(r["metrics/precision(M)"]), 4),
            "mask_recall": round(float(r["metrics/recall(M)"]), 4),
            "mask_mAP50": round(float(r["metrics/mAP50(M)"]), 4),
            "mask_mAP50_95": round(float(r["metrics/mAP50-95(M)"]), 4),
            "lr": float(r["lr/pg0"]),
        }

    return {
        "epochs_completed": len(rows),
        "selection_criterion": "Ultralytics validation fitness: 0.1*(mAP50_B + mAP50_M) + 0.9*(mAP50-95_B + mAP50-95_M)",
        "best_epoch_by_fitness": int(float(best_row["epoch"])),
        "best_epoch_by_mask_mAP50": int(float(best_mask50_row["epoch"])),
        "best_epoch_metrics": extract_row(best_row),
        "final_epoch_metrics": extract_row(last_row),
    }


def main():
    parser = argparse.ArgumentParser(description="RoadVision Phase 4A.2: YOLOv8n-seg Training")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--auto-batch", action="store_true", help="Allow stepping up batch 4->8 if probe VRAM < 70%")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--cos-lr", dest="cos_lr", action="store_true", default=True)
    parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default="pothole440_yolov8n_seg_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    model_path = args.model if os.path.isabs(args.model) else os.path.join(BASE_DIR, args.model)
    run_dir = os.path.join(args.project, args.name)

    if os.path.exists(run_dir) and not args.resume:
        raise FileExistsError(f"Run directory already exists (preserving existing experiments): {run_dir}")

    # Conservative online augmentations for dataset already containing 234 offline-augmented frames
    aug_config = {
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "cutmix": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "translate": 0.05,
        "scale": 0.15,
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.25,
        "erasing": 0.0,
    }

    # 1. Pre-training GPU memory probe
    probe_results = []
    chosen_batch = args.batch
    if not args.skip_probe and not args.resume and torch.cuda.is_available() and str(args.device) != "cpu":
        print(f"[PROBE] Probing GPU VRAM at initial batch={args.batch}...")
        p4 = probe_vram(model_path, args.data, args.imgsz, args.batch, args.optimizer, args.lr0, args.workers, args.seed, args.device)
        probe_results.append(p4)
        print(f"[PROBE] batch={args.batch} -> {p4}")
        if args.auto_batch and p4.get("ok") and p4.get("reserved_pct", 100.0) < 45.0:
            next_bs = args.batch * 2
            print(f"[PROBE] Headroom available ({p4['reserved_pct']}% used). Probing batch={next_bs}...")
            p8 = probe_vram(model_path, args.data, args.imgsz, next_bs, args.optimizer, args.lr0, args.workers, args.seed, args.device)
            probe_results.append(p8)
            print(f"[PROBE] batch={next_bs} -> {p8}")
            if p8.get("ok") and p8.get("reserved_pct", 100.0) < 70.0:
                chosen_batch = next_bs
                print(f"[PROBE] Safely increasing batch size to {chosen_batch} ({p8['reserved_pct']}% VRAM).")
            else:
                print(f"[PROBE] Keeping conservative batch={chosen_batch}.")

    # 2. Launch training with safe OOM fallback
    batch_candidates = [b for b in [chosen_batch, 4, 2] if b <= chosen_batch]
    seen = set()
    batch_candidates = [b for b in batch_candidates if not (b in seen or seen.add(b))]

    dev_arg = int(args.device) if str(args.device).isdigit() else args.device
    used_batch = None
    peak_alloc_mb = 0.0
    peak_reserved_mb = 0.0
    t_start = time.time()

    for bs in batch_candidates:
        try:
            if torch.cuda.is_available() and str(args.device) != "cpu":
                torch.cuda.set_device(dev_arg)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev_arg)

            model = YOLO(model_path)
            t_start = time.time()
            model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=bs,
                optimizer=args.optimizer,
                lr0=args.lr0,
                cos_lr=args.cos_lr,
                patience=args.patience,
                close_mosaic=args.close_mosaic,
                amp=True,
                workers=args.workers,
                seed=args.seed,
                device=dev_arg,
                project=args.project,
                name=args.name,
                exist_ok=args.resume,
                resume=args.resume,
                save=True,
                plots=True,
                val=True,
                **aug_config,
            )
            used_batch = bs
            if torch.cuda.is_available() and str(args.device) != "cpu":
                peak_alloc_mb = torch.cuda.max_memory_allocated(dev_arg) / (1024 ** 2)
                peak_reserved_mb = torch.cuda.max_memory_reserved(dev_arg) / (1024 ** 2)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError as oom_err:
            print(f"[OOM] CUDA out-of-memory at batch={bs}: {oom_err}. Cleaning up and retrying lower batch...")
            if not args.resume and os.path.exists(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if bs == batch_candidates[-1]:
                raise RuntimeError("CUDA OOM even at minimum fallback batch size.") from oom_err

    duration_sec = time.time() - t_start
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    curves = parse_results_csv(os.path.join(run_dir, "results.csv"))

    telemetry = {
        "run_name": args.name,
        "run_dir": run_dir,
        "base_weights": args.model,
        "base_weights_sha256": sha256_file(model_path),
        "best_weights_path": best_pt,
        "best_weights_sha256": sha256_file(best_pt) if os.path.exists(best_pt) else None,
        "best_weights_size_mb": round(os.path.getsize(best_pt) / (1024 ** 2), 2) if os.path.exists(best_pt) else None,
        "data_yaml": args.data,
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "imgsz": args.imgsz,
            "initial_batch_arg": args.batch,
            "used_batch_size": used_batch,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "cos_lr": args.cos_lr,
            "patience": args.patience,
            "close_mosaic": args.close_mosaic,
            "amp": True,
            "workers": args.workers,
            "seed": args.seed,
            "device": str(args.device),
        },
        "conservative_augmentations": aug_config,
        "vram_probes": probe_results,
        "training_duration_sec": round(duration_sec, 2),
        "training_duration_min": round(duration_sec / 60.0, 2),
        "peak_gpu_allocated_mb": round(peak_alloc_mb, 2),
        "peak_gpu_reserved_mb": round(peak_reserved_mb, 2),
        "total_gpu_vram_mb": round(torch.cuda.get_device_properties(dev_arg).total_memory / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0,
        "curves_summary": curves,
    }

    tele_path = os.path.join(run_dir, "phase4a2_train_telemetry.json")
    with open(tele_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f"[TELEMETRY] Saved {tele_path}")
    print(json.dumps(curves, indent=2))


if __name__ == "__main__":
    main()
