"""
========================================================================================
RoadVision AI -- Phase 2 Controlled Architecture Comparison: YOLOv8n vs YOLOv8s
========================================================================================
Controlled experiment comparing fresh COCO-pretrained YOLOv8s and YOLOv8n under
identical dataset splits (dataset_unified v1), optimizer (SGD), LR schedule (cos_lr),
batch size, image resolution, seed, and augmentation settings.

Preserves all existing models, runs, datasets, and production configuration.
========================================================================================
"""

import os
import sys
import csv
import json
import time
import shutil
import argparse
import hashlib
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_YAML = os.path.join(BASE_DIR, "dataset_unified", "data.yaml")
EXT_YAML = os.path.join(BASE_DIR, "dataset_unified", "data_external_rangoon.yaml")
MANIFEST_PATH = os.path.join(BASE_DIR, "dataset_unified", "manifests", "frozen_manifest_v1.json")
PROJECT_DIR = os.path.join(BASE_DIR, "runs", "detect")
BENCHMARK_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
os.makedirs(BENCHMARK_DIR, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_controlled_training(
    base_weights: str,
    run_name: str,
    initial_batch: int = 8,
    epochs: int = 100,
    imgsz: int = 640,
    seed: int = 42
) -> Dict[str, Any]:
    run_dir = os.path.join(PROJECT_DIR, run_name)
    meta_path = os.path.join(run_dir, "phase2_train_telemetry.json")

    if os.path.exists(run_dir):
        raise FileExistsError(
            f"Run directory already exists (preserving existing runs): {run_dir}"
        )

    torch.cuda.set_device(0)
    batch_candidates = [b for b in [initial_batch, 4, 2] if b <= initial_batch]
    # Remove duplicates while preserving descending order
    seen = set()
    batch_candidates = [b for b in batch_candidates if not (b in seen or seen.add(b))]

    used_batch = None
    peak_alloc_mb = 0.0
    peak_reserved_mb = 0.0
    t_start = time.time()

    for bs in batch_candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            model = YOLO(os.path.join(BASE_DIR, base_weights))
            t_start = time.time()
            model.train(
                data=DATA_YAML,
                epochs=epochs,
                imgsz=imgsz,
                batch=bs,
                optimizer="SGD",
                lr0=0.01,
                cos_lr=True,
                close_mosaic=10,
                patience=30,
                amp=True,
                workers=2,
                seed=seed,
                device=0,
                project=PROJECT_DIR,
                name=run_name,
                exist_ok=False,
                save=True,
                plots=True,
                val=True,
            )
            used_batch = bs
            peak_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
            del model
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError as oom_err:
            print(f"[OOM] CUDA out-of-memory at batch={bs}: {oom_err}. Cleaning up and retrying lower batch...")
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
            torch.cuda.empty_cache()
            if bs == batch_candidates[-1]:
                raise RuntimeError("CUDA OOM even at minimum fallback batch size.") from oom_err

    duration_sec = time.time() - t_start
    telemetry = {
        "run_name": run_name,
        "base_weights": base_weights,
        "base_weights_sha256": sha256_file(os.path.join(BASE_DIR, base_weights)),
        "used_batch_size": used_batch,
        "training_duration_sec": round(duration_sec, 2),
        "training_duration_min": round(duration_sec / 60.0, 2),
        "peak_gpu_allocated_mb": round(peak_alloc_mb, 2),
        "peak_gpu_reserved_mb": round(peak_reserved_mb, 2),
        "total_gpu_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f"[TELEMETRY] Saved {meta_path}: {telemetry}")
    return telemetry


def parse_training_curves(run_name: str) -> Dict[str, Any]:
    run_dir = os.path.join(PROJECT_DIR, run_name)
    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Ultralytics selects best.pt using validation fitness: 0.1*mAP50 + 0.9*mAP50-95
    def fitness(r):
        return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])

    best_fit_row = max(rows, key=fitness)
    best_map50_row = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    last_row = rows[-1]

    return {
        "epochs_completed": len(rows),
        "best_epoch_by_fitness": int(float(best_fit_row["epoch"])),
        "best_epoch_by_map50": int(float(best_map50_row["epoch"])),
        "best_val_at_fitness_epoch": {
            "precision": round(float(best_fit_row["metrics/precision(B)"]), 4),
            "recall": round(float(best_fit_row["metrics/recall(B)"]), 4),
            "mAP50": round(float(best_fit_row["metrics/mAP50(B)"]), 4),
            "mAP50_95": round(float(best_fit_row["metrics/mAP50-95(B)"]), 4),
            "train_box_loss": round(float(best_fit_row["train/box_loss"]), 4),
            "train_cls_loss": round(float(best_fit_row["train/cls_loss"]), 4),
            "train_dfl_loss": round(float(best_fit_row["train/dfl_loss"]), 4),
            "val_box_loss": round(float(best_fit_row["val/box_loss"]), 4),
            "val_cls_loss": round(float(best_fit_row["val/cls_loss"]), 4),
            "val_dfl_loss": round(float(best_fit_row["val/dfl_loss"]), 4),
        },
        "final_epoch_metrics": {
            "epoch": int(float(last_row["epoch"])),
            "precision": round(float(last_row["metrics/precision(B)"]), 4),
            "recall": round(float(last_row["metrics/recall(B)"]), 4),
            "mAP50": round(float(last_row["metrics/mAP50(B)"]), 4),
            "mAP50_95": round(float(last_row["metrics/mAP50-95(B)"]), 4),
            "train_box_loss": round(float(last_row["train/box_loss"]), 4),
            "train_cls_loss": round(float(last_row["train/cls_loss"]), 4),
            "train_dfl_loss": round(float(last_row["train/dfl_loss"]), 4),
            "val_box_loss": round(float(last_row["val/box_loss"]), 4),
            "val_cls_loss": round(float(last_row["val/cls_loss"]), 4),
            "val_dfl_loss": round(float(last_row["val/dfl_loss"]), 4),
            "lr_final": float(last_row["lr/pg0"]),
        },
    }


def evaluate_checkpoint(run_name: str) -> Dict[str, Any]:
    run_dir = os.path.join(PROJECT_DIR, run_name)
    weights = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Missing best.pt for {run_name}: {weights}")

    model = YOLO(weights)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    m_obj = ckpt.get("model")
    raw_params = sum(p.numel() for p in m_obj.parameters()) if m_obj is not None else 0
    weights_mb = os.path.getsize(weights) / (1024 ** 2)

    eval_out_root = os.path.join(BENCHMARK_DIR, "phase2_evals")
    os.makedirs(eval_out_root, exist_ok=True)

    def run_val(yaml_path: str, split: str, augment: bool, tag: str) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        res = model.val(
            data=yaml_path,
            split=split,
            imgsz=640,
            batch=8,
            augment=augment,
            device=0,
            project=eval_out_root,
            name=f"{run_name}_{tag}",
            exist_ok=True,
            plots=True,
            verbose=False,
        )
        p = float(res.box.mp)
        r = float(res.box.mr)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        cm_mat = res.confusion_matrix.matrix.tolist() if getattr(res, "confusion_matrix", None) is not None else None
        speed = dict(res.speed) if getattr(res, "speed", None) else {}
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(float(res.box.map50), 4),
            "mAP50_95": round(float(res.box.map), 4),
            "speed_ms_per_img": {k: round(float(v), 3) for k, v in speed.items()},
            "total_latency_ms_per_img": round(sum(float(v) for v in speed.values()), 3) if speed else None,
            "confusion_matrix": cm_mat,
            "eval_plots_dir": os.path.join(eval_out_root, f"{run_name}_{tag}"),
        }

    # Measure raw GPU forward+NMS latency cleanly on all 153 test images with warmup
    test_img_dir = os.path.join(BASE_DIR, "dataset_unified", "images", "test")
    test_imgs = [os.path.join(test_img_dir, f) for f in sorted(os.listdir(test_img_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for w_img in test_imgs[:10]:
        model.predict(w_img, imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize()
    t0 = time.time()
    for p_img in test_imgs:
        model.predict(p_img, imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize()
    e2e_ms = (time.time() - t0) / len(test_imgs) * 1000.0

    # Measure TTA latency
    torch.cuda.synchronize()
    t1 = time.time()
    for p_img in test_imgs:
        model.predict(p_img, imgsz=640, augment=True, device=0, verbose=False)
    torch.cuda.synchronize()
    e2e_tta_ms = (time.time() - t1) / len(test_imgs) * 1000.0

    telemetry_path = os.path.join(run_dir, "phase2_train_telemetry.json")
    telemetry = json.load(open(telemetry_path, "r")) if os.path.exists(telemetry_path) else {}

    return {
        "run_name": run_name,
        "weights_path": weights,
        "weights_sha256": sha256_file(weights),
        "weights_size_mb": round(weights_mb, 2),
        "parameters": raw_params,
        "parameters_m": round(raw_params / 1e6, 3),
        "training_telemetry": telemetry,
        "training_curves": parse_training_curves(run_name),
        "predict_latency_test_153": {
            "no_tta_ms": round(e2e_ms, 2),
            "no_tta_fps": round(1000.0 / max(1e-6, e2e_ms), 2),
            "tta_ms": round(e2e_tta_ms, 2),
            "tta_fps": round(1000.0 / max(1e-6, e2e_tta_ms), 2),
        },
        "evaluations": {
            "val_no_tta": run_val(DATA_YAML, "val", False, "val_notta"),
            "val_tta": run_val(DATA_YAML, "val", True, "val_tta"),
            "test_exploratory_no_tta": run_val(DATA_YAML, "test", False, "test_notta"),
            "test_exploratory_tta": run_val(DATA_YAML, "test", True, "test_tta"),
            "external_rangoon_exploratory_no_tta": run_val(EXT_YAML, "val", False, "ext_notta"),
            "external_rangoon_exploratory_tta": run_val(EXT_YAML, "val", True, "ext_tta"),
        },
        "training_artifacts": {
            "results_csv": os.path.join(run_dir, "results.csv"),
            "results_png": os.path.join(run_dir, "results.png"),
            "confusion_matrix_png": os.path.join(run_dir, "confusion_matrix.png"),
            "pr_curve_png": os.path.join(run_dir, "BoxPR_curve.png"),
        },
    }


def write_comparison_reports(s_eval: Dict[str, Any], n_eval: Dict[str, Any]):
    manifest = json.load(open(MANIFEST_PATH, "r", encoding="utf-8"))
    report_data = {
        "experiment": "Phase 2 Controlled Architecture Comparison: YOLOv8n vs YOLOv8s",
        "dataset_manifest_version": manifest["manifest_version"],
        "dataset_manifest_sha256": sha256_file(MANIFEST_PATH),
        "data_yaml_sha256": sha256_file(DATA_YAML),
        "external_yaml_sha256": sha256_file(EXT_YAML),
        "shared_hyperparameters": {
            "epochs": 100,
            "imgsz": 640,
            "optimizer": "SGD",
            "lr0": 0.01,
            "cos_lr": True,
            "close_mosaic": 10,
            "patience": 30,
            "amp": True,
            "workers": 2,
            "seed": 42,
            "device": 0,
        },
        "models": {
            "yolov8n_sgd_100e": n_eval,
            "yolov8s_sgd_100e": s_eval,
        },
    }
    json_out = os.path.join(BENCHMARK_DIR, "phase2_yolov8n_vs_yolov8s_controlled.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    def fmt_row(label, n_val, s_val):
        return f"| {label} | {n_val} | {s_val} |"

    nc = n_eval["training_curves"]
    sc = s_eval["training_curves"]
    nt = n_eval["training_telemetry"]
    st = s_eval["training_telemetry"]

    md_lines = [
        "# Phase 2 Controlled Architecture Comparison: YOLOv8n vs. YOLOv8s",
        "",
        f"- **Dataset Manifest:** `{manifest['manifest_version']}` (`SHA256: {report_data['dataset_manifest_sha256']}`)",
        "- **Shared Settings:** `SGD`, `lr0=0.01`, `cos_lr=True`, `epochs=100`, `imgsz=640`, `close_mosaic=10`, `patience=30`, `amp=True`, `workers=2`, `seed=42`, `device=0`",
        "- **Note on Evaluation Splits:** `val` (153 imgs) was used strictly for checkpoint selection (`best.pt`). `test` (153 imgs) and `external_rangoon` (35 imgs) were previously inspected during Phase 1/exploration and are explicitly reported as **Exploratory Benchmark Splits**; an additional untouched final test set is required for final unbiased model certification.",
        "",
        "## 1. Model Capacity, Training Telemetry & Hardware Footprint",
        "",
        "| Metric | YOLOv8n (`yolov8n.pt` -> SGD 100e) | YOLOv8s (`yolov8s.pt` -> SGD 100e) |",
        "|---|:---:|:---:|",
        fmt_row("Parameters", f"{n_eval['parameters']:,} ({n_eval['parameters_m']}M)", f"{s_eval['parameters']:,} ({s_eval['parameters_m']}M)"),
        fmt_row("Checkpoint Size", f"{n_eval['weights_size_mb']} MB", f"{s_eval['weights_size_mb']} MB"),
        fmt_row("Batch Size Used", str(nt.get("used_batch_size")), str(st.get("used_batch_size"))),
        fmt_row("Epochs Completed", str(nc["epochs_completed"]), str(sc["epochs_completed"])),
        fmt_row("Best Epoch (Fitness)", str(nc["best_epoch_by_fitness"]), str(sc["best_epoch_by_fitness"])),
        fmt_row("Best Epoch (mAP@50)", str(nc["best_epoch_by_map50"]), str(sc["best_epoch_by_map50"])),
        fmt_row("Training Duration", f"{nt.get('training_duration_min')} min", f"{st.get('training_duration_min')} min"),
        fmt_row("Peak GPU Allocated", f"{nt.get('peak_gpu_allocated_mb')} MB", f"{st.get('peak_gpu_allocated_mb')} MB"),
        fmt_row("Peak GPU Reserved", f"{nt.get('peak_gpu_reserved_mb')} MB", f"{st.get('peak_gpu_reserved_mb')} MB"),
        fmt_row("Predict Latency (No TTA)", f"{n_eval['predict_latency_test_153']['no_tta_ms']} ms ({n_eval['predict_latency_test_153']['no_tta_fps']} FPS)", f"{s_eval['predict_latency_test_153']['no_tta_ms']} ms ({s_eval['predict_latency_test_153']['no_tta_fps']} FPS)"),
        fmt_row("Predict Latency (TTA)", f"{n_eval['predict_latency_test_153']['tta_ms']} ms ({n_eval['predict_latency_test_153']['tta_fps']} FPS)", f"{s_eval['predict_latency_test_153']['tta_ms']} ms ({s_eval['predict_latency_test_153']['tta_fps']} FPS)"),
        "",
        "## 2. Losses at Best Validation Epoch & Final Epoch",
        "",
        "| Loss Term | YOLOv8n (Best / Final) | YOLOv8s (Best / Final) |",
        "|---|:---:|:---:|",
        fmt_row("Train Box Loss", f"{nc['best_val_at_fitness_epoch']['train_box_loss']} / {nc['final_epoch_metrics']['train_box_loss']}", f"{sc['best_val_at_fitness_epoch']['train_box_loss']} / {sc['final_epoch_metrics']['train_box_loss']}"),
        fmt_row("Val Box Loss", f"{nc['best_val_at_fitness_epoch']['val_box_loss']} / {nc['final_epoch_metrics']['val_box_loss']}", f"{sc['best_val_at_fitness_epoch']['val_box_loss']} / {sc['final_epoch_metrics']['val_box_loss']}"),
        fmt_row("Train Cls Loss", f"{nc['best_val_at_fitness_epoch']['train_cls_loss']} / {nc['final_epoch_metrics']['train_cls_loss']}", f"{sc['best_val_at_fitness_epoch']['train_cls_loss']} / {sc['final_epoch_metrics']['train_cls_loss']}"),
        fmt_row("Val Cls Loss", f"{nc['best_val_at_fitness_epoch']['val_cls_loss']} / {nc['final_epoch_metrics']['val_cls_loss']}", f"{sc['best_val_at_fitness_epoch']['val_cls_loss']} / {sc['final_epoch_metrics']['val_cls_loss']}"),
        "",
        "## 3. Detection Accuracy Across Identical Splits (No TTA vs. TTA)",
        "",
        "| Split & Mode | Model | Precision | Recall | F1 | mAP@50 | mAP@50-95 |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for split_key, split_label in [
        ("val_no_tta", "Val (153 imgs) — No TTA"),
        ("val_tta", "Val (153 imgs) — TTA"),
        ("test_exploratory_no_tta", "Test Exploratory (153 imgs) — No TTA"),
        ("test_exploratory_tta", "Test Exploratory (153 imgs) — TTA"),
        ("external_rangoon_exploratory_no_tta", "External Rangoon Exploratory (35 imgs) — No TTA"),
        ("external_rangoon_exploratory_tta", "External Rangoon Exploratory (35 imgs) — TTA"),
    ]:
        nv = n_eval["evaluations"][split_key]
        sv = s_eval["evaluations"][split_key]
        md_lines.append(f"| {split_label} | YOLOv8n | {nv['precision']:.4f} | {nv['recall']:.4f} | {nv['f1_score']:.4f} | {nv['mAP50']:.4f} | {nv['mAP50_95']:.4f} |")
        md_lines.append(f"| {split_label} | YOLOv8s | {sv['precision']:.4f} | {sv['recall']:.4f} | {sv['f1_score']:.4f} | {sv['mAP50']:.4f} | {sv['mAP50_95']:.4f} |")

    md_lines.extend([
        "",
        "## 4. Artifact Paths (Confusion Matrices & PR Curves)",
        "",
        f"- **YOLOv8n Training Curves & PR/CM:** `{n_eval['training_artifacts']['results_png']}`, `{n_eval['training_artifacts']['pr_curve_png']}`, `{n_eval['training_artifacts']['confusion_matrix_png']}`",
        f"- **YOLOv8s Training Curves & PR/CM:** `{s_eval['training_artifacts']['results_png']}`, `{s_eval['training_artifacts']['pr_curve_png']}`, `{s_eval['training_artifacts']['confusion_matrix_png']}`",
    ])

    md_out = os.path.join(BENCHMARK_DIR, "phase2_yolov8n_vs_yolov8s_report.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[REPORT] Wrote {json_out}")
    print(f"[REPORT] Wrote {md_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["s", "n", "eval", "all"], default="all")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    s_name = "pothole_unified_yolov8s_100e"
    n_name = "pothole_unified_yolov8n_100e_sgd"

    if args.stage in ("s", "all"):
        run_controlled_training("yolov8s.pt", s_name, initial_batch=args.batch)
    if args.stage in ("n", "all"):
        run_controlled_training("yolov8n.pt", n_name, initial_batch=args.batch)
    if args.stage in ("eval", "all"):
        s_eval = evaluate_checkpoint(s_name)
        n_eval = evaluate_checkpoint(n_name)
        write_comparison_reports(s_eval, n_eval)


if __name__ == "__main__":
    main()
