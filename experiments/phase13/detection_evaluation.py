"""
Phase 13 Step 13: Controlled Detection Model Evaluation (YOLOv8n vs YOLOv8s).

Evaluates existing frozen checkpoints under identical settings (`conf=0.25`, `iou=0.50`,
`imgsz=640`) on `dataset_unified` annotated splits (`val`, `test`, `external_rangoon`),
records GPU latency/throughput/VRAM, collects qualitative failure examples from exploratory
footage, and reports independent detection generalization as BLOCKED when no untouched
annotated test split exists.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _cuda_sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark_detector_latency(
    model: YOLO,
    img_paths: List[str],
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> Dict[str, Any]:
    if not img_paths:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "fps": None, "peak_allocated_mb": None}

    # Warmup on synthetic frame
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(dummy, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    _cuda_sync(device)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies: List[float] = []
    imgs = [cv2.imread(p) for p in img_paths]
    imgs = [im for im in imgs if im is not None]
    for im in imgs:
        _cuda_sync(device)
        t0 = time.perf_counter()
        model.predict(im, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        _cuda_sync(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies, dtype=np.float64)
    total_sec = float(np.sum(arr) / 1000.0)
    peak_alloc = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if (device == "cuda" and torch.cuda.is_available())
        else None
    )
    peak_res = (
        round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
        if (device == "cuda" and torch.cuda.is_available())
        else None
    )
    return {
        "n": int(len(arr)),
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95.0)), 3),
        "throughput_fps": round(len(arr) / total_sec, 2) if total_sec > 0 else None,
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_res,
    }


def _collect_qualitative_candidates(
    project_root: str,
    model: YOLO,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Collect qualitative detector proposal observations (NOT ground-truth accuracy!) from
    exploratory road clips for manual failure taxonomy.
    """
    examples: List[Dict[str, Any]] = []
    sources = [
        ("rangoon_clip", "Watch Video_ Terrible potholes in Rangoon Road.mp4", [60, 150, 330, 500], "water/shadow/urban_boundary"),
        ("ruralRoad_202f", "uploads/upload_1794847200_ruralRoad_potHoles.mp4", [15, 60, 120, 175], "unpaved_texture/distant_small"),
        ("rawSurvey_160f", "roadvision-dashboard/public/videos/raw_survey.mp4", [20, 75, 130], "road_patch/roadside_shadow"),
    ]
    for src_id, rel_path, frame_indices, focus_tags in sources:
        full_path = os.path.join(project_root, rel_path)
        if not os.path.isfile(full_path):
            continue
        cap = cv2.VideoCapture(full_path)
        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            H, W = frame.shape[:2]
            res = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy().tolist() if (res.boxes is not None and len(res.boxes) > 0) else []
            confs = res.boxes.conf.cpu().numpy().tolist() if (res.boxes is not None and len(res.boxes) > 0) else []
            clipped = sum(1 for b in boxes if b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
            small_distant = sum(1 for b in boxes if ((b[2] - b[0]) * (b[3] - b[1])) / max(1, W * H) < 0.005)
            examples.append(
                {
                    "source_id": src_id,
                    "frame_idx": fidx,
                    "frame_wh": [W, H],
                    "proposal_count": len(boxes),
                    "clipped_boundary_proposals": clipped,
                    "small_distant_proposals": small_distant,
                    "max_conf": round(max(confs), 4) if confs else None,
                    "min_conf": round(min(confs), 4) if confs else None,
                    "potential_failure_tags": focus_tags,
                    "note": "Unlabeled exploratory proposal inspection only (not ground-truth verified).",
                }
            )
        cap.release()
    return examples


def run_detection_evaluation(
    project_root: str,
    cfg: Dict[str, Any],
    *,
    smoke: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    det_cfg = cfg["detection_eval"]
    conf = float(det_cfg["conf_threshold"])
    iou = float(det_cfg["iou_threshold"])
    imgsz = int(det_cfg["imgsz"])
    lat_cap = 16 if smoke else 60

    models_out: Dict[str, Any] = {}
    checkpoints = cfg["detector_checkpoints"]
    qualitative: List[Dict[str, Any]] = []

    val_imgs_dir = os.path.join(project_root, "dataset_unified", "images", "test")
    sample_img_paths = (
        [os.path.join(val_imgs_dir, f) for f in sorted(os.listdir(val_imgs_dir))[:lat_cap]]
        if os.path.isdir(val_imgs_dir)
        else []
    )

    for model_name, rel_ckpt in checkpoints.items():
        ckpt_path = os.path.join(project_root, rel_ckpt)
        if not os.path.isfile(ckpt_path):
            models_out[model_name] = {"status": "BLOCKED", "reason": f"Missing checkpoint {rel_ckpt}"}
            continue

        model = YOLO(ckpt_path)
        splits_metrics: Dict[str, Any] = {}
        splits_to_run = det_cfg["splits"][:2] if smoke else det_cfg["splits"]

        for sp in splits_to_run:
            sp_id = sp["id"]
            data_yaml = os.path.join(project_root, sp["data_yaml"])
            if not os.path.isfile(data_yaml):
                continue
            val_res = model.val(
                data=data_yaml,
                split=sp["split"],
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                workers=0,
                verbose=False,
                plots=False,
            )
            box = val_res.box
            prec = float(box.mp)
            rec = float(box.mr)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            cm = getattr(val_res, "confusion_matrix", None)
            fp_cnt = fn_cnt = tp_cnt = None
            if cm is not None and getattr(cm, "matrix", None) is not None:
                mat = cm.matrix
                if mat.shape[0] >= 2 and mat.shape[1] >= 2:
                    tp_cnt = int(mat[0, 0])
                    fp_cnt = int(mat[0, 1])
                    fn_cnt = int(mat[1, 0])

            splits_metrics[sp_id] = {
                "role": sp["role"],
                "independent": bool(sp["independent"]),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "map50": round(float(box.map50), 4),
                "map50_95": round(float(box.map), 4),
                "tp": tp_cnt,
                "fp": fp_cnt,
                "fn": fn_cnt,
            }

        latency_stats = _benchmark_detector_latency(
            model,
            sample_img_paths,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
        if model_name == "yolov8n_100e_sgd":
            qualitative = _collect_qualitative_candidates(
                project_root, model, conf=conf, iou=iou, imgsz=imgsz, device=device
            )

        models_out[model_name] = {
            "checkpoint": rel_ckpt,
            "parameters": int(sum(p.numel() for p in model.model.parameters())),
            "splits": splits_metrics,
            "runtime_benchmark": latency_stats,
        }
        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "status": "COMPLETED",
        "independent_generalization_status": "BLOCKED",
        "independent_generalization_reason": (
            "No untouched independent annotated pothole detection dataset exists (evaluation/ground_truth_labels/ "
            "has 0 annotations and dataset_unified contains cross-split perceptual near-duplicates). "
            "Reported dataset_unified metrics are EXPLORATORY ONLY."
        ),
        "evaluation_settings": {"conf_threshold": conf, "iou_threshold": iou, "imgsz": imgsz, "device": device},
        "models": models_out,
        "qualitative_exploratory_proposals": qualitative,
    }
