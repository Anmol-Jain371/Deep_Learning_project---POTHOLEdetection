"""
Phase 13 / 13B: Controlled Detection Model Evaluation (YOLOv8n vs YOLOv8s).

Separates:
  A. Standard AP Evaluation (`conf=0.001`, `iou=0.50`, `imgsz=640`) for PR-curve integration.
  B. Fixed Operating-Point Evaluation (`conf=0.25`, `nms_iou=0.50`, `matching_iou=0.50`)
     computing exact one-to-one IoU prediction-to-GT matching counts (TP, FP, FN) and
     mathematically reconciled Precision, Recall, and F1 without artificial zero counts.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO


def _cuda_sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(512 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def _resolve_split_image_paths(data_yaml: str, split_name: str) -> List[str]:
    with open(data_yaml, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    base_dir = os.path.dirname(os.path.abspath(data_yaml))
    root_val = d.get("path")
    root_dir = (
        root_val
        if (root_val and os.path.isabs(str(root_val)))
        else os.path.normpath(os.path.join(base_dir, str(root_val or ".")))
    )
    split_rel = d.get(split_name)
    if not split_rel:
        return []
    target_dir = split_rel if os.path.isabs(str(split_rel)) else os.path.normpath(os.path.join(root_dir, str(split_rel)))
    if not os.path.isdir(target_dir):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        os.path.join(target_dir, fn)
        for fn in sorted(os.listdir(target_dir))
        if os.path.splitext(fn)[1].lower() in exts
    ]


def _load_gt_boxes_xyxy(img_path: str, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
    """Load YOLO normalized ground-truth boxes for an image path."""
    norm_p = img_path.replace("\\", "/")
    if "/images/" in norm_p:
        lbl_p = norm_p.replace("/images/", "/labels/", 1)
    else:
        lbl_p = norm_p
    lbl_p = os.path.splitext(lbl_p)[0] + ".txt"
    if not os.path.isfile(lbl_p):
        return []
    gts: List[Tuple[int, np.ndarray]] = []
    with open(lbl_p, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (xc - 0.5 * bw) * img_w
            y1 = (yc - 0.5 * bh) * img_h
            x2 = (xc + 0.5 * bw) * img_w
            y2 = (yc + 0.5 * bh) * img_h
            gts.append((cls_id, np.array([x1, y1, x2, y2], dtype=np.float64)))
    return gts


def _evaluate_fixed_operating_point(
    model: YOLO,
    img_paths: List[str],
    *,
    conf_threshold: float,
    nms_iou: float,
    matching_iou: float,
    imgsz: int,
    device: str,
) -> Dict[str, Any]:
    """
    Explicit one-to-one greedy IoU prediction-to-ground-truth matching at fixed confidence.
    Rules:
      - Predictions above conf_threshold (post-NMS at nms_iou) are sorted by confidence descending.
      - Each prediction matches at most one unmatched GT box of identical class with IoU >= matching_iou.
      - Duplicate predictions targeting an already-matched GT box and unmatched background predictions count as FP.
      - Unmatched GT boxes count as FN.
    """
    tp = fp = fn = 0
    dup_fp = bg_fp = gt_total = pred_total = images_eval = 0

    for im_p in img_paths:
        img = cv2.imread(im_p)
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = _load_gt_boxes_xyxy(im_p, W, H)
        gt_total += len(gts)
        matched_gt = [False] * len(gts)
        images_eval += 1

        res = model.predict(
            img,
            imgsz=imgsz,
            conf=conf_threshold,
            iou=nms_iou,
            device=device,
            verbose=False,
        )[0]
        preds: List[Tuple[float, int, np.ndarray]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxys = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for bx, cf, cl in zip(xyxys, confs, clss):
                preds.append((float(cf), int(cl), np.asarray(bx, dtype=np.float64)))
        preds.sort(key=lambda item: -item[0])
        pred_total += len(preds)

        for _, pred_cls, pred_box in preds:
            best_idx = -1
            best_iou = -1.0
            best_any_matched_iou = -1.0
            for gi, (gt_cls, gt_box) in enumerate(gts):
                if gt_cls != pred_cls:
                    continue
                ov = _box_iou_xyxy(pred_box, gt_box)
                if matched_gt[gi]:
                    if ov > best_any_matched_iou:
                        best_any_matched_iou = ov
                    continue
                if ov >= matching_iou and ov > best_iou:
                    best_iou = ov
                    best_idx = gi
            if best_idx >= 0:
                matched_gt[best_idx] = True
                tp += 1
            else:
                fp += 1
                if best_any_matched_iou >= matching_iou:
                    dup_fp += 1
                else:
                    bg_fp += 1

        fn += sum(1 for m in matched_gt if not m)

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "images_evaluated": images_eval,
        "gt_boxes": gt_total,
        "pred_boxes": pred_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicate_fp": dup_fp,
        "background_fp": bg_fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "matching_protocol": {
            "conf_threshold": conf_threshold,
            "nms_iou_threshold": nms_iou,
            "matching_iou_threshold": matching_iou,
            "class_matching": "exact_class_id",
            "one_to_one_greedy_by_confidence": True,
        },
    }


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
    op_conf = float(det_cfg["conf_threshold"])  # 0.25 fixed operating point
    ap_conf = float(det_cfg.get("ap_conf_threshold", 0.001))
    matching_iou = float(det_cfg["iou_threshold"])  # 0.50
    nms_iou = float(det_cfg.get("nms_iou_threshold", 0.50))
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

        ckpt_sha = _sha256_file(ckpt_path)
        model = YOLO(ckpt_path)
        splits_metrics: Dict[str, Any] = {}
        splits_to_run = det_cfg["splits"][:2] if smoke else det_cfg["splits"]

        for sp in splits_to_run:
            sp_id = sp["id"]
            data_yaml = os.path.join(project_root, sp["data_yaml"])
            if not os.path.isfile(data_yaml):
                continue

            # A. Standard PR-curve AP evaluation (low conf=0.001, ap_iou=0.50 vs default 0.70 documented)
            val_std = model.val(
                data=data_yaml,
                split=sp["split"],
                imgsz=imgsz,
                conf=ap_conf,
                iou=matching_iou,
                device=device,
                workers=0,
                verbose=False,
                plots=False,
            )
            box_std = val_std.box

            # B. Thresholded val(conf=0.25) for reconciliation with Phase 13
            val_op = model.val(
                data=data_yaml,
                split=sp["split"],
                imgsz=imgsz,
                conf=op_conf,
                iou=matching_iou,
                device=device,
                workers=0,
                verbose=False,
                plots=False,
            )
            box_op = val_op.box
            op_prec_lib = float(box_op.mp)
            op_rec_lib = float(box_op.mr)
            op_f1_lib = (2 * op_prec_lib * op_rec_lib / (op_prec_lib + op_rec_lib)) if (op_prec_lib + op_rec_lib) > 0 else 0.0

            # C. Explicit one-to-one IoU matching for exact TP / FP / FN at conf=0.25, nms_iou=0.50, match_iou=0.50
            sp_imgs = _resolve_split_image_paths(data_yaml, sp["split"])
            op_counts = _evaluate_fixed_operating_point(
                model,
                sp_imgs,
                conf_threshold=op_conf,
                nms_iou=nms_iou,
                matching_iou=matching_iou,
                imgsz=imgsz,
                device=device,
            )

            splits_metrics[sp_id] = {
                "role": f"{sp['role']} (EXPLORATORY ONLY)",
                "independent": bool(sp["independent"]),
                "ap_eval_conf_0_001": {
                    "conf_threshold": ap_conf,
                    "nms_iou": matching_iou,
                    "map50": round(float(box_std.map50), 4),
                    "map50_95": round(float(box_std.map), 4),
                    "library_best_f1_precision": round(float(box_std.mp), 4),
                    "library_best_f1_recall": round(float(box_std.mr), 4),
                },
                "thresholded_val_conf_0_25": {
                    "conf_threshold": op_conf,
                    "iou": matching_iou,
                    "map50": round(float(box_op.map50), 4),
                    "map50_95": round(float(box_op.map), 4),
                    "library_best_f1_precision": round(op_prec_lib, 4),
                    "library_best_f1_recall": round(op_rec_lib, 4),
                    "library_f1": round(op_f1_lib, 4),
                },
                "operating_point_exact_matching": op_counts,
                # Top-level canonical fixed-operating-point metrics derived directly from TP/FP/FN
                "precision": op_counts["precision"],
                "recall": op_counts["recall"],
                "f1_score": op_counts["f1_score"],
                "map50": round(float(box_std.map50), 4),
                "map50_conf_0_25": round(float(box_op.map50), 4),
                "map50_95": round(float(box_std.map), 4),
                "tp": op_counts["tp"],
                "fp": op_counts["fp"],
                "fn": op_counts["fn"],
                "duplicate_fp": op_counts["duplicate_fp"],
                "background_fp": op_counts["background_fp"],
                "gt_boxes": op_counts["gt_boxes"],
                "images_evaluated": op_counts["images_evaluated"],
            }

        latency_stats = _benchmark_detector_latency(
            model,
            sample_img_paths,
            conf=op_conf,
            iou=nms_iou,
            imgsz=imgsz,
            device=device,
        )
        if model_name == "yolov8n_100e_sgd":
            qualitative = _collect_qualitative_candidates(
                project_root, model, conf=op_conf, iou=nms_iou, imgsz=imgsz, device=device
            )

        models_out[model_name] = {
            "checkpoint": rel_ckpt,
            "checkpoint_sha256": ckpt_sha,
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
            "has 0 annotations and dataset_unified contains 328 suspected cross-split near-duplicate pairs dHash<=6). "
            "All dataset_unified and external_rangoon metrics are EXPLORATORY ONLY."
        ),
        "software_versions": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "evaluation_settings": {
            "standard_ap_conf_threshold": ap_conf,
            "operating_point_conf_threshold": op_conf,
            "nms_iou_threshold": nms_iou,
            "matching_iou_threshold": matching_iou,
            "imgsz": imgsz,
            "device": device,
        },
        "models": models_out,
        "qualitative_exploratory_proposals": qualitative,
    }
