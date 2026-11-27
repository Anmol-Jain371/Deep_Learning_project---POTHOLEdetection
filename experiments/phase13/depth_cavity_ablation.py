"""
Phase 13 Steps 15 & 16: Sequential Depth Model/Resolution Ablation & Cavity Extraction Ablation.

Loads depth backbones strictly sequentially to fit comfortably inside 4 GB RTX 2050 VRAM,
separates initialization from CUDA-synchronized inference, evaluates FP32 vs FP16, compares
DAv2 Small resolutions against DAv2 518 as an internal consistency reference (NOT physical GT),
and evaluates full_box vs rect_trim_20 vs elliptical_core + Pothole440 grouped segmentation IoU/Dice.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from experiments.core_aware_cavity_extraction import CoreExtractionConfig, extract_cavity_with_strategy
from experiments.dav2_inference_optimization import spearman_fast
from experiments.depth_model_comparison import DepthAnythingV2SmallAdapter, MiDaSSmallAdapter
from experiments.optimized_relative_depth import OptimizedDepthConfig


def _load_sample_frames(project_root: str, cfg: Dict[str, Any], max_per_vid: int) -> List[Dict[str, Any]]:
    det_ckpt = os.path.join(project_root, cfg["detector_checkpoints"]["yolov8n_100e_sgd"])
    det = YOLO(det_ckpt)
    samples: List[Dict[str, Any]] = []
    for vid in cfg["exploratory_videos"]:
        vp = os.path.join(project_root, vid["path"])
        if not os.path.isfile(vp):
            continue
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
        indices = np.linspace(5, max(6, min(total - 2, 180)), max_per_vid, dtype=int).tolist()
        for fidx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            res = det.predict(frame, imgsz=640, conf=0.25, iou=0.50, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy().tolist() if (res.boxes is not None and len(res.boxes) > 0) else []
            if boxes:
                samples.append({"source_id": vid["id"], "frame_idx": int(fidx), "frame": frame, "boxes": boxes})
        cap.release()
    del det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return samples


def _eval_pothole440_segmentation(project_root: str, cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    seg_ckpt = os.path.join(project_root, cfg["segmentation_checkpoint"]["yolov8n_seg"])
    test_img_dir = os.path.join(project_root, "dataset_pothole440_seg_grouped", "images", "test")
    mask_root = os.path.join(project_root, "Pothole440-main", "Pothole440-main")
    if not (os.path.isfile(seg_ckpt) and os.path.isdir(test_img_dir)):
        return {"status": "BLOCKED", "reason": "Missing grouped Pothole440 test split or yolov8n-seg weights."}

    model = YOLO(seg_ckpt)
    imgs = sorted(f for f in os.listdir(test_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    ious: List[float] = []
    dices: List[float] = []

    for fname in imgs:
        im_path = os.path.join(test_img_dir, fname)
        # Find original binary mask inside Pothole440-main/{train,val,test}/mask/
        gt_mask = None
        for sp in ("train", "val", "test"):
            cand = os.path.join(mask_root, sp, "mask", fname)
            if os.path.isfile(cand):
                gt_mask = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
                break
        if gt_mask is None:
            continue
        gt_bin = gt_mask > 127
        H, W = gt_bin.shape[:2]
        res = model.predict(im_path, imgsz=640, conf=0.25, iou=0.50, device=device, verbose=False)[0]
        pred_bin = np.zeros((H, W), dtype=bool)
        if res.masks is not None and res.masks.data is not None and len(res.masks.data) > 0:
            m_arr = res.masks.data.cpu().numpy()
            for m in m_arr:
                m_rs = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                pred_bin |= m_rs

        inter = int(np.sum(gt_bin & pred_bin))
        union = int(np.sum(gt_bin | pred_bin))
        denom_dice = int(np.sum(gt_bin) + np.sum(pred_bin))
        ious.append(inter / union if union > 0 else 1.0)
        dices.append((2.0 * inter) / denom_dice if denom_dice > 0 else 1.0)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "COMPLETED",
        "benchmark_name": "dataset_pothole440_seg_grouped_test (reconstructed_grouped_originals)",
        "n_images": len(ious),
        "mean_mask_iou": round(float(np.mean(ious)), 4) if ious else None,
        "median_mask_iou": round(float(np.median(ious)), 4) if ious else None,
        "mean_dice": round(float(np.mean(dices)), 4) if dices else None,
        "domain_shift_note": (
            "Pothole440 contains close-up overhead pothole images domain-shifted from dashcam road scenes "
            "(N=15 unaugmented originals). Improved mask IoU does NOT prove physical depth accuracy."
        ),
    }


def run_depth_and_cavity_ablation(
    project_root: str,
    cfg: Dict[str, Any],
    *,
    smoke: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    max_per_vid = 2 if smoke else int(cfg["full_limits"]["depth_frames_per_video"] // 2)
    samples = _load_sample_frames(project_root, cfg, max_per_vid=max(2, max_per_vid))
    if not samples:
        return {"status": "BLOCKED", "reason": "No exploratory video frames with detections available."}

    opt_cfg = OptimizedDepthConfig(seed=42)
    core_cfg = CoreExtractionConfig()

    # Arms run sequentially so VRAM stays well below 4 GB
    arms = (
        [
            ("midas_256_fp16", "midas_small", 256, True),
            ("dav2_266_fp16", "depth_anything_v2_small", 266, True),
            ("dav2_336_fp16", "depth_anything_v2_small", 336, True),
            ("dav2_336_fp32", "depth_anything_v2_small", 336, False),
            ("dav2_518_fp16", "depth_anything_v2_small", 518, True),
        ]
        if smoke
        else [
            ("midas_256_fp32", "midas_small", 256, False),
            ("midas_256_fp16", "midas_small", 256, True),
            ("dav2_266_fp16", "depth_anything_v2_small", 266, True),
            ("dav2_336_fp16", "depth_anything_v2_small", 336, True),
            ("dav2_336_fp32", "depth_anything_v2_small", 336, False),
            ("dav2_392_fp16", "depth_anything_v2_small", 392, True),
            ("dav2_518_fp16", "depth_anything_v2_small", 518, True),
        ]
    )

    depth_results: Dict[str, Any] = {}
    ref_maps: Dict[int, np.ndarray] = {}
    dav2_336_maps: Dict[int, np.ndarray] = {}

    # Compute 518 reference first if running in reversed or capture at end — actually run 518 first or compute Spearman after!
    ordered_arms = sorted(arms, key=lambda a: 0 if a[0] == "dav2_518_fp16" else 1)

    for arm_id, mtype, inp_size, use_fp16 in ordered_arms:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_init0 = time.perf_counter()
        if mtype == "midas_small":
            adapter = MiDaSSmallAdapter(device=device, use_fp16=use_fp16)
        else:
            adapter = DepthAnythingV2SmallAdapter(device=device, input_size=inp_size, use_fp16=use_fp16)
        adapter.warmup(360, 640)
        init_ms = (time.perf_counter() - t_init0) * 1000.0

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        inf_lats: List[float] = []
        tot_lats: List[float] = []
        inliers: List[float] = []
        scores: List[float] = []
        spearman_vs_518: List[float] = []
        total_boxes = 0
        valid_boxes = 0
        rej_reasons: Dict[str, int] = {}

        for s_idx, smp in enumerate(samples):
            pred = adapter.predict(smp["frame"])
            inf_lats.append(pred.timings_ms["inference_ms"])
            tot_lats.append(pred.timings_ms["total_ms"])
            if arm_id == "dav2_518_fp16":
                ref_maps[s_idx] = pred.depth_map
            if arm_id == "dav2_336_fp16":
                dav2_336_maps[s_idx] = pred.depth_map

            if s_idx in ref_maps:
                spearman_vs_518.append(spearman_fast(pred.depth_map, ref_maps[s_idx], stride=8))

            for bx in smp["boxes"]:
                total_boxes += 1
                out = extract_cavity_with_strategy(
                    pred.depth_map,
                    tuple(bx),
                    strategy="elliptical_core",
                    opt_cfg=opt_cfg,
                    core_cfg=core_cfg,
                    img_bgr=smp["frame"],
                )
                ir = out.get("ref_diagnostics", {}).get("inlier_ratio")
                if ir is not None:
                    inliers.append(float(ir))
                if out["is_valid"]:
                    valid_boxes += 1
                    p75 = (out.get("core_cavity_stats") or {}).get("p75_residual")
                    if p75 is not None:
                        scores.append(float(p75))
                else:
                    for rc in out.get("failure_reasons") or []:
                        rej_reasons[rc] = rej_reasons.get(rc, 0) + 1

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

        depth_results[arm_id] = {
            "model_id": mtype,
            "input_size": inp_size,
            "precision": "fp16_autocast" if (use_fp16 and device == "cuda") else "fp32",
            "init_warmup_ms": round(init_ms, 1),
            "n_frames": len(samples),
            "n_boxes": total_boxes,
            "fresh_valid_count": valid_boxes,
            "fresh_valid_coverage_pct": round((valid_boxes / max(1, total_boxes)) * 100.0, 2),
            "mean_inlier_ratio": round(float(np.mean(inliers)), 4) if inliers else None,
            "relative_score_cv": round(float(np.std(scores) / max(np.mean(scores), 1e-6)), 4) if len(scores) >= 2 else None,
            "mean_spearman_vs_dav2_518": round(float(np.mean(spearman_vs_518)), 4) if spearman_vs_518 else None,
            "inference_ms_mean": round(float(np.mean(inf_lats)), 3),
            "inference_ms_median": round(float(np.median(inf_lats)), 3),
            "inference_ms_p95": round(float(np.percentile(inf_lats, 95.0)), 3),
            "total_ms_mean": round(float(np.mean(tot_lats)), 3),
            "peak_allocated_mb": peak_alloc,
            "peak_reserved_mb": peak_res,
            "top_rejections": rej_reasons,
        }

        del adapter
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Cavity extraction ablation over identical DAv2 336 depth maps
    cavity_strategies = ("full_box", "rect_trim_20", "elliptical_core")
    cavity_results: Dict[str, Any] = {}
    for strat in cavity_strategies:
        n_box = n_val = 0
        inliers: List[float] = []
        scores: List[float] = []
        lats: List[float] = []
        for s_idx, smp in enumerate(samples):
            dmap = dav2_336_maps.get(s_idx)
            if dmap is None:
                continue
            for bx in smp["boxes"]:
                n_box += 1
                t0 = time.perf_counter()
                out = extract_cavity_with_strategy(
                    dmap,
                    tuple(bx),
                    strategy=strat,
                    opt_cfg=opt_cfg,
                    core_cfg=core_cfg,
                    img_bgr=smp["frame"],
                )
                lats.append((time.perf_counter() - t0) * 1000.0)
                ir = out.get("ref_diagnostics", {}).get("inlier_ratio")
                if ir is not None:
                    inliers.append(float(ir))
                if out["is_valid"]:
                    n_val += 1
                    p75 = (out.get("core_cavity_stats") or {}).get("p75_residual")
                    if p75 is not None:
                        scores.append(float(p75))
        cavity_results[strat] = {
            "n_boxes": n_box,
            "valid_count": n_val,
            "validity_coverage_pct": round((n_val / max(1, n_box)) * 100.0, 2),
            "mean_inlier_ratio": round(float(np.mean(inliers)), 4) if inliers else None,
            "relative_score_cv": round(float(np.std(scores) / max(np.mean(scores), 1e-6)), 4) if len(scores) >= 2 else None,
            "mean_extraction_ms": round(float(np.mean(lats)), 3) if lats else None,
        }

    seg_eval = _eval_pothole440_segmentation(project_root, cfg, device=device)

    return {
        "status": "COMPLETED",
        "depth_model_and_resolution_ablation": depth_results,
        "cavity_extraction_ablation": cavity_results,
        "segmentation_grouped_test_evaluation": seg_eval,
        "scientific_caveat": (
            "DAv2 518 serves strictly as a higher-resolution relative consistency reference, NOT physical metric ground truth. "
            "MiDaS and DAv2 raw residual magnitudes are not directly comparable."
        ),
    }
