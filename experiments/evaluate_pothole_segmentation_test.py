"""
========================================================================================
RoadVision AI -- Phase 4A.3: Independent Grouped Test & Box-vs-Mask Evaluation
========================================================================================
Evaluates the frozen validation-selected YOLOv8n-seg checkpoint strictly once under the
pre-registered protocol (`reports/pothole440_segmentation_evaluation_protocol.md`):

  1. Integrity checks: verifies best.pt SHA256 and frozen dataset_manifest SHA256s.
  2. Official Ultralytics `test` AP metrics (Box & Mask Precision, Recall, F1, mAP@50, mAP@50-95).
  3. Region-level evaluation at primary `conf=0.50` (+ secondary `conf=0.25` reference)
     against original 1200x800 Pothole440 binary PNG masks:
       - Macro & Micro IoU, Dice, Pixel Precision, Pixel Recall (empty preds -> zero mask)
       - Boundary IoU & Boundary F1 at predefined tolerances {2, 5, 10} pixels
       - Connected-component missed GT & false-positive mask regions
       - 10,000-resample 95% percentile bootstrap confidence intervals
  4. Box-vs-Mask representation comparison on the exact same matched predictions:
       - Rectangular filled box mask vs reconstructed segmentation mask
       - Macro/Micro IoU & Dice, FP pavement area, FN cavity area, relative gain
  5. Test segmentation 8-panel overlays + Box-vs-Mask side-by-side visualizations.
========================================================================================
"""

import os
import csv
import json
import hashlib
import argparse
from typing import Dict, Any, List, Tuple

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_YAML = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "data.yaml")
MANIFEST_JSON = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "manifests", "dataset_manifest.json")
SPLIT_CSV = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "manifests", "split_manifest.csv")
SRC_ROOT = os.path.join(BASE_DIR, "Pothole440-main", "Pothole440-main")
DEFAULT_RUN = os.path.join(BASE_DIR, "runs", "segment", "pothole440_yolov8n_seg_v1")
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
TEST_VIZ_DIR = os.path.join(BENCH_DIR, "pothole440_seg_test_viz")
BVM_VIZ_DIR = os.path.join(BENCH_DIR, "pothole440_box_vs_mask_viz")

EXPECTED_BEST_SHA256 = "73f525322681468322dd129589279b016eeeab1197879a8876e3071e8e0d6d6d"
EXPECTED_MANIFEST_SHA256 = "cff0e21e8341a96a7c95c114aa1ea1509874c1047dfae1a9b2ea85c1fe972a16"
EXPECTED_SPLIT_CSV_SHA256 = "63b2a1ca49e49b1fd859c8f9b809a9bbba67ff51c629f0092ea0bd7bd8dae643"
EXPECTED_YAML_SHA256 = "bd97956a2912ece0a30500ac040c86380aae6b8327d696ec170688a027cb8cfc"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_boundary(mask_bool: np.ndarray) -> np.ndarray:
    u8 = mask_bool.astype(np.uint8)
    eroded = cv2.erode(u8, np.ones((3, 3), np.uint8), iterations=1)
    return (u8 - eroded) > 0


def compute_boundary_metrics(gt: np.ndarray, pred: np.ndarray, tol_px: int) -> Tuple[float, float]:
    gb = extract_boundary(gt)
    pb = extract_boundary(pred)
    g_cnt = int(gb.sum())
    p_cnt = int(pb.sum())
    if g_cnt == 0 and p_cnt == 0:
        return 1.0, 1.0
    if g_cnt == 0 or p_cnt == 0:
        return 0.0, 0.0
    k = 2 * tol_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gb_dil = cv2.dilate(gb.astype(np.uint8), kernel) > 0
    pb_dil = cv2.dilate(pb.astype(np.uint8), kernel) > 0
    tp_g = int(np.logical_and(gb, pb_dil).sum())
    tp_p = int(np.logical_and(pb, gb_dil).sum())
    b_rec = tp_g / float(g_cnt)
    b_prec = tp_p / float(p_cnt)
    b_f1 = (2.0 * b_prec * b_rec / (b_prec + b_rec)) if (b_prec + b_rec) > 0 else 0.0
    union = int(np.logical_or(gb_dil, pb_dil).sum())
    b_iou = int(np.logical_and(gb_dil, pb_dil).sum()) / float(union) if union > 0 else 1.0
    return b_iou, b_f1


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": round(float(values.mean()), 4),
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
    }


def predict_seg_and_box_masks(
    model: YOLO, img_bgr: np.ndarray, imgsz: int, conf: float, iou_nms: float, device: Any
) -> Tuple[np.ndarray, np.ndarray, int, List[float]]:
    H, W = img_bgr.shape[:2]
    res = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou_nms,
        retina_masks=True,
        device=device,
        verbose=False,
    )
    empty = np.zeros((H, W), dtype=bool)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return empty, empty.copy(), 0, []

    boxes_xyxy = res[0].boxes.xyxy.cpu().numpy()
    confs = [round(float(c), 4) for c in res[0].boxes.conf.cpu().numpy()]

    box_mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy:
        ix1 = max(0, int(np.floor(x1)))
        iy1 = max(0, int(np.floor(y1)))
        ix2 = min(W, int(np.ceil(x2)))
        iy2 = min(H, int(np.ceil(y2)))
        if ix2 > ix1 and iy2 > iy1:
            box_mask[iy1:iy2, ix1:ix2] = True

    if res[0].masks is None or res[0].masks.data is None or len(res[0].masks.data) == 0:
        return empty, box_mask, int(len(boxes_xyxy)), confs

    m_np = res[0].masks.data.cpu().numpy()
    if m_np.shape[1] != H or m_np.shape[2] != W:
        resized = [cv2.resize(m_np[i], (W, H), interpolation=cv2.INTER_LINEAR) >= 0.5 for i in range(m_np.shape[0])]
        seg_mask = np.any(np.stack(resized, axis=0), axis=0)
    else:
        seg_mask = np.any(m_np >= 0.5, axis=0)
    return seg_mask, box_mask, int(len(boxes_xyxy)), confs


def compute_mask_metrics(gt: np.ndarray, pred: np.ndarray, include_multiscale_boundary: bool = True) -> Dict[str, Any]:
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())

    union = tp + fp + fn
    iou = (tp / float(union)) if union > 0 else 1.0
    dice = (2.0 * tp / float(2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    prec = (tp / float(tp + fp)) if (tp + fp) > 0 else (1.0 if (tp + fn) == 0 else 0.0)
    rec = (tp / float(tp + fn)) if (tp + fn) > 0 else 1.0

    out: Dict[str, Any] = {
        "tp_px": tp,
        "fp_px": fp,
        "fn_px": fn,
        "tn_px": tn,
        "gt_fg_pct": round(float(gt.mean() * 100.0), 3),
        "pred_fg_pct": round(float(pred.mean() * 100.0), 3),
        "iou": iou,
        "dice": dice,
        "pixel_precision": prec,
        "pixel_recall": rec,
    }

    if include_multiscale_boundary:
        for tol in (2, 5, 10):
            biou, bf1 = compute_boundary_metrics(gt, pred, tol_px=tol)
            out[f"boundary_iou_{tol}px"] = biou
            out[f"boundary_f1_{tol}px"] = bf1

        n_gt_lbl, gt_lbls, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
        n_gt_comp = max(0, n_gt_lbl - 1)
        missed_gt_comp = 0
        for cid in range(1, n_gt_lbl):
            if not np.any(np.logical_and(gt_lbls == cid, pred)):
                missed_gt_comp += 1

        n_pr_lbl, pr_lbls, pr_stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
        n_pr_comp = max(0, n_pr_lbl - 1)
        fp_pr_comp = 0
        for cid in range(1, n_pr_lbl):
            if pr_stats[cid, cv2.CC_STAT_AREA] < 4:
                continue
            if not np.any(np.logical_and(pr_lbls == cid, gt)):
                fp_pr_comp += 1

        out["gt_components"] = n_gt_comp
        out["missed_gt_regions"] = missed_gt_comp
        out["pred_components"] = n_pr_comp
        out["fp_pred_regions"] = fp_pr_comp

    return out


def render_test_visualization(idx: int, fn: str, rgb_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray, m: Dict[str, Any]) -> str:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    tp_map = np.logical_and(gt, pred)
    fp_map = np.logical_and(~gt, pred)
    fn_map = np.logical_and(gt, ~pred)

    err_vis = (rgb * 0.35).astype(np.uint8)
    err_vis[tp_map] = [34, 197, 94]
    err_vis[fp_map] = [239, 68, 68]
    err_vis[fn_map] = [59, 130, 246]

    ov = rgb.copy()
    tint = np.zeros_like(rgb)
    tint[pred] = [245, 158, 11]
    ov = cv2.addWeighted(ov, 0.70, tint, 0.30, 0)
    gt_cnt, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, gt_cnt, -1, (0, 255, 255), 3)

    def mono_panel(mask_b, color_rgb):
        p = np.zeros((H, W, 3), dtype=np.uint8)
        p[mask_b] = color_rgb
        return p

    fig = plt.figure(figsize=(16, 8.6), dpi=135)
    gs = GridSpec(2, 4, figure=fig, hspace=0.22, wspace=0.14)
    panels = [
        (rgb, "Original RGB", None),
        (gt.astype(np.uint8) * 255, f"Original GT Mask ({m['gt_fg_pct']}%)", "gray"),
        (pred.astype(np.uint8) * 255, f"Predicted Mask ({m['pred_fg_pct']}%, inst={m['pred_instances']})", "gray"),
        (ov, "Overlay (Cyan=GT, Amber=Pred)", None),
        (err_vis, "Error Map (G=TP, R=FP, B=FN)", None),
        (mono_panel(tp_map, [34, 197, 94]), f"TP Pixels ({m['tp_px']})", None),
        (mono_panel(fp_map, [239, 68, 68]), f"FP Pixels ({m['fp_px']})", None),
        (mono_panel(fn_map, [59, 130, 246]), f"FN Pixels ({m['fn_px']})", None),
    ]
    for i, (img_p, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.imshow(img_p, cmap=cmap)
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    fig.suptitle(
        f"Test #{idx:02d}: {fn} | IoU={m['iou']:.4f}, Dice={m['dice']:.4f}, Prec={m['pixel_precision']:.4f}, "
        f"Rec={m['pixel_recall']:.4f}, B-F1(2/5/10px)={m['boundary_f1_2px']:.2f}/{m['boundary_f1_5px']:.2f}/{m['boundary_f1_10px']:.2f}",
        fontsize=11.0, fontweight="bold"
    )
    out_p = os.path.join(TEST_VIZ_DIR, f"test_{idx:02d}_{os.path.splitext(fn)[0]}.png")
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def render_box_vs_mask_viz(
    idx: int, fn: str, rgb_bgr: np.ndarray, gt: np.ndarray, box_mask: np.ndarray, seg_mask: np.ndarray, bm: Dict[str, Any], sm: Dict[str, Any]
) -> str:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    gt_cnt, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def make_err(pred_m):
        e = (rgb * 0.35).astype(np.uint8)
        e[np.logical_and(gt, pred_m)] = [34, 197, 94]
        e[np.logical_and(~gt, pred_m)] = [239, 68, 68]
        e[np.logical_and(gt, ~pred_m)] = [59, 130, 246]
        cv2.drawContours(e, gt_cnt, -1, (0, 255, 255), 2)
        return e

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=135)
    axes[0].imshow(rgb)
    cv2.drawContours(rgb_ov := rgb.copy(), gt_cnt, -1, (0, 255, 255), 3)
    axes[0].imshow(rgb_ov)
    axes[0].set_title(f"GT Contour ({sm['gt_fg_pct']}% area)", fontsize=10)
    axes[1].imshow(make_err(box_mask))
    axes[1].set_title(f"Box Mask | IoU={bm['iou']:.3f}, Dice={bm['dice']:.3f}, FP={bm['fp_px']}px", fontsize=9.5)
    axes[2].imshow(make_err(seg_mask))
    axes[2].set_title(f"Seg Mask | IoU={sm['iou']:.3f}, Dice={sm['dice']:.3f}, FP={sm['fp_px']}px", fontsize=9.5)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Box vs Seg #{idx:02d}: {fn} (Green=TP, Red=FP Pavement, Blue=FN Cavity, Cyan=GT)", fontsize=11, fontweight="bold")
    out_p = os.path.join(BVM_VIZ_DIR, f"bvm_{idx:02d}_{os.path.splitext(fn)[0]}.png")
    plt.tight_layout()
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def summarize_seg_suite(per_img: List[Dict[str, Any]], conf_th: float, iou_nms: float) -> Dict[str, Any]:
    ious = np.array([r["iou"] for r in per_img], dtype=float)
    dices = np.array([r["dice"] for r in per_img], dtype=float)
    precs = np.array([r["pixel_precision"] for r in per_img], dtype=float)
    recs = np.array([r["pixel_recall"] for r in per_img], dtype=float)
    b2_iou = np.array([r["boundary_iou_2px"] for r in per_img], dtype=float)
    b5_iou = np.array([r["boundary_iou_5px"] for r in per_img], dtype=float)
    b10_iou = np.array([r["boundary_iou_10px"] for r in per_img], dtype=float)
    b2_f1 = np.array([r["boundary_f1_2px"] for r in per_img], dtype=float)
    b5_f1 = np.array([r["boundary_f1_5px"] for r in per_img], dtype=float)
    b10_f1 = np.array([r["boundary_f1_10px"] for r in per_img], dtype=float)

    tot_tp = sum(r["tp_px"] for r in per_img)
    tot_fp = sum(r["fp_px"] for r in per_img)
    tot_fn = sum(r["fn_px"] for r in per_img)
    tot_tn = sum(r["tn_px"] for r in per_img)

    return {
        "conf_threshold": conf_th,
        "iou_nms_threshold": iou_nms,
        "test_images_count": len(per_img),
        "empty_prediction_images": sum(1 for r in per_img if r["pred_instances"] == 0),
        "macro_mean_iou": round(float(ious.mean()), 4),
        "macro_std_iou": round(float(ious.std(ddof=1)), 4),
        "macro_median_iou": round(float(np.median(ious)), 4),
        "macro_min_iou": round(float(ious.min()), 4),
        "macro_max_iou": round(float(ious.max()), 4),
        "macro_mean_dice": round(float(dices.mean()), 4),
        "macro_std_dice": round(float(dices.std(ddof=1)), 4),
        "macro_median_dice": round(float(np.median(dices)), 4),
        "macro_min_dice": round(float(dices.min()), 4),
        "macro_max_dice": round(float(dices.max()), 4),
        "macro_mean_pixel_precision": round(float(precs.mean()), 4),
        "macro_mean_pixel_recall": round(float(recs.mean()), 4),
        "macro_boundary_2px": {"iou": round(float(b2_iou.mean()), 4), "f1": round(float(b2_f1.mean()), 4)},
        "macro_boundary_5px": {"iou": round(float(b5_iou.mean()), 4), "f1": round(float(b5_f1.mean()), 4)},
        "macro_boundary_10px": {"iou": round(float(b10_iou.mean()), 4), "f1": round(float(b10_f1.mean()), 4)},
        "micro_pooled_iou": round(float(tot_tp / max(1, tot_tp + tot_fp + tot_fn)), 4),
        "micro_pooled_dice": round(float(2.0 * tot_tp / max(1, 2 * tot_tp + tot_fp + tot_fn)), 4),
        "micro_pooled_pixel_precision": round(float(tot_tp / max(1, tot_tp + tot_fp)), 4),
        "micro_pooled_pixel_recall": round(float(tot_tp / max(1, tot_tp + tot_fn)), 4),
        "pooled_pixel_totals": {"tp_px": tot_tp, "fp_px": tot_fp, "fn_px": tot_fn, "tn_px": tot_tn},
        "total_gt_regions": sum(r["gt_components"] for r in per_img),
        "completely_missed_gt_regions": sum(r["missed_gt_regions"] for r in per_img),
        "false_positive_mask_regions": sum(r["fp_pred_regions"] for r in per_img),
        "bootstrap_95_ci_10000": {
            "macro_iou": bootstrap_ci(ious),
            "macro_dice": bootstrap_ci(dices),
            "macro_pixel_precision": bootstrap_ci(precs),
            "macro_pixel_recall": bootstrap_ci(recs),
            "boundary_f1_2px": bootstrap_ci(b2_f1),
            "boundary_f1_5px": bootstrap_ci(b5_f1),
            "boundary_f1_10px": bootstrap_ci(b10_f1),
        },
    }


def summarize_box_vs_mask(per_img_pairs: List[Dict[str, Any]], conf_th: float) -> Dict[str, Any]:
    b_ious = np.array([r["box_iou"] for r in per_img_pairs], dtype=float)
    s_ious = np.array([r["seg_iou"] for r in per_img_pairs], dtype=float)
    b_dices = np.array([r["box_dice"] for r in per_img_pairs], dtype=float)
    s_dices = np.array([r["seg_dice"] for r in per_img_pairs], dtype=float)

    b_tp = sum(r["box_tp_px"] for r in per_img_pairs)
    b_fp = sum(r["box_fp_px"] for r in per_img_pairs)
    b_fn = sum(r["box_fn_px"] for r in per_img_pairs)
    b_tn = sum(r["box_tn_px"] for r in per_img_pairs)

    s_tp = sum(r["seg_tp_px"] for r in per_img_pairs)
    s_fp = sum(r["seg_fp_px"] for r in per_img_pairs)
    s_fn = sum(r["seg_fn_px"] for r in per_img_pairs)
    s_tn = sum(r["seg_tn_px"] for r in per_img_pairs)

    b_micro_iou = b_tp / float(max(1, b_tp + b_fp + b_fn))
    s_micro_iou = s_tp / float(max(1, s_tp + s_fp + s_fn))
    b_micro_dice = (2.0 * b_tp) / float(max(1, 2 * b_tp + b_fp + b_fn))
    s_micro_dice = (2.0 * s_tp) / float(max(1, 2 * s_tp + s_fp + s_fn))

    tot_bg_px = b_fp + b_tn
    tot_gt_px = b_tp + b_fn
    tot_img_px = tot_bg_px + tot_gt_px

    b_macro_iou = float(b_ious.mean())
    s_macro_iou = float(s_ious.mean())
    b_macro_dice = float(b_dices.mean())
    s_macro_dice = float(s_dices.mean())

    return {
        "comparison_type": "box_versus_mask_representation_comparison",
        "conf_threshold": conf_th,
        "images_count": len(per_img_pairs),
        "empty_prediction_images": sum(1 for r in per_img_pairs if r["pred_instances"] == 0),
        "box_mask": {
            "macro_mean_iou": round(b_macro_iou, 4),
            "macro_mean_dice": round(b_macro_dice, 4),
            "micro_pooled_iou": round(b_micro_iou, 4),
            "micro_pooled_dice": round(b_micro_dice, 4),
            "total_fp_pavement_px": b_fp,
            "fp_pavement_pct_of_bg": round(100.0 * b_fp / max(1, tot_bg_px), 3),
            "fp_pavement_pct_of_image": round(100.0 * b_fp / max(1, tot_img_px), 3),
            "total_fn_cavity_px": b_fn,
            "fn_cavity_pct_of_gt": round(100.0 * b_fn / max(1, tot_gt_px), 3),
        },
        "seg_mask": {
            "macro_mean_iou": round(s_macro_iou, 4),
            "macro_mean_dice": round(s_macro_dice, 4),
            "micro_pooled_iou": round(s_micro_iou, 4),
            "micro_pooled_dice": round(s_micro_dice, 4),
            "total_fp_pavement_px": s_fp,
            "fp_pavement_pct_of_bg": round(100.0 * s_fp / max(1, tot_bg_px), 3),
            "fp_pavement_pct_of_image": round(100.0 * s_fp / max(1, tot_img_px), 3),
            "total_fn_cavity_px": s_fn,
            "fn_cavity_pct_of_gt": round(100.0 * s_fn / max(1, tot_gt_px), 3),
        },
        "delta_seg_minus_box": {
            "macro_iou_abs": round(s_macro_iou - b_macro_iou, 4),
            "macro_iou_rel_pct": round(100.0 * (s_macro_iou - b_macro_iou) / max(1e-9, b_macro_iou), 2),
            "micro_iou_abs": round(s_micro_iou - b_micro_iou, 4),
            "micro_iou_rel_pct": round(100.0 * (s_micro_iou - b_micro_iou) / max(1e-9, b_micro_iou), 2),
            "macro_dice_abs": round(s_macro_dice - b_macro_dice, 4),
            "macro_dice_rel_pct": round(100.0 * (s_macro_dice - b_macro_dice) / max(1e-9, b_macro_dice), 2),
            "micro_dice_abs": round(s_micro_dice - b_micro_dice, 4),
            "micro_dice_rel_pct": round(100.0 * (s_micro_dice - b_micro_dice) / max(1e-9, b_micro_dice), 2),
            "fp_pavement_px_reduced": b_fp - s_fp,
            "fp_pavement_rel_reduction_pct": round(100.0 * (b_fp - s_fp) / max(1, b_fp), 2),
            "fn_cavity_px_delta": s_fn - b_fn,
            "fn_cavity_rel_change_pct": round(100.0 * (s_fn - b_fn) / max(1, b_fn), 2),
            "seg_beats_box_iou_count": int(sum(1 for r in per_img_pairs if r["seg_iou"] > r["box_iou"])),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4A.3 Grouped Test & Box-vs-Mask Evaluation")
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN)
    parser.add_argument("--primary-conf", type=float, default=0.50)
    parser.add_argument("--iou-nms", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(TEST_VIZ_DIR, exist_ok=True)
    os.makedirs(BVM_VIZ_DIR, exist_ok=True)

    best_pt = os.path.join(args.run_dir, "weights", "best.pt")
    assert sha256_file(best_pt) == EXPECTED_BEST_SHA256, "best.pt SHA256 mismatch!"
    assert sha256_file(MANIFEST_JSON) == EXPECTED_MANIFEST_SHA256, "dataset_manifest.json SHA256 mismatch!"
    assert sha256_file(SPLIT_CSV) == EXPECTED_SPLIT_CSV_SHA256, "split_manifest.csv SHA256 mismatch!"
    assert sha256_file(DATA_YAML) == EXPECTED_YAML_SHA256, "data.yaml SHA256 mismatch!"

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(best_pt)
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    params = sum(p.numel() for p in ckpt["model"].parameters())

    # 1. Ultralytics standard AP evaluation on grouped test split
    ul_eval_dir = os.path.join(BENCH_DIR, "pothole440_seg_ultralytics_test")
    res_test = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=args.imgsz,
        batch=4,
        iou=args.iou_nms,
        device=device,
        project=ul_eval_dir,
        name="test_eval",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    box_p, box_r = float(res_test.box.mp), float(res_test.box.mr)
    box_f1 = (2 * box_p * box_r / (box_p + box_r)) if (box_p + box_r) > 0 else 0.0
    msk_p, msk_r = float(res_test.seg.mp), float(res_test.seg.mr)
    msk_f1 = (2 * msk_p * msk_r / (msk_p + msk_r)) if (msk_p + msk_r) > 0 else 0.0

    ul_metrics = {
        "box": {
            "precision": round(box_p, 4),
            "recall": round(box_r, 4),
            "f1": round(box_f1, 4),
            "mAP50": round(float(res_test.box.map50), 4),
            "mAP50_95": round(float(res_test.box.map), 4),
        },
        "mask": {
            "precision": round(msk_p, 4),
            "recall": round(msk_r, 4),
            "f1": round(msk_f1, 4),
            "mAP50": round(float(res_test.seg.map50), 4),
            "mAP50_95": round(float(res_test.seg.map), 4),
        },
        "ultralytics_test_speed_ms": {k: round(float(v), 3) for k, v in res_test.speed.items()},
    }

    # 2. Load 15 grouped test records & original PNG masks
    test_records = []
    with open(SPLIT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["disposition"] == "INCLUDED_TEST":
                fn = row["filename"]
                orig_sp = row["orig_pothole440_split"]
                rgb_p = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "images", "test", fn)
                gt_mask_p = os.path.join(SRC_ROOT, orig_sp, "mask", fn)
                test_records.append({
                    "filename": fn,
                    "base_id": row["base_id"],
                    "session": row["session"],
                    "orig_pothole440_split": orig_sp,
                    "rgb_path": rgb_p,
                    "gt_mask_path": gt_mask_p,
                })
    assert len(test_records) == 15, f"Expected 15 test images, got {len(test_records)}"

    def run_conf_pass(conf_th: float, render_viz: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        seg_per_img = []
        bvm_per_img = []
        for idx, rec in enumerate(test_records, 1):
            img_bgr = cv2.imread(rec["rgb_path"])
            gt_raw = cv2.imread(rec["gt_mask_path"], cv2.IMREAD_UNCHANGED)
            if gt_raw.ndim == 3:
                gt_raw = gt_raw[:, :, 0]
            gt = gt_raw > 0

            seg_mask, box_mask, n_inst, confs = predict_seg_and_box_masks(
                model, img_bgr, args.imgsz, conf_th, args.iou_nms, device
            )
            sm = compute_mask_metrics(gt, seg_mask, include_multiscale_boundary=True)
            bm = compute_mask_metrics(gt, box_mask, include_multiscale_boundary=False)

            sm_row = {
                "test_idx": idx,
                "filename": rec["filename"],
                "base_id": rec["base_id"],
                "session": rec["session"],
                "orig_pothole440_split": rec["orig_pothole440_split"],
                "pred_instances": n_inst,
                "pred_confs": confs,
                **sm,
                "box_iou": bm["iou"],
                "box_dice": bm["dice"],
                "box_fp_px": bm["fp_px"],
                "box_fn_px": bm["fn_px"],
            }
            seg_per_img.append(sm_row)

            bvm_row = {
                "test_idx": idx,
                "filename": rec["filename"],
                "pred_instances": n_inst,
                "gt_fg_pct": sm["gt_fg_pct"],
                "box_fg_pct": bm["pred_fg_pct"],
                "seg_fg_pct": sm["pred_fg_pct"],
                "box_iou": bm["iou"],
                "seg_iou": sm["iou"],
                "iou_gain_abs": round(sm["iou"] - bm["iou"], 4),
                "box_dice": bm["dice"],
                "seg_dice": sm["dice"],
                "dice_gain_abs": round(sm["dice"] - bm["dice"], 4),
                "box_tp_px": bm["tp_px"],
                "box_fp_px": bm["fp_px"],
                "box_fn_px": bm["fn_px"],
                "box_tn_px": bm["tn_px"],
                "seg_tp_px": sm["tp_px"],
                "seg_fp_px": sm["fp_px"],
                "seg_fn_px": sm["fn_px"],
                "seg_tn_px": sm["tn_px"],
            }
            bvm_per_img.append(bvm_row)

            if render_viz:
                render_test_visualization(idx, rec["filename"], img_bgr, gt, seg_mask, sm_row)
                render_box_vs_mask_viz(idx, rec["filename"], img_bgr, gt, box_mask, seg_mask, bm, sm)

        seg_sum = summarize_seg_suite(seg_per_img, conf_th, args.iou_nms)
        bvm_sum = summarize_box_vs_mask(bvm_per_img, conf_th)
        return seg_sum, seg_per_img, bvm_sum, bvm_per_img

    primary_seg_sum, primary_seg_rows, primary_bvm_sum, primary_bvm_rows = run_conf_pass(args.primary_conf, render_viz=True)
    ref025_seg_sum, _, ref025_bvm_sum, _ = run_conf_pass(0.25, render_viz=False)

    # Save CSV
    csv_out = os.path.join(BENCH_DIR, "pothole440_segmentation_test_per_image.csv")
    csv_cols = [
        "test_idx", "filename", "base_id", "session", "orig_pothole440_split", "pred_instances",
        "gt_fg_pct", "pred_fg_pct", "iou", "dice", "pixel_precision", "pixel_recall",
        "boundary_iou_2px", "boundary_f1_2px", "boundary_iou_5px", "boundary_f1_5px",
        "boundary_iou_10px", "boundary_f1_10px",
        "gt_components", "missed_gt_regions", "pred_components", "fp_pred_regions",
        "tp_px", "fp_px", "fn_px", "box_iou", "box_dice", "box_fp_px", "box_fn_px",
    ]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in primary_seg_rows:
            w.writerow({k: r[k] for k in csv_cols})

    # Save primary test JSON
    test_json = {
        "experiment": "Phase 4A.3 Independent Grouped Test Evaluation (Reconstructed Grouped Benchmark)",
        "weights_path": best_pt,
        "weights_sha256": EXPECTED_BEST_SHA256,
        "parameters": params,
        "frozen_manifest_sha256": {
            "dataset_manifest.json": EXPECTED_MANIFEST_SHA256,
            "split_manifest.csv": EXPECTED_SPLIT_CSV_SHA256,
            "data.yaml": EXPECTED_YAML_SHA256,
        },
        "protocol_reference": "reports/pothole440_segmentation_evaluation_protocol.md",
        "ultralytics_test_metrics": ul_metrics,
        "region_level_primary_conf050": primary_seg_sum,
        "region_level_reference_conf025": ref025_seg_sum,
        "per_image_test_results_conf050": primary_seg_rows,
    }
    test_json_path = os.path.join(BENCH_DIR, "pothole440_segmentation_test.json")
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f, indent=2)

    # Save Box vs Mask JSON
    bvm_json = {
        "experiment": "Phase 4A.3 Box-versus-Mask Representation Comparison (Matched YOLOv8n-seg Predictions)",
        "weights_path": best_pt,
        "weights_sha256": EXPECTED_BEST_SHA256,
        "split": "dataset_pothole440_seg_grouped/test (15 unaugmented originals)",
        "primary_conf_050": primary_bvm_sum,
        "reference_conf_025": ref025_bvm_sum,
        "per_image_comparison_conf050": primary_bvm_rows,
    }
    bvm_json_path = os.path.join(BENCH_DIR, "pothole440_box_vs_mask.json")
    with open(bvm_json_path, "w", encoding="utf-8") as f:
        json.dump(bvm_json, f, indent=2)

    print("=== TEST EVALUATION COMPLETE ===")
    print("Ultralytics AP:", json.dumps(ul_metrics))
    print("Region Primary conf=0.50:", json.dumps({k: primary_seg_sum[k] for k in (
        "empty_prediction_images", "macro_mean_iou", "macro_median_iou", "macro_min_iou",
        "macro_mean_dice", "micro_pooled_iou", "micro_pooled_dice",
        "completely_missed_gt_regions", "false_positive_mask_regions"
    )}))
    print("Box vs Mask conf=0.50:", json.dumps(primary_bvm_sum["delta_seg_minus_box"]))


if __name__ == "__main__":
    main()
