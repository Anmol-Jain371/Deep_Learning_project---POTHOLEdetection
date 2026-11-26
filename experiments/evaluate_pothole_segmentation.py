"""
========================================================================================
RoadVision AI -- Phase 4A.2: Validation-Only Segmentation & Region Benchmark
========================================================================================
Evaluates the validation-selected YOLOv8n-seg best.pt checkpoint strictly on the
`val` split (15 unaugmented originals) — NEVER touches the reconstructed grouped `test` split.

Computes:
  1. Official Ultralytics `val` box & mask metrics (Precision, Recall, F1, mAP@50, mAP@50-95).
  2. Region-level pixel evaluation against original Pothole440 binary PNG masks:
       - Foreground IoU, Dice, Pixel Precision, Pixel Recall
       - Boundary IoU & Boundary F1 (2-px tolerance around 1-px morphological boundary)
       - Completely missed GT regions & False-Positive predicted regions
       - Sensitivity sweep across confidence thresholds (0.15, 0.25, 0.50)
       - Explicit empty-mask handling for images with 0 detections
  3. Validation-only 8-panel visualizations (RGB, GT, Pred, Overlay, Error Map, TP, FP, FN)
     for all 15 validation images.
  4. RTX 2050 CUDA inference latency benchmark (mean, median, p95, FPS, peak VRAM)
     explicitly separating YOLOv8n-seg segmentation from full RoadVision pipeline speed.
========================================================================================
"""

import os
import csv
import json
import time
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
SPLIT_CSV = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "manifests", "split_manifest.csv")
SRC_ROOT = os.path.join(BASE_DIR, "Pothole440-main", "Pothole440-main")
DEFAULT_RUN = os.path.join(BASE_DIR, "runs", "segment", "pothole440_yolov8n_seg_v1")
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "pothole440_seg_val_viz")
os.makedirs(BENCH_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


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


def compute_boundary_metrics(gt: np.ndarray, pred: np.ndarray, tol_px: int = 2) -> Tuple[float, float]:
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
    # Dilated boundary IoU
    b_iou = int(np.logical_and(gb_dil, pb_dil).sum()) / float(int(np.logical_or(gb_dil, pb_dil).sum()))
    return b_iou, b_f1


def predict_foreground_mask(model: YOLO, img_bgr: np.ndarray, imgsz: int, conf: float, iou_nms: float, device: Any) -> Tuple[np.ndarray, int]:
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
    if not res or res[0].masks is None or res[0].masks.data is None or len(res[0].masks.data) == 0:
        return np.zeros((H, W), dtype=bool), 0

    m_tensor = res[0].masks.data  # [N, H_m, W_m] in [0, 1] float
    m_np = m_tensor.cpu().numpy()
    if m_np.shape[1] != H or m_np.shape[2] != W:
        resized = [cv2.resize(m_np[i], (W, H), interpolation=cv2.INTER_LINEAR) >= 0.5 for i in range(m_np.shape[0])]
        fg_pred = np.any(np.stack(resized, axis=0), axis=0)
    else:
        fg_pred = np.any(m_np >= 0.5, axis=0)
    return fg_pred, int(m_np.shape[0])


def region_metrics_for_image(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())

    union = tp + fp + fn
    iou = (tp / float(union)) if union > 0 else 1.0
    dice = (2.0 * tp / float(2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    prec = (tp / float(tp + fp)) if (tp + fp) > 0 else (1.0 if (tp + fn) == 0 else 0.0)
    rec = (tp / float(tp + fn)) if (tp + fn) > 0 else 1.0
    b_iou, b_f1 = compute_boundary_metrics(gt, pred, tol_px=2)

    # Region-level missed GT & false-positive components
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

    return {
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
        "boundary_iou_2px": b_iou,
        "boundary_f1_2px": b_f1,
        "gt_components": n_gt_comp,
        "missed_gt_regions": missed_gt_comp,
        "pred_components": n_pr_comp,
        "fp_pred_regions": fp_pr_comp,
    }


def render_val_visualization(idx: int, fn: str, rgb_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray, m: Dict[str, Any]):
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    tp_map = np.logical_and(gt, pred)
    fp_map = np.logical_and(~gt, pred)
    fn_map = np.logical_and(gt, ~pred)

    # Composite Error Map: TP=Green, FP=Red, FN=Blue on dimmed RGB
    err_vis = (rgb * 0.35).astype(np.uint8)
    err_vis[tp_map] = [34, 197, 94]
    err_vis[fp_map] = [239, 68, 68]
    err_vis[fn_map] = [59, 130, 246]

    # RGB + Overlay (amber pred fill + cyan GT contour)
    ov = rgb.copy()
    tint = np.zeros_like(rgb)
    tint[pred] = [245, 158, 11]
    ov = cv2.addWeighted(ov, 0.70, tint, 0.30, 0)
    gt_cnt, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, gt_cnt, -1, (0, 255, 255), 3)

    # Isolated TP / FP / FN panels
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
        f"Val #{idx:02d}: {fn} | IoU={m['iou']:.4f}, Dice={m['dice']:.4f}, Prec={m['pixel_precision']:.4f}, Rec={m['pixel_recall']:.4f}, B-F1={m['boundary_f1_2px']:.4f}",
        fontsize=11.5, fontweight="bold"
    )
    out_p = os.path.join(VIZ_DIR, f"val_{idx:02d}_{os.path.splitext(fn)[0]}.png")
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Phase 4A.2 Validation-Only Evaluation")
    parser.add_argument("--run-dir", type=str, default=DEFAULT_RUN)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-nms", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    best_pt = os.path.join(args.run_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"Missing best.pt: {best_pt}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(best_pt)
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    params = sum(p.numel() for p in ckpt["model"].parameters())

    # 1. Official Ultralytics val metrics (strictly on val split)
    ul_eval_dir = os.path.join(BENCH_DIR, "pothole440_seg_ultralytics_val")
    res_val = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=args.imgsz,
        batch=4,
        iou=args.iou_nms,
        device=device,
        project=ul_eval_dir,
        name="val_eval",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    box_p, box_r = float(res_val.box.mp), float(res_val.box.mr)
    box_f1 = (2 * box_p * box_r / (box_p + box_r)) if (box_p + box_r) > 0 else 0.0
    msk_p, msk_r = float(res_val.seg.mp), float(res_val.seg.mr)
    msk_f1 = (2 * msk_p * msk_r / (msk_p + msk_r)) if (msk_p + msk_r) > 0 else 0.0

    ul_metrics = {
        "box": {
            "precision": round(box_p, 4),
            "recall": round(box_r, 4),
            "f1": round(box_f1, 4),
            "mAP50": round(float(res_val.box.map50), 4),
            "mAP50_95": round(float(res_val.box.map), 4),
        },
        "mask": {
            "precision": round(msk_p, 4),
            "recall": round(msk_r, 4),
            "f1": round(msk_f1, 4),
            "mAP50": round(float(res_val.seg.map50), 4),
            "mAP50_95": round(float(res_val.seg.map), 4),
        },
        "ultralytics_val_speed_ms": {k: round(float(v), 3) for k, v in res_val.speed.items()},
    }

    # 2. Load validation ground-truth paths from original Pothole440-main/mask/ via split_manifest.csv
    val_gt_records = []
    with open(SPLIT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["disposition"] == "INCLUDED_VAL":
                fn = row["filename"]
                orig_sp = row["orig_pothole440_split"]
                rgb_p = os.path.join(BASE_DIR, "dataset_pothole440_seg_grouped", "images", "val", fn)
                gt_mask_p = os.path.join(SRC_ROOT, orig_sp, "mask", fn)
                val_gt_records.append({"filename": fn, "base_id": row["base_id"], "session": row["session"], "rgb_path": rgb_p, "gt_mask_path": gt_mask_p})

    assert len(val_gt_records) == 15, f"Expected 15 validation images, got {len(val_gt_records)}"

    # 3. Evaluate region-level metrics at primary conf=0.25 + sensitivity sweep (0.15, 0.50)
    def eval_at_conf(conf_th: float, save_viz: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        per_img = []
        viz_paths = []
        tot_tp = tot_fp = tot_fn = 0
        tot_missed_gt = tot_fp_reg = tot_gt_comp = 0

        for idx, rec in enumerate(val_gt_records, 1):
            img_bgr = cv2.imread(rec["rgb_path"])
            gt_raw = cv2.imread(rec["gt_mask_path"], cv2.IMREAD_UNCHANGED)
            if gt_raw.ndim == 3:
                gt_raw = gt_raw[:, :, 0]
            gt = gt_raw > 0

            pred, n_inst = predict_foreground_mask(model, img_bgr, args.imgsz, conf_th, args.iou_nms, device)
            rm = region_metrics_for_image(gt, pred)
            rm["filename"] = rec["filename"]
            rm["base_id"] = rec["base_id"]
            rm["session"] = rec["session"]
            rm["pred_instances"] = n_inst
            per_img.append(rm)

            tot_tp += rm["tp_px"]
            tot_fp += rm["fp_px"]
            tot_fn += rm["fn_px"]
            tot_missed_gt += rm["missed_gt_regions"]
            tot_fp_reg += rm["fp_pred_regions"]
            tot_gt_comp += rm["gt_components"]

            if save_viz:
                vp = render_val_visualization(idx, rec["filename"], img_bgr, gt, pred, rm)
                viz_paths.append(vp)

        ious = np.array([r["iou"] for r in per_img])
        dices = np.array([r["dice"] for r in per_img])
        precs = np.array([r["pixel_precision"] for r in per_img])
        recs = np.array([r["pixel_recall"] for r in per_img])
        b_ious = np.array([r["boundary_iou_2px"] for r in per_img])
        b_f1s = np.array([r["boundary_f1_2px"] for r in per_img])

        micro_iou = tot_tp / float(max(1, tot_tp + tot_fp + tot_fn))
        micro_dice = (2.0 * tot_tp) / float(max(1, 2 * tot_tp + tot_fp + tot_fn))
        micro_prec = tot_tp / float(max(1, tot_tp + tot_fp))
        micro_rec = tot_tp / float(max(1, tot_tp + tot_fn))

        summary = {
            "conf_threshold": conf_th,
            "iou_nms_threshold": args.iou_nms,
            "val_images_count": len(per_img),
            "empty_prediction_images": sum(1 for r in per_img if r["pred_instances"] == 0),
            "macro_mean_iou": round(float(np.mean(ious)), 4),
            "macro_std_iou": round(float(np.std(ious, ddof=1)), 4),
            "macro_median_iou": round(float(np.median(ious)), 4),
            "macro_min_iou": round(float(np.min(ious)), 4),
            "macro_max_iou": round(float(np.max(ious)), 4),
            "macro_mean_dice": round(float(np.mean(dices)), 4),
            "macro_std_dice": round(float(np.std(dices, ddof=1)), 4),
            "macro_median_dice": round(float(np.median(dices)), 4),
            "macro_min_dice": round(float(np.min(dices)), 4),
            "macro_mean_pixel_precision": round(float(np.mean(precs)), 4),
            "macro_mean_pixel_recall": round(float(np.mean(recs)), 4),
            "macro_mean_boundary_iou_2px": round(float(np.mean(b_ious)), 4),
            "macro_mean_boundary_f1_2px": round(float(np.mean(b_f1s)), 4),
            "micro_pooled_iou": round(float(micro_iou), 4),
            "micro_pooled_dice": round(float(micro_dice), 4),
            "micro_pooled_pixel_precision": round(float(micro_prec), 4),
            "micro_pooled_pixel_recall": round(float(micro_rec), 4),
            "total_gt_regions": tot_gt_comp,
            "completely_missed_gt_regions": tot_missed_gt,
            "false_positive_mask_regions": tot_fp_reg,
            "viz_paths": viz_paths,
        }
        return summary, per_img

    primary_summary, primary_per_img = eval_at_conf(args.conf, save_viz=True)
    sens_015, _ = eval_at_conf(0.15, save_viz=False)
    sens_050, _ = eval_at_conf(0.50, save_viz=False)

    # Save per-image CSV cleanly
    csv_out = os.path.join(BENCH_DIR, "pothole440_segmentation_per_image.csv")
    csv_cols = [
        "filename", "base_id", "session", "pred_instances",
        "gt_fg_pct", "pred_fg_pct", "iou", "dice",
        "pixel_precision", "pixel_recall", "boundary_iou_2px", "boundary_f1_2px",
        "gt_components", "missed_gt_regions", "pred_components", "fp_pred_regions",
        "tp_px", "fp_px", "fn_px",
    ]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in primary_per_img:
            w.writerow({k: r[k] for k in csv_cols})

    # 4. Performance Latency Benchmark on RTX 2050 (15 val images x 10 loops = 150 passes)
    imgs_cached = [cv2.imread(rec["rgb_path"]) for rec in val_gt_records]
    for im in imgs_cached[:10]:
        _ = predict_foreground_mask(model, im, args.imgsz, args.conf, args.iou_nms, device)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)

    e2e_times_ms = []
    model_only_times_ms = []
    for _ in range(10):
        for im in imgs_cached:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = model.predict(source=im, imgsz=args.imgsz, conf=args.conf, iou=args.iou_nms, retina_masks=True, device=device, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            # Include binary mask union reconstruction timing
            H, W = im.shape[:2]
            if res and res[0].masks is not None and res[0].masks.data is not None and len(res[0].masks.data) > 0:
                m_np = res[0].masks.data.cpu().numpy()
                _ = np.any(m_np >= 0.5, axis=0)
            else:
                _ = np.zeros((H, W), dtype=bool)
            t2 = time.perf_counter()
            model_only_times_ms.append((t1 - t0) * 1000.0)
            e2e_times_ms.append((t2 - t0) * 1000.0)

    peak_inf_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    peak_inf_res_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    def lat_stats(arr_ms):
        a = np.array(arr_ms)
        mean_ms = float(np.mean(a))
        return {
            "mean_ms": round(mean_ms, 2),
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "fps": round(1000.0 / max(1e-6, mean_ms), 2),
        }

    # Load training telemetry
    tele_p = os.path.join(args.run_dir, "phase4a2_train_telemetry.json")
    train_tele = json.load(open(tele_p, "r")) if os.path.exists(tele_p) else {}

    out_json_data = {
        "experiment": "Phase 4A.2 YOLOv8n-seg Validation Evaluation (Test Split Untouched)",
        "weights_path": best_pt,
        "weights_sha256": sha256_file(best_pt),
        "weights_size_mb": round(os.path.getsize(best_pt) / (1024 ** 2), 2),
        "parameters": params,
        "parameters_m": round(params / 1e6, 3),
        "training_telemetry": train_tele,
        "evaluation_protocol": {
            "split_evaluated": "val (15 unaugmented originals; test untouched)",
            "ground_truth_source": "Original Pothole440 binary PNG masks (1200x800)",
            "imgsz": args.imgsz,
            "primary_conf_threshold": args.conf,
            "iou_nms_threshold": args.iou_nms,
            "retina_masks": True,
            "empty_predictions_included_as_zero_mask": True,
        },
        "ultralytics_val_metrics": ul_metrics,
        "region_level_metrics_primary_conf025": {k: v for k, v in primary_summary.items() if k != "viz_paths"},
        "region_level_sensitivity": {
            "conf_0.15": {k: v for k, v in sens_015.items() if k != "viz_paths"},
            "conf_0.50": {k: v for k, v in sens_050.items() if k != "viz_paths"},
        },
        "per_image_val_results": primary_per_img,
        "latency_benchmark_rtx2050": {
            "samples_timed": len(e2e_times_ms),
            "includes_preprocess_forward_nms_mask_upsample": True,
            "predict_only": lat_stats(model_only_times_ms),
            "predict_plus_binary_union_reconstruction": lat_stats(e2e_times_ms),
            "peak_inference_gpu_allocated_mb": round(peak_inf_alloc_mb, 2),
            "peak_inference_gpu_reserved_mb": round(peak_inf_res_mb, 2),
            "note": "Standalone YOLOv8n-seg 640p segmentation only; excludes ByteTrack, MiDaS depth, and planar road calibration.",
        },
    }

    json_out = os.path.join(BENCH_DIR, "pothole440_segmentation_validation.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(out_json_data, f, indent=2)

    print("=== PHASE 4A.2 VALIDATION SUMMARY ===")
    print("Ultralytics Val:", json.dumps(ul_metrics))
    print("Region Level (conf=0.25):", json.dumps({k: v for k, v in primary_summary.items() if k != "viz_paths"}, indent=2))
    print("Latency RTX 2050:", json.dumps(out_json_data["latency_benchmark_rtx2050"], indent=2))
    print(f"Saved: {json_out}")
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
