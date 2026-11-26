"""
========================================================================================
RoadVision AI -- Phase 4A.3: Exploratory Dashcam Domain-Transfer Evaluation
========================================================================================
Compares:
  - Phase 2 controlled YOLOv8n detector (runs/detect/pothole_unified_yolov8n_100e_sgd/weights/best.pt)
  - Phase 4A.2 YOLOv8n-seg model (runs/segment/pothole440_yolov8n_seg_v1/weights/best.pt)

STRICT SCIENTIFIC RULES:
  - Exploratory domain-transfer analysis only.
  - Reports bounding-box detection metrics ONLY on datasets with reviewed bounding-box
    labels (Rangoon dataset_v2_candidates 35 frames & dataset_phone_potholes val 14 frames),
    with explicit annotation-quality caveats.
  - NEVER computes segmentation IoU/Dice without genuine pixel segmentation ground truth.
  - NEVER treats empty labels in evaluation/ground_truth_labels/ (40 PENDING_MANUAL_LABEL
    files) as verified negative ground truth.
  - Generates side-by-side 3-panel visualizations for 16 reproducible frames across
    Rangoon Road, phone footage, and rural dashcam footage.
========================================================================================
"""

import os
import glob
import json
from typing import List, Dict, Any, Tuple

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DET_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "pothole_unified_yolov8n_100e_sgd", "weights", "best.pt")
SEG_WEIGHTS = os.path.join(BASE_DIR, "runs", "segment", "pothole440_yolov8n_seg_v1", "weights", "best.pt")
OUT_DIR = os.path.join(BASE_DIR, "runs", "benchmark", "pothole440_dashcam_transfer")


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=float)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    a1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    a2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_yolo_bbox_gt(lbl_path: str, w: int, h: int) -> np.ndarray:
    boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, xc, yc, bw, bh = map(float, parts[:5])
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                    boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=float) if boxes else np.zeros((0, 4), dtype=float)


def eval_bbox_dataset(
    model: YOLO, image_paths: List[str], label_dir: str, device: Any, imgsz: int = 640, iou_nms: float = 0.70
) -> Dict[str, Any]:
    """Computes identical box-only P/R/F1 at conf=0.25 & conf=0.50, plus all-point AP@50 (conf>=0.001)."""
    gt_by_img = {}
    all_preds = []  # (conf, img_idx, xyxy)
    total_gt = 0

    for idx, ip in enumerate(image_paths):
        im = cv2.imread(ip)
        h, w = im.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]
        gt = load_yolo_bbox_gt(os.path.join(label_dir, f"{stem}.txt"), w, h)
        gt_by_img[idx] = gt
        total_gt += len(gt)

        res = model.predict(
            source=im, imgsz=imgsz, conf=0.001, iou=iou_nms, max_det=300, device=device, verbose=False
        )[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs):
                all_preds.append((float(c), idx, b))

    def pr_at_conf(conf_th: float) -> Dict[str, Any]:
        preds = [p for p in all_preds if p[0] >= conf_th]
        preds.sort(key=lambda x: -x[0])
        matched = {idx: np.zeros(len(gt_by_img[idx]), dtype=bool) for idx in gt_by_img}
        tp = fp = 0
        for _, idx, box in preds:
            gt = gt_by_img[idx]
            if len(gt) == 0:
                fp += 1
                continue
            ious = compute_iou_matrix(box[None, :], gt)[0]
            best_j = int(np.argmax(ious))
            if ious[best_j] >= 0.50 and not matched[idx][best_j]:
                matched[idx][best_j] = True
                tp += 1
            else:
                fp += 1
        fn = total_gt - tp
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(total_gt) if total_gt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    # All-point interpolated AP@50 (Ultralytics/VOC2010+ style)
    preds_sorted = sorted(all_preds, key=lambda x: -x[0])
    matched = {idx: np.zeros(len(gt_by_img[idx]), dtype=bool) for idx in gt_by_img}
    tp_arr = np.zeros(len(preds_sorted), dtype=float)
    fp_arr = np.zeros(len(preds_sorted), dtype=float)
    for k, (_, idx, box) in enumerate(preds_sorted):
        gt = gt_by_img[idx]
        if len(gt) == 0:
            fp_arr[k] = 1.0
            continue
        ious = compute_iou_matrix(box[None, :], gt)[0]
        best_j = int(np.argmax(ious))
        if ious[best_j] >= 0.50 and not matched[idx][best_j]:
            matched[idx][best_j] = True
            tp_arr[k] = 1.0
        else:
            fp_arr[k] = 1.0
    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / np.maximum(1e-9, tp_cum + fp_cum)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap50 = float(np.trapezoid(np.interp(x, mrec, mpre), x)) if hasattr(np, "trapezoid") else float(np.trapz(np.interp(x, mrec, mpre), x))

    return {
        "images": len(image_paths),
        "total_gt_boxes": total_gt,
        "box_ap50_conf001": round(ap50, 4),
        "op_conf_025": pr_at_conf(0.25),
        "op_conf_050": pr_at_conf(0.50),
    }


def extract_video_frame(video_path: str, frame_num: int) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed reading frame {frame_num} from {video_path}")
    return frame, round(frame_num / float(fps), 2)


def draw_det_panel(img_bgr: np.ndarray, res0) -> Tuple[np.ndarray, int, List[float]]:
    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = vis.shape[:2]
    n_det = 0
    confs = []
    if res0.boxes is not None and len(res0.boxes) > 0:
        boxes = res0.boxes.xyxy.cpu().numpy()
        confs_np = res0.boxes.conf.cpu().numpy()
        n_det = len(boxes)
        confs = [round(float(c), 3) for c in confs_np]
        for (x1, y1, x2, y2), sc in zip(boxes, confs_np):
            p1 = (max(0, int(x1)), max(0, int(y1)))
            p2 = (min(w - 1, int(x2)), min(h - 1, int(y2)))
            cv2.rectangle(vis, p1, p2, (34, 197, 94), 2)
            cv2.putText(vis, f"{sc:.2f}", (p1[0], max(14, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 2, cv2.LINE_AA)
    return vis, n_det, confs


def draw_seg_panel(img_bgr: np.ndarray, res0) -> Tuple[np.ndarray, int, List[float]]:
    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = vis.shape[:2]
    n_det = 0
    confs = []
    if res0.boxes is not None and len(res0.boxes) > 0:
        boxes = res0.boxes.xyxy.cpu().numpy()
        confs_np = res0.boxes.conf.cpu().numpy()
        n_det = len(boxes)
        confs = [round(float(c), 3) for c in confs_np]
        if res0.masks is not None and res0.masks.data is not None and len(res0.masks.data) > 0:
            m_np = res0.masks.data.cpu().numpy()
            union_mask = np.zeros((h, w), dtype=bool)
            for i in range(m_np.shape[0]):
                mi = m_np[i]
                if mi.shape[0] != h or mi.shape[1] != w:
                    mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_LINEAR)
                union_mask |= (mi >= 0.5)
            tint = vis.copy()
            tint[union_mask] = [245, 158, 11]
            vis = cv2.addWeighted(vis, 0.60, tint, 0.40, 0)
            cnts, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 220, 0), 2)

        for (x1, y1, x2, y2), sc in zip(boxes, confs_np):
            p1 = (max(0, int(x1)), max(0, int(y1)))
            p2 = (min(w - 1, int(x2)), min(h - 1, int(y2)))
            cv2.rectangle(vis, p1, p2, (56, 189, 248), 2)
            cv2.putText(vis, f"{sc:.2f}", (p1[0], max(14, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 2, cv2.LINE_AA)
    return vis, n_det, confs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    det_model = YOLO(DET_WEIGHTS)
    seg_model = YOLO(SEG_WEIGHTS)

    # 1. Reproducible detection metrics on reviewed bounding-box sets (with explicit caveats)
    rangoon_imgs = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_v2_candidates", "*.jpg")))
    phone_imgs = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_phone_potholes", "images", "val", "*.jpg")))

    box_benchmarks = {
        "note": "Box-only evaluation on existing bounding-box ground truth; empty evaluation/ground_truth_labels (40 PENDING_MANUAL_LABEL) excluded and NOT treated as negative GT; NO segmentation IoU/Dice computed.",
        "rangoon_35": {
            "source": "dataset_v2_candidates/ (35 reviewed frames from Watch Video_ Terrible potholes in Rangoon Road.mp4, 76 boxes)",
            "caveat": "Previously inspected exploratory external set; non-consecutive temporal frames from single video.",
            "yolov8n_phase2_detector": eval_bbox_dataset(det_model, rangoon_imgs, os.path.join(BASE_DIR, "dataset_v2_candidates", "labels"), device),
            "yolov8n_seg_phase4a2": eval_bbox_dataset(seg_model, rangoon_imgs, os.path.join(BASE_DIR, "dataset_v2_candidates", "labels"), device),
        },
        "phone_val_14": {
            "source": "dataset_phone_potholes/images/val (14 frames, 52 boxes)",
            "caveat": "Part of dataset_unified training/val provenance (YOLOv8n Phase 2 had phone images in unified pool); YOLOv8n-seg had ZERO phone/dashcam training exposure.",
            "yolov8n_phase2_detector": eval_bbox_dataset(det_model, phone_imgs, os.path.join(BASE_DIR, "dataset_phone_potholes", "labels", "val"), device),
            "yolov8n_seg_phase4a2": eval_bbox_dataset(seg_model, phone_imgs, os.path.join(BASE_DIR, "dataset_phone_potholes", "labels", "val"), device),
        },
    }

    # 2. Reproducible 16-frame visual comparison across all available dashcam & phone sources
    samples = [
        # Rangoon Road video (640x360, 25 fps) — also matches dataset_v2_candidates exact frame indices
        {"id": "dt_01_rangoon_f0000", "source": "Watch Video_ Terrible potholes in Rangoon Road.mp4", "type": "video", "frame": 0, "scenario": "Near-field cavity + approach"},
        {"id": "dt_02_rangoon_f0040", "source": "Watch Video_ Terrible potholes in Rangoon Road.mp4", "type": "video", "frame": 40, "scenario": "Shadowed / low-contrast potholes (det wins, seg misses)"},
        {"id": "dt_03_rangoon_f0211", "source": "Watch Video_ Terrible potholes in Rangoon Road.mp4", "type": "video", "frame": 211, "scenario": "Water/mud-filled cavity boundary test"},
        {"id": "dt_04_rangoon_f0255", "source": "Watch Video_ Terrible potholes in Rangoon Road.mp4", "type": "video", "frame": 255, "scenario": "Multiple water-filled potholes cluster"},
        {"id": "dt_05_rangoon_f0715", "source": "Watch Video_ Terrible potholes in Rangoon Road.mp4", "type": "video", "frame": 715, "scenario": "Forward-corridor crater (seg fires 0.92, det misses)"},
        {"id": "dt_06_rangoon_f0803", "source": "Watch Video_ Terrible potholes in Rangoon Road.mp4", "type": "video", "frame": 803, "scenario": "Shadowed deep craters (det wins, seg misses)"},
        # Phone validation frames (1280x720)
        {"id": "dt_07_phone_f0005", "source": "dataset_phone_potholes/images/val/frame_0005.jpg", "type": "image", "frame": 5, "scenario": "Phone multi-pothole oblique street view (8 GT boxes)"},
        {"id": "dt_08_phone_f0028", "source": "dataset_phone_potholes/images/val/frame_0028.jpg", "type": "image", "frame": 28, "scenario": "Phone oblique potholes (det=2, seg=1 partial)"},
        {"id": "dt_09_phone_f0035", "source": "dataset_phone_potholes/images/val/frame_0035.jpg", "type": "image", "frame": 35, "scenario": "Phone oblique crack/cavity (det=3, seg=1)"},
        # Phone survey video (raw_survey.mp4, 1280x720, 10 fps)
        {"id": "dt_10_survey_f0040", "source": "roadvision-dashboard/public/videos/raw_survey.mp4", "type": "video", "frame": 40, "scenario": "Survey clip t=4.0s (seg=1 high conf, det=0 — check FP/TP)"},
        {"id": "dt_11_survey_f0080", "source": "roadvision-dashboard/public/videos/raw_survey.mp4", "type": "video", "frame": 80, "scenario": "Survey clip t=8.0s (both detect foreground feature)"},
        {"id": "dt_12_survey_f0120", "source": "roadvision-dashboard/public/videos/raw_survey.mp4", "type": "video", "frame": 120, "scenario": "Survey clip t=12.0s (multiple oblique potholes, seg misses)"},
        # Rural dashcam video (uploads/upload_1794847200_ruralRoad_potHoles.mp4, 640x360, 25 fps)
        {"id": "dt_13_rural_f0060", "source": "uploads/upload_1794847200_ruralRoad_potHoles.mp4", "type": "video", "frame": 60, "scenario": "Rural dashcam mid-range potholes (both detect)"},
        {"id": "dt_14_rural_f0075", "source": "uploads/upload_1794847200_ruralRoad_potHoles.mp4", "type": "video", "frame": 75, "scenario": "Rural dashcam close cavities (both detect)"},
        {"id": "dt_15_rural_f0105", "source": "uploads/upload_1794847200_ruralRoad_potHoles.mp4", "type": "video", "frame": 105, "scenario": "Rural dashcam small/distant potholes (det=5, seg=0)"},
        {"id": "dt_16_rural_f0150", "source": "uploads/upload_1794847200_ruralRoad_potHoles.mp4", "type": "video", "frame": 150, "scenario": "Rural dashcam sunlit multi-pothole cluster (det=5, seg=1 weak)"},
    ]

    sample_records = []
    for s in samples:
        src_full = os.path.join(BASE_DIR, s["source"])
        if s["type"] == "video":
            img_bgr, ts = extract_video_frame(src_full, s["frame"])
        else:
            img_bgr = cv2.imread(src_full)
            ts = None

        rd = det_model.predict(source=img_bgr, imgsz=640, conf=0.25, iou=0.70, device=device, verbose=False)[0]
        rs = seg_model.predict(source=img_bgr, imgsz=640, conf=0.25, iou=0.70, retina_masks=True, device=device, verbose=False)[0]

        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        det_vis, nd, dconfs = draw_det_panel(img_bgr, rd)
        seg_vis, ns, sconfs = draw_seg_panel(img_bgr, rs)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), dpi=135)
        axes[0].imshow(orig_rgb)
        ts_str = f"t={ts:.2f}s | " if ts is not None else ""
        axes[0].set_title(f"Original ({ts_str}frame {s['frame']})", fontsize=9.5)
        axes[1].imshow(det_vis)
        axes[1].set_title(f"Phase 2 YOLOv8n Det (n={nd}, >=0.50: {sum(c>=0.5 for c in dconfs)})", fontsize=9.5)
        axes[2].imshow(seg_vis)
        axes[2].set_title(f"Phase 4A.2 YOLOv8n-seg (n={ns}, >=0.50: {sum(c>=0.5 for c in sconfs)})", fontsize=9.5)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{s['id']} | {s['source']} | {s['scenario']} (conf=0.25, imgsz=640)", fontsize=10.5, fontweight="bold")
        plt.tight_layout()
        viz_path = os.path.join(OUT_DIR, f"{s['id']}.png")
        plt.savefig(viz_path, bbox_inches="tight")
        plt.close(fig)

        sample_records.append({
            **s,
            "timestamp_sec": ts,
            "resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
            "det_count_conf025": nd,
            "det_count_conf050": sum(c >= 0.5 for c in dconfs),
            "det_confs": dconfs,
            "seg_count_conf025": ns,
            "seg_count_conf050": sum(c >= 0.5 for c in sconfs),
            "seg_confs": sconfs,
            "viz_path": viz_path,
        })

    out_json = os.path.join(OUT_DIR, "dashcam_transfer_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "inference_config": {"imgsz": 640, "visual_conf": 0.25, "iou_nms": 0.70, "retina_masks": True},
            "box_benchmarks": box_benchmarks,
            "selected_frames": sample_records,
        }, f, indent=2)

    print("=== DASHCAM TRANSFER COMPLETE ===")
    print(json.dumps(box_benchmarks, indent=2))


if __name__ == "__main__":
    main()
