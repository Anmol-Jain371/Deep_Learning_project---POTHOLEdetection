"""
========================================================================================
RoadVision AI -- Phase 4A.4: Experimental Detector-Guided ROI Segmentation & Fallback
========================================================================================
Experimental cascade combining:
  1. Phase 2 controlled YOLOv8n detector (full-frame dashcam proposal generator)
  2. Aspect-ratio-preserving ROI crop extraction + padding + boundary clipping
  3. Phase 4A.2 YOLOv8n-seg applied to each ROI crop (sequential OR batched)
  4. Exact coordinate inverse-mapping & detector-box mask association
  5. Deterministic geometric mask quality assessment with explicit bounding-box fallback

STRICT SCIENTIFIC DISCLAIMERS:
  - Quality acceptance rules are deterministic geometric heuristics, NOT calibrated
    probabilities of segmentation correctness.
  - Accepted mask counts measure heuristic plausibility, NOT verified pixel IoU/Dice.
  - Experimental research module only; does NOT modify roadvision_engine.py or server.py.
========================================================================================
"""

import os
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

import cv2
import torch
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DET_WEIGHTS = os.path.join(
    BASE_DIR, "runs", "detect", "pothole_unified_yolov8n_100e_sgd", "weights", "best.pt"
)
DEFAULT_SEG_WEIGHTS = os.path.join(
    BASE_DIR, "runs", "segment", "pothole440_yolov8n_seg_v1", "weights", "best.pt"
)


@dataclass
class ROISegConfig:
    det_conf: float = 0.25
    det_iou: float = 0.70
    det_imgsz: int = 640
    crop_pad_ratio: float = 0.40
    min_pad_px: int = 24
    seg_conf: float = 0.25
    seg_iou: float = 0.70
    seg_imgsz: int = 640
    mask_bin_thresh: float = 0.50
    batch_roi: bool = False
    max_batch_size: int = 8

    # Deterministic geometric mask quality thresholds (NOT calibrated probabilities)
    min_seg_conf: float = 0.25
    min_mask_area_px: int = 36
    min_area_ratio_to_box: float = 0.12
    max_area_ratio_to_box: float = 1.35
    min_frac_inside_det_box: float = 0.55
    min_det_box_coverage: float = 0.15
    max_crop_boundary_contact: float = 0.30
    max_connected_components: int = 4
    ambiguous_iou_thresh: float = 0.20
    reject_on_ambiguity: bool = True


def expand_and_clip_box(
    box_xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.40,
    min_pad_px: int = 24,
) -> Tuple[int, int, int, int]:
    """Expand a detector box with surrounding context and clip cleanly to valid image bounds."""
    x1, y1, x2, y2 = box_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(float(min_pad_px), bw * pad_ratio)
    pad_y = max(float(min_pad_px), bh * pad_ratio)

    cx1 = max(0, int(np.floor(x1 - pad_x)))
    cy1 = max(0, int(np.floor(y1 - pad_y)))
    cx2 = min(img_w, int(np.ceil(x2 + pad_x)))
    cy2 = min(img_h, int(np.ceil(y2 + pad_y)))
    if cx2 <= cx1:
        cx2 = min(img_w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(img_h, cy1 + 1)
    return cx1, cy1, cx2, cy2


def letterbox_to_square(
    crop_bgr: np.ndarray, target_size: int = 640, pad_value: int = 114
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resize crop preserving exact aspect ratio into a (target_size x target_size) canvas.
    Returns (canvas_bgr, letterbox_metadata) for exact inverse projection.
    """
    ch, cw = crop_bgr.shape[:2]
    scale = min(target_size / float(max(1, ch)), target_size / float(max(1, cw)))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=interp)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
    )
    meta = {
        "crop_w": cw,
        "crop_h": ch,
        "target_size": target_size,
        "scale": scale,
        "new_w": new_w,
        "new_h": new_h,
        "left": left,
        "top": top,
    }
    return canvas, meta


def unletterbox_mask_to_crop(
    mask_canvas: np.ndarray, meta: Dict[str, Any], bin_thresh: float = 0.50
) -> np.ndarray:
    """Invert letterbox padding and rescale a probability mask back to exact (crop_h, crop_w) coordinates."""
    mh, mw = mask_canvas.shape[:2]
    t = float(meta["target_size"])
    # Handle cases where Ultralytics returns proto/native mask resolution differing from target_size
    if mh != meta["target_size"] or mw != meta["target_size"]:
        mask_canvas = cv2.resize(mask_canvas.astype(np.float32), (meta["target_size"], meta["target_size"]), interpolation=cv2.INTER_LINEAR)

    y1 = meta["top"]
    y2 = y1 + meta["new_h"]
    x1 = meta["left"]
    x2 = x1 + meta["new_w"]
    active = mask_canvas[y1:y2, x1:x2]
    if active.size == 0:
        return np.zeros((meta["crop_h"], meta["crop_w"]), dtype=bool)
    restored = cv2.resize(active.astype(np.float32), (meta["crop_w"], meta["crop_h"]), interpolation=cv2.INTER_LINEAR)
    return restored >= bin_thresh


def unletterbox_box_to_frame(
    box_canvas_xyxy: np.ndarray, crop_xyxy: Tuple[int, int, int, int], meta: Dict[str, Any]
) -> np.ndarray:
    """Map predicted [x1, y1, x2, y2] in letterboxed canvas coordinates back to full-frame pixel space."""
    cx1, cy1, _, _ = crop_xyxy
    scale = max(1e-9, float(meta["scale"]))
    bx1 = (float(box_canvas_xyxy[0]) - meta["left"]) / scale + cx1
    by1 = (float(box_canvas_xyxy[1]) - meta["top"]) / scale + cy1
    bx2 = (float(box_canvas_xyxy[2]) - meta["left"]) / scale + cx1
    by2 = (float(box_canvas_xyxy[3]) - meta["top"]) / scale + cy1
    return np.array([
        np.clip(bx1, cx1, cx1 + meta["crop_w"]),
        np.clip(by1, cy1, cy1 + meta["crop_h"]),
        np.clip(bx2, cx1, cx1 + meta["crop_w"]),
        np.clip(by2, cy1, cy1 + meta["crop_h"]),
    ], dtype=float)


def place_crop_mask_in_frame(
    crop_mask: np.ndarray, crop_xyxy: Tuple[int, int, int, int], img_h: int, img_w: int
) -> np.ndarray:
    """Embed a boolean (crop_h, crop_w) mask into a full-frame (img_h, img_w) boolean array."""
    full_mask = np.zeros((img_h, img_w), dtype=bool)
    cx1, cy1, cx2, cy2 = crop_xyxy
    full_mask[cy1:cy2, cx1:cx2] = crop_mask[: (cy2 - cy1), : (cx2 - cx1)]
    return full_mask


def box_to_bool_mask(box_xyxy: Tuple[float, float, float, float], img_h: int, img_w: int) -> np.ndarray:
    m = np.zeros((img_h, img_w), dtype=bool)
    x1 = max(0, int(np.floor(box_xyxy[0])))
    y1 = max(0, int(np.floor(box_xyxy[1])))
    x2 = min(img_w, int(np.ceil(box_xyxy[2])))
    y2 = min(img_h, int(np.ceil(box_xyxy[3])))
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def evaluate_roi_mask_quality(
    det_box_xyxy: Tuple[float, float, float, float],
    crop_xyxy: Tuple[int, int, int, int],
    cand_masks_crop: List[np.ndarray],
    cand_confs: List[float],
    cfg: ROISegConfig,
) -> Dict[str, Any]:
    """
    Associate candidate ROI masks with the original detector box and apply deterministic
    geometric quality rules. Always preserves original detection on rejection via bbox fallback.
    """
    cx1, cy1, cx2, cy2 = crop_xyxy
    ch, cw = cy2 - cy1, cx2 - cx1

    # Detector box relative to crop coordinates
    dx1 = max(0, int(np.floor(det_box_xyxy[0] - cx1)))
    dy1 = max(0, int(np.floor(det_box_xyxy[1] - cy1)))
    dx2 = min(cw, int(np.ceil(det_box_xyxy[2] - cx1)))
    dy2 = min(ch, int(np.ceil(det_box_xyxy[3] - cy1)))
    det_crop_mask = np.zeros((ch, cw), dtype=bool)
    if dx2 > dx1 and dy2 > dy1:
        det_crop_mask[dy1:dy2, dx1:dx2] = True
    det_area = max(1, int(det_crop_mask.sum()))

    if len(cand_masks_crop) == 0:
        return {
            "accepted": False,
            "fallback_used": True,
            "rejection_reasons": ["empty_mask_prediction"],
            "best_crop_mask": None,
            "num_roi_candidates": 0,
            "ambiguous_association": False,
            "seg_conf": 0.0,
            "mask_area_px": 0,
            "det_box_area_px": det_area,
            "area_ratio_to_box": 0.0,
            "frac_inside_det_box": 0.0,
            "det_box_coverage": 0.0,
            "mask_det_iou": 0.0,
            "crop_boundary_contact": 0.0,
            "connected_components": 0,
        }

    # Score all candidates for association with the detector box
    scored = []
    for m, sc in zip(cand_masks_crop, cand_confs):
        area = int(m.sum())
        inter = int(np.logical_and(m, det_crop_mask).sum())
        union = int(np.logical_or(m, det_crop_mask).sum())
        iou = inter / float(max(1, union))
        scored.append((iou, inter, area, sc, m))

    scored.sort(key=lambda x: (x[0], x[3]), reverse=True)
    best_iou, best_inter, best_area, best_conf, best_mask = scored[0]

    competing = [s for s in scored[1:] if s[0] >= cfg.ambiguous_iou_thresh and s[3] >= cfg.min_seg_conf]
    ambiguous = len(competing) > 0 and (competing[0][0] / max(1e-6, best_iou) >= 0.70)

    reasons: List[str] = []
    if best_area == 0 or best_inter == 0:
        reasons.append("no_overlap_with_detector_box" if best_area > 0 else "empty_mask_prediction")

    area_ratio = best_area / float(det_area)
    frac_inside = best_inter / float(max(1, best_area))
    det_cov = best_inter / float(det_area)

    # Boundary contact along expanded crop borders (only penalize margins that actually have padding)
    border_mask = np.zeros((ch, cw), dtype=bool)
    if dx1 > 3:
        border_mask[:, 0] = True
    if (cw - dx2) > 3:
        border_mask[:, -1] = True
    if dy1 > 3:
        border_mask[0, :] = True
    if (ch - dy2) > 3:
        border_mask[-1, :] = True

    u8 = best_mask.astype(np.uint8)
    eroded = cv2.erode(u8, np.ones((3, 3), np.uint8), iterations=1)
    perim = (u8 - eroded) > 0
    perim_cnt = max(1, int(perim.sum()))
    border_contact_ratio = int(np.logical_and(best_mask, border_mask).sum()) / float(perim_cnt)

    # Connected components >= 4 px
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    n_comp = sum(1 for cid in range(1, n_lbl) if stats[cid, cv2.CC_STAT_AREA] >= 4)

    if best_conf < cfg.min_seg_conf:
        reasons.append("low_seg_confidence")
    if best_area < cfg.min_mask_area_px and "empty_mask_prediction" not in reasons:
        reasons.append("below_min_mask_area")
    if area_ratio < cfg.min_area_ratio_to_box and "empty_mask_prediction" not in reasons:
        reasons.append("undersized_vs_det_box")
    if area_ratio > cfg.max_area_ratio_to_box:
        reasons.append("excessive_area_ratio")
    if best_area > 0 and frac_inside < cfg.min_frac_inside_det_box:
        reasons.append("outside_detector_box_spill")
    if best_area > 0 and det_cov < cfg.min_det_box_coverage:
        reasons.append("low_det_box_coverage")
    if border_contact_ratio > cfg.max_crop_boundary_contact:
        reasons.append("excessive_crop_boundary_contact")
    if n_comp > cfg.max_connected_components:
        reasons.append("excessive_fragmentation")
    if cfg.reject_on_ambiguity and ambiguous:
        reasons.append("ambiguous_roi_association")

    accepted = len(reasons) == 0
    return {
        "accepted": accepted,
        "fallback_used": not accepted,
        "rejection_reasons": reasons,
        "best_crop_mask": best_mask,
        "num_roi_candidates": len(cand_masks_crop),
        "ambiguous_association": ambiguous,
        "seg_conf": round(float(best_conf), 4),
        "mask_area_px": best_area,
        "det_box_area_px": det_area,
        "area_ratio_to_box": round(float(area_ratio), 4),
        "frac_inside_det_box": round(float(frac_inside), 4),
        "det_box_coverage": round(float(det_cov), 4),
        "mask_det_iou": round(float(best_iou), 4),
        "crop_boundary_contact": round(float(border_contact_ratio), 4),
        "connected_components": n_comp,
    }


class ROIGuidedSegmenter:
    """Detector-guided ROI segmentation cascade with quality-aware bounding-box fallback."""

    def __init__(
        self,
        det_weights: str = DEFAULT_DET_WEIGHTS,
        seg_weights: str = DEFAULT_SEG_WEIGHTS,
        config: Optional[ROISegConfig] = None,
        device: Optional[Any] = None,
    ):
        self.cfg = config or ROISegConfig()
        self.device = (0 if torch.cuda.is_available() else "cpu") if device is None else device
        self.det_model = YOLO(det_weights)
        self.seg_model = YOLO(seg_weights)

    def warmup(self, h: int = 360, w: int = 640) -> None:
        dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
        dummy_crop = np.zeros((self.cfg.seg_imgsz, self.cfg.seg_imgsz, 3), dtype=np.uint8)
        _ = self.det_model.predict(
            source=dummy_frame, imgsz=self.cfg.det_imgsz, conf=self.cfg.det_conf, iou=self.cfg.det_iou, device=self.device, verbose=False
        )
        _ = self.seg_model.predict(
            source=dummy_crop, imgsz=self.cfg.seg_imgsz, conf=self.cfg.seg_conf, iou=self.cfg.seg_iou, retina_masks=True, device=self.device, verbose=False
        )

    def _sync(self) -> None:
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()

    def process_frame(self, frame_bgr: np.ndarray, cfg_override: Optional[ROISegConfig] = None) -> Dict[str, Any]:
        cfg = cfg_override or self.cfg
        H, W = frame_bgr.shape[:2]

        # 1. Detector inference
        self._sync()
        t_det0 = time.perf_counter()
        det_res = self.det_model.predict(
            source=frame_bgr,
            imgsz=cfg.det_imgsz,
            conf=cfg.det_conf,
            iou=cfg.det_iou,
            device=self.device,
            verbose=False,
        )[0]
        self._sync()
        det_ms = (time.perf_counter() - t_det0) * 1000.0

        if det_res.boxes is None or len(det_res.boxes) == 0:
            return {
                "proposals": [],
                "union_accepted_mask": np.zeros((H, W), dtype=bool),
                "union_effective_mask": np.zeros((H, W), dtype=bool),
                "summary": {
                    "num_proposals": 0,
                    "seg_attempts": 0,
                    "accepted_masks": 0,
                    "rejected_masks": 0,
                    "empty_masks": 0,
                    "bbox_fallbacks": 0,
                    "detections_preserved": 0,
                },
                "timing_ms": {
                    "detector_ms": round(det_ms, 3),
                    "roi_extract_ms": 0.0,
                    "seg_roi_ms": 0.0,
                    "mask_reconstruct_ms": 0.0,
                    "quality_eval_ms": 0.0,
                    "combined_total_ms": round(det_ms, 3),
                },
            }

        det_boxes = det_res.boxes.xyxy.cpu().numpy()
        det_confs = det_res.boxes.conf.cpu().numpy()

        # 2. ROI extraction & letterboxing
        t_ext0 = time.perf_counter()
        roi_items = []
        for idx, (box, sc) in enumerate(zip(det_boxes, det_confs)):
            crop_xyxy = expand_and_clip_box(
                tuple(float(v) for v in box), W, H, pad_ratio=cfg.crop_pad_ratio, min_pad_px=cfg.min_pad_px
            )
            cx1, cy1, cx2, cy2 = crop_xyxy
            crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
            sq_bgr, lb_meta = letterbox_to_square(crop_bgr, target_size=cfg.seg_imgsz)
            roi_items.append({
                "idx": idx,
                "det_box_xyxy": [float(v) for v in box],
                "det_conf": float(sc),
                "crop_xyxy": crop_xyxy,
                "sq_bgr": sq_bgr,
                "lb_meta": lb_meta,
            })
        roi_extract_ms = (time.perf_counter() - t_ext0) * 1000.0

        # 3. ROI segmentation inference (batched or sequential)
        self._sync()
        t_seg0 = time.perf_counter()
        seg_results = []
        if cfg.batch_roi and len(roi_items) > 1:
            for start in range(0, len(roi_items), cfg.max_batch_size):
                chunk_imgs = [it["sq_bgr"] for it in roi_items[start : start + cfg.max_batch_size]]
                chunk_res = self.seg_model.predict(
                    source=chunk_imgs,
                    imgsz=cfg.seg_imgsz,
                    conf=cfg.seg_conf,
                    iou=cfg.seg_iou,
                    retina_masks=True,
                    device=self.device,
                    verbose=False,
                )
                seg_results.extend(chunk_res)
        else:
            for it in roi_items:
                r = self.seg_model.predict(
                    source=it["sq_bgr"],
                    imgsz=cfg.seg_imgsz,
                    conf=cfg.seg_conf,
                    iou=cfg.seg_iou,
                    retina_masks=True,
                    device=self.device,
                    verbose=False,
                )[0]
                seg_results.append(r)
        self._sync()
        seg_roi_ms = (time.perf_counter() - t_seg0) * 1000.0

        # 4 & 5. Mask reconstruction + quality assessment + fallback
        proposals_out = []
        union_accepted = np.zeros((H, W), dtype=bool)
        union_effective = np.zeros((H, W), dtype=bool)

        reconstruct_ms_acc = 0.0
        quality_ms_acc = 0.0

        for it, r0 in zip(roi_items, seg_results):
            t_rec0 = time.perf_counter()
            cand_masks_crop: List[np.ndarray] = []
            cand_confs: List[float] = []
            if r0.boxes is not None and len(r0.boxes) > 0 and r0.masks is not None and r0.masks.data is not None:
                m_np = r0.masks.data.cpu().numpy()
                c_np = r0.boxes.conf.cpu().numpy()
                for k in range(m_np.shape[0]):
                    cm = unletterbox_mask_to_crop(m_np[k], it["lb_meta"], bin_thresh=cfg.mask_bin_thresh)
                    cand_masks_crop.append(cm)
                    cand_confs.append(float(c_np[k]))
            reconstruct_ms_acc += (time.perf_counter() - t_rec0) * 1000.0

            t_qa0 = time.perf_counter()
            qa = evaluate_roi_mask_quality(
                det_box_xyxy=tuple(it["det_box_xyxy"]),
                crop_xyxy=it["crop_xyxy"],
                cand_masks_crop=cand_masks_crop,
                cand_confs=cand_confs,
                cfg=cfg,
            )
            quality_ms_acc += (time.perf_counter() - t_qa0) * 1000.0

            t_rec1 = time.perf_counter()
            raw_best_full = (
                place_crop_mask_in_frame(qa["best_crop_mask"], it["crop_xyxy"], H, W)
                if qa["best_crop_mask"] is not None
                else None
            )
            bbox_full = box_to_bool_mask(tuple(it["det_box_xyxy"]), H, W)
            if qa["accepted"] and raw_best_full is not None:
                final_mask = raw_best_full
                final_rep = "roi_seg_mask"
                union_accepted |= final_mask
            else:
                final_mask = bbox_full
                final_rep = "bbox_fallback"
            union_effective |= final_mask
            reconstruct_ms_acc += (time.perf_counter() - t_rec1) * 1000.0

            proposals_out.append({
                "idx": it["idx"],
                "det_box_xyxy": [round(v, 2) for v in it["det_box_xyxy"]],
                "det_conf": round(it["det_conf"], 4),
                "crop_xyxy": list(it["crop_xyxy"]),
                "final_representation": final_rep,
                "accepted": qa["accepted"],
                "fallback_used": qa["fallback_used"],
                "rejection_reasons": qa["rejection_reasons"],
                "num_roi_candidates": qa["num_roi_candidates"],
                "ambiguous_association": qa["ambiguous_association"],
                "seg_conf": qa["seg_conf"],
                "mask_area_px": qa["mask_area_px"],
                "det_box_area_px": qa["det_box_area_px"],
                "area_ratio_to_box": qa["area_ratio_to_box"],
                "frac_inside_det_box": qa["frac_inside_det_box"],
                "det_box_coverage": qa["det_box_coverage"],
                "mask_det_iou": qa["mask_det_iou"],
                "crop_boundary_contact": qa["crop_boundary_contact"],
                "connected_components": qa["connected_components"],
                "final_mask": final_mask,
                "rejected_raw_mask": raw_best_full if qa["fallback_used"] else None,
            })

        total_ms = det_ms + roi_extract_ms + seg_roi_ms + reconstruct_ms_acc + quality_ms_acc
        n_prop = len(proposals_out)
        n_acc = sum(1 for p in proposals_out if p["accepted"])
        n_rej = n_prop - n_acc
        n_emp = sum(1 for p in proposals_out if "empty_mask_prediction" in p["rejection_reasons"])

        return {
            "proposals": proposals_out,
            "union_accepted_mask": union_accepted,
            "union_effective_mask": union_effective,
            "summary": {
                "num_proposals": n_prop,
                "seg_attempts": n_prop,
                "accepted_masks": n_acc,
                "rejected_masks": n_rej,
                "empty_masks": n_emp,
                "bbox_fallbacks": n_rej,
                "detections_preserved": n_prop,
            },
            "timing_ms": {
                "detector_ms": round(det_ms, 3),
                "roi_extract_ms": round(roi_extract_ms, 3),
                "seg_roi_ms": round(seg_roi_ms, 3),
                "mask_reconstruct_ms": round(reconstruct_ms_acc, 3),
                "quality_eval_ms": round(quality_ms_acc, 3),
                "combined_total_ms": round(total_ms, 3),
            },
        }


def config_to_dict(cfg: ROISegConfig) -> Dict[str, Any]:
    return asdict(cfg)
