"""
========================================================================================
RoadVision AI -- Phase 4A.5: Zero-Shot Box-Prompted Segmentation Adapter (MobileSAM)
========================================================================================
Combines:
  1. Phase 2 controlled YOLOv8n detector (full-frame pothole box proposals)
  2. Ultralytics SAMPredictor with pretrained MobileSAM (`mobile_sam.pt`, Apache-2.0,
     10.13M params: TinyViT image encoder + SAM prompt encoder & mask decoder)
  3. Reused full-frame image embeddings across all box prompts (plus cropped-ROI prompting
     mode for controlled architectural comparison)
  4. Multi-mask hypothesis selection (`multimask_output=True`, 3 scales per prompt)
     combined with deterministic geometric quality checks and bounding-box fallback.

STRICT SCIENTIFIC DISCLAIMERS:
  - MobileSAM predicted IoU scores are model-internal scores, NOT calibrated
    probabilities that a bounding box contains a true pothole.
  - Geometric acceptance indicates bounding-box consistency, NOT verified pixel IoU.
  - Experimental research adapter only; does NOT modify roadvision_engine.py or server.py.
========================================================================================
"""

import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.roi_guided_segmentation import (
    DEFAULT_DET_WEIGHTS,
    ROISegConfig,
    expand_and_clip_box,
    box_to_bool_mask,
    evaluate_roi_mask_quality,
    place_crop_mask_in_frame,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MOBILESAM_WEIGHTS = os.path.join(BASE_DIR, "mobile_sam.pt")


@dataclass
class PromptSegConfig:
    det_conf: float = 0.25
    det_iou: float = 0.70
    det_imgsz: int = 640
    sam_imgsz: int = 1024
    multimask_output: bool = True
    mode: str = "full_frame_embed"  # "full_frame_embed" or "cropped_roi_prompt"
    crop_pad_ratio: float = 0.40
    min_pad_px: int = 24

    # Deterministic geometric mask quality thresholds (aligned with Phase 4A.4 ROISegConfig)
    min_sam_score: float = 0.25
    min_mask_area_px: int = 36
    min_area_ratio_to_box: float = 0.12
    max_area_ratio_to_box: float = 1.35
    min_frac_inside_det_box: float = 0.55
    min_det_box_coverage: float = 0.15
    max_crop_boundary_contact: float = 0.30
    max_connected_components: int = 4


def unpad_and_resize_sam_logits(
    logits_k: torch.Tensor,
    src_h: int,
    src_w: int,
    input_size: int = 1024,
    mask_threshold: float = 0.0,
) -> np.ndarray:
    """
    Invert Ultralytics SAMPredictor top-left LetterBox(imgsz=1024, center=False)
    for [K, H_m, W_m] mask logits back to exact (K, src_h, src_w) boolean masks.
    """
    if logits_k.ndim == 2:
        logits_k = logits_k[None]
    gain = min(input_size / float(src_h), input_size / float(src_w))
    active_h = max(1, int(round(src_h * gain)))
    active_w = max(1, int(round(src_w * gain)))

    up = F.interpolate(
        logits_k[:, None].float(), size=(input_size, input_size), mode="bilinear", align_corners=False
    )[:, 0]
    cropped = up[:, :active_h, :active_w]
    restored = F.interpolate(
        cropped[:, None], size=(src_h, src_w), mode="bilinear", align_corners=False
    )[:, 0]
    return (restored > mask_threshold).cpu().numpy()


def assess_sam_candidates_for_box(
    det_box_xyxy: Tuple[float, float, float, float],
    cand_masks_full: np.ndarray,  # (C, H, W) bool
    cand_scores: List[float],
    img_w: int,
    img_h: int,
    cfg: PromptSegConfig,
) -> Dict[str, Any]:
    """
    Evaluate SAM multi-mask candidates (up to 3 scale hypotheses) against deterministic
    geometric consistency rules relative to the enclosing detector box.
    """
    if cand_masks_full is None or len(cand_masks_full) == 0:
        return {
            "accepted": False,
            "fallback_used": True,
            "rejection_reasons": ["empty_mask_prediction"],
            "best_full_mask": None,
            "best_candidate_idx": -1,
            "num_candidates": 0,
            "all_candidate_scores": [],
            "sam_score": 0.0,
            "mask_area_px": 0,
            "det_box_area_px": max(1, int(box_to_bool_mask(det_box_xyxy, img_h, img_w).sum())),
            "area_ratio_to_box": 0.0,
            "frac_inside_det_box": 0.0,
            "det_box_coverage": 0.0,
            "mask_det_iou": 0.0,
            "crop_boundary_contact": 0.0,
            "connected_components": 0,
        }

    # Virtual expanded reference region mirrors Phase 4A.4 crop boundary semantics
    crop_xyxy = expand_and_clip_box(
        det_box_xyxy, img_w, img_h, pad_ratio=cfg.crop_pad_ratio, min_pad_px=cfg.min_pad_px
    )
    cx1, cy1, cx2, cy2 = crop_xyxy
    det_full = box_to_bool_mask(det_box_xyxy, img_h, img_w)
    det_area = max(1, int(det_full.sum()))

    qa_cfg = ROISegConfig(
        min_seg_conf=cfg.min_sam_score,
        min_mask_area_px=cfg.min_mask_area_px,
        min_area_ratio_to_box=cfg.min_area_ratio_to_box,
        max_area_ratio_to_box=cfg.max_area_ratio_to_box,
        min_frac_inside_det_box=cfg.min_frac_inside_det_box,
        min_det_box_coverage=cfg.min_det_box_coverage,
        max_crop_boundary_contact=cfg.max_crop_boundary_contact,
        max_connected_components=cfg.max_connected_components,
        reject_on_ambiguity=False,  # Multi-mask scales for one prompt are hypotheses, not separate objects
    )

    evaluated = []
    for k in range(len(cand_masks_full)):
        full_m = cand_masks_full[k]
        sc = float(cand_scores[k])
        crop_m = full_m[cy1:cy2, cx1:cx2]
        qa = evaluate_roi_mask_quality(
            det_box_xyxy=det_box_xyxy,
            crop_xyxy=crop_xyxy,
            cand_masks_crop=[crop_m],
            cand_confs=[sc],
            cfg=qa_cfg,
        )
        # Enforce full-frame containment & area ratio (catch pixels spilling beyond virtual crop)
        full_area = int(full_m.sum())
        full_inter = int(np.logical_and(full_m, det_full).sum())
        full_union = int(np.logical_or(full_m, det_full).sum())
        full_ratio = full_area / float(det_area)
        full_inside = full_inter / float(max(1, full_area))
        full_iou = full_inter / float(max(1, full_union))
        reasons = list(qa["rejection_reasons"])
        if full_ratio > cfg.max_area_ratio_to_box and "excessive_area_ratio" not in reasons:
            reasons.append("excessive_area_ratio")
        if full_area > 0 and full_inside < cfg.min_frac_inside_det_box and "outside_detector_box_spill" not in reasons:
            reasons.append("outside_detector_box_spill")
        accepted = len(reasons) == 0

        evaluated.append({
            "cand_idx": k,
            "accepted": accepted,
            "fallback_used": not accepted,
            "rejection_reasons": reasons,
            "best_full_mask": full_m,
            "sam_score": round(sc, 4),
            "mask_area_px": full_area,
            "det_box_area_px": det_area,
            "area_ratio_to_box": round(full_ratio, 4),
            "frac_inside_det_box": round(full_inside, 4),
            "det_box_coverage": round(full_inter / float(det_area), 4),
            "mask_det_iou": round(full_iou, 4),
            "crop_boundary_contact": qa["crop_boundary_contact"],
            "connected_components": qa["connected_components"],
        })

    # Prefer accepted candidate with highest SAM score + IoU, else highest SAM score diagnostic
    accepted_cands = [e for e in evaluated if e["accepted"]]
    if accepted_cands:
        chosen = max(accepted_cands, key=lambda e: (e["sam_score"], e["mask_det_iou"]))
    else:
        chosen = max(evaluated, key=lambda e: (e["mask_det_iou"], e["sam_score"]))

    return {
        **chosen,
        "best_candidate_idx": chosen["cand_idx"],
        "num_candidates": len(evaluated),
        "all_candidate_scores": [round(float(s), 4) for s in cand_scores],
    }


class MobileSAMBoxPromptAdapter:
    """Detector-driven zero-shot MobileSAM segmentation with embedding reuse & fallback."""

    def __init__(
        self,
        det_weights: str = DEFAULT_DET_WEIGHTS,
        sam_weights: str = DEFAULT_MOBILESAM_WEIGHTS,
        config: Optional[PromptSegConfig] = None,
        device: Optional[Any] = None,
    ):
        self.cfg = config or PromptSegConfig()
        self.device = (0 if torch.cuda.is_available() else "cpu") if device is None else device

        t0 = time.perf_counter()
        self.det_model = YOLO(det_weights)
        self.sam_predictor = SAMPredictor(
            overrides=dict(
                conf=0.0,
                task="segment",
                model=sam_weights,
                device=self.device,
                imgsz=self.cfg.sam_imgsz,
                verbose=False,
            )
        )
        self.sam_predictor.setup_model(verbose=False)
        self.startup_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def _sync(self) -> None:
        if torch.cuda.is_available() and str(self.device) != "cpu":
            torch.cuda.synchronize()

    def warmup(self, h: int = 360, w: int = 640) -> None:
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        _ = self.det_model.predict(
            source=dummy, imgsz=self.cfg.det_imgsz, conf=self.cfg.det_conf, iou=self.cfg.det_iou, device=self.device, verbose=False
        )
        self.sam_predictor.setup_source(dummy)
        for batch in self.sam_predictor.dataset:
            im = self.sam_predictor.preprocess(batch[1])
            feats = self.sam_predictor.get_im_features(im)
            prompts = self.sam_predictor._prepare_prompts(im.shape[2:], (h, w), bboxes=[[20, 20, 120, 120]])
            _ = self.sam_predictor._inference_features(feats, *prompts, multimask_output=self.cfg.multimask_output)
            break
        self.sam_predictor.reset_image()

    def _encode_sam_image(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        self.sam_predictor.setup_source(img_bgr)
        im = None
        self._sync()
        t_pre0 = time.perf_counter()
        for batch in self.sam_predictor.dataset:
            im = self.sam_predictor.preprocess(batch[1])
            break
        self._sync()
        preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

        self._sync()
        t_enc0 = time.perf_counter()
        with torch.inference_mode():
            feats = self.sam_predictor.get_im_features(im)
        self._sync()
        encoder_ms = (time.perf_counter() - t_enc0) * 1000.0
        return im, feats, preprocess_ms, encoder_ms

    def process_frame(self, frame_bgr: np.ndarray, mode: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.cfg
        active_mode = mode or cfg.mode
        H, W = frame_bgr.shape[:2]

        # 1. Detector proposal generation
        self._sync()
        t_det0 = time.perf_counter()
        det_res = self.det_model.predict(
            source=frame_bgr, imgsz=cfg.det_imgsz, conf=cfg.det_conf, iou=cfg.det_iou, device=self.device, verbose=False
        )[0]
        self._sync()
        det_ms = (time.perf_counter() - t_det0) * 1000.0

        if det_res.boxes is None or len(det_res.boxes) == 0:
            return {
                "mode": active_mode,
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
                    "sam_preprocess_ms": 0.0,
                    "sam_encoder_ms": 0.0,
                    "sam_decoder_ms": 0.0,
                    "sam_decoder_per_box_ms": 0.0,
                    "mask_reconstruct_ms": 0.0,
                    "quality_eval_ms": 0.0,
                    "combined_total_ms": round(det_ms, 3),
                },
            }

        det_boxes = det_res.boxes.xyxy.cpu().numpy()
        det_confs = det_res.boxes.conf.cpu().numpy()
        n_prop = len(det_boxes)

        preprocess_ms = encoder_ms = decoder_ms = reconstruct_ms = quality_ms = 0.0
        per_box_cands: List[Tuple[np.ndarray, List[float]]] = []

        if active_mode == "full_frame_embed":
            # Reuse ONE full-frame TinyViT image embedding across all N detector boxes
            im, feats, preprocess_ms, encoder_ms = self._encode_sam_image(frame_bgr)

            self._sync()
            t_dec0 = time.perf_counter()
            with torch.inference_mode():
                prompts = self.sam_predictor._prepare_prompts(im.shape[2:], (H, W), bboxes=det_boxes.tolist())
                pred_logits, pred_scores = self.sam_predictor._inference_features(
                    feats, *prompts, multimask_output=cfg.multimask_output
                )
            self._sync()
            decoder_ms = (time.perf_counter() - t_dec0) * 1000.0
            self.sam_predictor.reset_image()

            t_rec0 = time.perf_counter()
            c_per_box = 3 if cfg.multimask_output else 1
            bool_masks_all = unpad_and_resize_sam_logits(
                pred_logits, H, W, input_size=cfg.sam_imgsz, mask_threshold=self.sam_predictor.model.mask_threshold
            )
            scores_all = pred_scores.cpu().numpy().tolist()
            for i in range(n_prop):
                s_idx = i * c_per_box
                e_idx = s_idx + c_per_box
                per_box_cands.append((bool_masks_all[s_idx:e_idx], scores_all[s_idx:e_idx]))
            reconstruct_ms = (time.perf_counter() - t_rec0) * 1000.0

        elif active_mode == "cropped_roi_prompt":
            # Run TinyViT + decoder separately inside each padded ROI crop (no embedding reuse across crops)
            for box in det_boxes:
                cx1, cy1, cx2, cy2 = expand_and_clip_box(
                    tuple(float(v) for v in box), W, H, pad_ratio=cfg.crop_pad_ratio, min_pad_px=cfg.min_pad_px
                )
                crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
                ch, cw = crop_bgr.shape[:2]
                im, feats, p_ms, e_ms = self._encode_sam_image(crop_bgr)
                preprocess_ms += p_ms
                encoder_ms += e_ms

                rel_box = [
                    float(box[0] - cx1),
                    float(box[1] - cy1),
                    float(box[2] - cx1),
                    float(box[3] - cy1),
                ]
                self._sync()
                t_dec0 = time.perf_counter()
                with torch.inference_mode():
                    prompts = self.sam_predictor._prepare_prompts(im.shape[2:], (ch, cw), bboxes=[rel_box])
                    pred_logits, pred_scores = self.sam_predictor._inference_features(
                        feats, *prompts, multimask_output=cfg.multimask_output
                    )
                self._sync()
                decoder_ms += (time.perf_counter() - t_dec0) * 1000.0
                self.sam_predictor.reset_image()

                t_rec0 = time.perf_counter()
                crop_bool = unpad_and_resize_sam_logits(
                    pred_logits, ch, cw, input_size=cfg.sam_imgsz, mask_threshold=self.sam_predictor.model.mask_threshold
                )
                full_cands = np.stack(
                    [place_crop_mask_in_frame(crop_bool[k], (cx1, cy1, cx2, cy2), H, W) for k in range(len(crop_bool))],
                    axis=0,
                )
                per_box_cands.append((full_cands, pred_scores.cpu().numpy().tolist()))
                reconstruct_ms += (time.perf_counter() - t_rec0) * 1000.0
        else:
            raise ValueError(f"Unknown mode: {active_mode}")

        # Quality assessment & fallback
        proposals_out = []
        union_accepted = np.zeros((H, W), dtype=bool)
        union_effective = np.zeros((H, W), dtype=bool)

        for idx, (box, d_conf, (cands_full, c_scores)) in enumerate(zip(det_boxes, det_confs, per_box_cands)):
            t_qa0 = time.perf_counter()
            qa = assess_sam_candidates_for_box(
                det_box_xyxy=tuple(float(v) for v in box),
                cand_masks_full=cands_full,
                cand_scores=c_scores,
                img_w=W,
                img_h=H,
                cfg=cfg,
            )
            quality_ms += (time.perf_counter() - t_qa0) * 1000.0

            bbox_full = box_to_bool_mask(tuple(float(v) for v in box), H, W)
            if qa["accepted"] and qa["best_full_mask"] is not None:
                final_mask = qa["best_full_mask"]
                final_rep = "mobilesam_mask"
                union_accepted |= final_mask
            else:
                final_mask = bbox_full
                final_rep = "bbox_fallback"
            union_effective |= final_mask

            proposals_out.append({
                "idx": idx,
                "det_box_xyxy": [round(float(v), 2) for v in box],
                "det_conf": round(float(d_conf), 4),
                "final_representation": final_rep,
                "accepted": qa["accepted"],
                "fallback_used": qa["fallback_used"],
                "rejection_reasons": qa["rejection_reasons"],
                "best_candidate_idx": qa["best_candidate_idx"],
                "num_candidates": qa["num_candidates"],
                "all_candidate_scores": qa["all_candidate_scores"],
                "sam_score": qa["sam_score"],
                "mask_area_px": qa["mask_area_px"],
                "det_box_area_px": qa["det_box_area_px"],
                "area_ratio_to_box": qa["area_ratio_to_box"],
                "frac_inside_det_box": qa["frac_inside_det_box"],
                "det_box_coverage": qa["det_box_coverage"],
                "mask_det_iou": qa["mask_det_iou"],
                "crop_boundary_contact": qa["crop_boundary_contact"],
                "connected_components": qa["connected_components"],
                "final_mask": final_mask,
                "raw_best_mask": qa["best_full_mask"],
            })

        total_ms = det_ms + preprocess_ms + encoder_ms + decoder_ms + reconstruct_ms + quality_ms
        n_acc = sum(1 for p in proposals_out if p["accepted"])
        n_rej = n_prop - n_acc
        n_emp = sum(1 for p in proposals_out if "empty_mask_prediction" in p["rejection_reasons"])

        return {
            "mode": active_mode,
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
                "sam_preprocess_ms": round(preprocess_ms, 3),
                "sam_encoder_ms": round(encoder_ms, 3),
                "sam_decoder_ms": round(decoder_ms, 3),
                "sam_decoder_per_box_ms": round(decoder_ms / max(1, n_prop), 3),
                "mask_reconstruct_ms": round(reconstruct_ms, 3),
                "quality_eval_ms": round(quality_ms, 3),
                "combined_total_ms": round(total_ms, 3),
            },
        }
