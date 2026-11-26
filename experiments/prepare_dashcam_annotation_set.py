"""
========================================================================================
RoadVision AI -- Phase 4A.4: Development Dashcam Segmentation Annotation Set Prep
========================================================================================
Creates `dataset_dashcam_seg_dev/` with 40 diverse representative frames organized strictly
by video/capture session to prevent split leakage, along with detector proposals and
optional ROI-accepted polygon/mask suggestions for manual human verification.

STRICT RULES:
  - All generated boxes/masks are explicitly marked UNVERIFIED SUGGESTIONS.
  - NEVER used as ground truth without human polygon review.
  - Recommend a completely separate untouched final dashcam evaluation video set.
========================================================================================
"""

import os
import sys
import csv
import json
from typing import List, Dict, Any

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.roi_guided_segmentation import (
    ROISegConfig,
    ROIGuidedSegmenter,
    DEFAULT_DET_WEIGHTS,
    DEFAULT_SEG_WEIGHTS,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT = os.path.join(BASE_DIR, "dataset_dashcam_seg_dev")

RANGOON_V2_INDICES = {
    0, 40, 61, 100, 120, 143, 182, 211, 255, 284, 313, 321, 370, 401, 422, 454, 481, 488,
    522, 549, 588, 625, 641, 676, 715, 721, 760, 803, 806, 844, 883, 892, 922, 963, 1002
}


def select_40_frames() -> List[Dict[str, Any]]:
    items = []
    # 1. rural_road_unseen (12 frames)
    for fnum in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 180]:
        items.append({
            "sample_id": f"rural_unseen_f{fnum:04d}",
            "session_id": "rural_road_unseen",
            "source_video": "uploads/upload_1794847200_ruralRoad_potHoles.mp4",
            "frame_number": fnum,
            "prior_exposure": "UNSEEN_BY_DETECTOR_AND_SEG",
        })
    # 2. rangoon_road_unseen_temporal (14 frames outside dataset_v2_candidates)
    for fnum in [20, 80, 160, 230, 270, 340, 440, 505, 605, 655, 695, 740, 780, 860]:
        assert fnum not in RANGOON_V2_INDICES
        items.append({
            "sample_id": f"rangoon_unseen_f{fnum:04d}",
            "session_id": "rangoon_road_unseen_temporal",
            "source_video": "Watch Video_ Terrible potholes in Rangoon Road.mp4",
            "frame_number": fnum,
            "prior_exposure": "UNSEEN_TEMPORAL_FRAME_SINGLE_VIDEO",
        })
    # 3. raw_survey_video (8 frames)
    for fnum in [0, 20, 40, 60, 80, 100, 120, 140]:
        items.append({
            "sample_id": f"survey_f{fnum:04d}",
            "session_id": "raw_survey_video",
            "source_video": "roadvision-dashboard/public/videos/raw_survey.mp4",
            "frame_number": fnum,
            "prior_exposure": "UNSEEN_RAW_FRAMES_SURVEY_CLIP",
        })
    # 4. phone_val_session (6 frames — explicitly marked detector-exposed)
    for fn, fnum in [
        ("frame_0005.jpg", 5), ("frame_0014.jpg", 14), ("frame_0027.jpg", 27),
        ("frame_0028.jpg", 28), ("frame_0035.jpg", 35), ("frame_0043.jpg", 43),
    ]:
        items.append({
            "sample_id": f"phone_val_{os.path.splitext(fn)[0]}",
            "session_id": "phone_val_session",
            "source_video": f"dataset_phone_potholes/images/val/{fn}",
            "frame_number": fnum,
            "prior_exposure": "DETECTOR_UNIFIED_POOL_PROVENANCE",
        })
    return items


def mask_to_polygons(mask_bool: np.ndarray, min_area: float = 20.0) -> List[List[List[int]]]:
    cnts, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area or len(c) < 3:
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) >= 3:
            polys.append(approx.reshape(-1, 2).tolist())
    return polys


def main():
    items = select_40_frames()
    cfg = ROISegConfig(batch_roi=True)
    seg_sys = ROIGuidedSegmenter(DEFAULT_DET_WEIGHTS, DEFAULT_SEG_WEIGHTS, config=cfg)
    seg_sys.warmup(360, 640)

    manifest_rows = []
    for it in items:
        src = os.path.join(BASE_DIR, it["source_video"])
        if src.lower().endswith(".mp4"):
            cap = cv2.VideoCapture(src)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, it["frame_number"])
            ok, img = cap.read()
            cap.release()
            if not ok or img is None:
                raise RuntimeError(f"Failed reading {src} frame {it['frame_number']}")
            ts = round(it["frame_number"] / float(fps), 3)
        else:
            img = cv2.imread(src)
            ts = None

        H, W = img.shape[:2]
        sess = it["session_id"]
        sid = it["sample_id"]

        frames_dir = os.path.join(OUT_ROOT, "frames", sess)
        prop_dir = os.path.join(OUT_ROOT, "unverified_proposals", sess)
        sug_dir = os.path.join(OUT_ROOT, "unverified_mask_suggestions", sess)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(prop_dir, exist_ok=True)
        os.makedirs(sug_dir, exist_ok=True)

        frame_rel = f"frames/{sess}/{sid}.png"
        sug_mask_rel = f"unverified_mask_suggestions/{sess}/{sid}_suggested_mask.png"
        prop_rel = f"unverified_proposals/{sess}/{sid}_proposals.json"

        cv2.imwrite(os.path.join(OUT_ROOT, frame_rel), img)
        res = seg_sys.process_frame(img)

        # Save binary suggested mask image (ONLY accepted ROI masks; zero where fallback/empty)
        cv2.imwrite(os.path.join(OUT_ROOT, sug_mask_rel), (res["union_accepted_mask"].astype(np.uint8) * 255))

        prop_entries = []
        for p in res["proposals"]:
            polys = mask_to_polygons(p["final_mask"]) if p["accepted"] else []
            prop_entries.append({
                "proposal_idx": p["idx"],
                "det_box_xyxy": p["det_box_xyxy"],
                "det_conf": p["det_conf"],
                "roi_seg_suggestion_accepted": p["accepted"],
                "rejection_reasons": p["rejection_reasons"],
                "seg_conf": p["seg_conf"],
                "unverified_polygon_px": polys,
                "status": "UNVERIFIED_MODEL_SUGGESTION_DO_NOT_USE_AS_GT",
            })

        with open(os.path.join(OUT_ROOT, prop_rel), "w", encoding="utf-8") as f:
            json.dump({
                "sample_id": sid,
                "session_id": sess,
                "source_video": it["source_video"],
                "frame_number": it["frame_number"],
                "timestamp_sec": ts,
                "width": W,
                "height": H,
                "verified_ground_truth": False,
                "warning": "Model-generated proposals & ROI mask suggestions require full manual verification before use.",
                "proposals": prop_entries,
            }, f, indent=2)

        s = res["summary"]
        manifest_rows.append({
            "sample_id": sid,
            "session_id": sess,
            "source_video": it["source_video"],
            "frame_number": it["frame_number"],
            "timestamp_sec": "" if ts is None else ts,
            "width": W,
            "height": H,
            "prior_exposure": it["prior_exposure"],
            "frame_path": frame_rel,
            "proposals_json_path": prop_rel,
            "suggested_mask_path": sug_mask_rel,
            "num_detector_proposals": s["num_proposals"],
            "num_roi_seg_suggestions": s["accepted_masks"],
            "num_bbox_fallbacks": s["bbox_fallbacks"],
            "mask_verification_status": "PENDING_MANUAL_POLYGON_MASK",
            "verified_ground_truth": False,
            "recommended_split_role": "DEV_ANNOTATION_ONLY",
        })

    csv_p = os.path.join(OUT_ROOT, "annotation_manifest.csv")
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    json_p = os.path.join(OUT_ROOT, "annotation_manifest.json")
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": "dataset_dashcam_seg_dev",
            "purpose": "Development-only dashcam segmentation annotation prep (NOT final test benchmark)",
            "total_frames": len(manifest_rows),
            "sessions": sorted(list({r["session_id"] for r in manifest_rows})),
            "warning": "All proposals & masks are UNVERIFIED suggestions. Separate unseen videos must be reserved for final testing.",
            "frames": manifest_rows,
        }, f, indent=2)

    print(f"Prepared {len(manifest_rows)} frames in {OUT_ROOT}")


if __name__ == "__main__":
    main()
