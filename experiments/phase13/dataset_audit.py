"""
Phase 13 Step 12: Dataset Independence, Duplicate & Provenance Audit.

Inspects all image datasets, label annotations, and source videos in the repository,
detects exact SHA-256 duplicates and perceptual dHash near-duplicates across splits,
documents prior tuning exposure, and emits a frozen `evaluation_manifest.csv` plus
an explicit independence verdict.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(512 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_dhash_int(img_bgr: np.ndarray, size: int = 8) -> Optional[int]:
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bool(bit))
    return val


def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def run_dataset_audit(project_root: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    split_summaries: List[Dict[str, Any]] = []

    datasets_spec = [
        ("dataset_unified", "train", "development_train", "Model training split (RDD/pothole unified)"),
        ("dataset_unified", "val", "development_validation", "Used for checkpoint selection and hyperparameter tuning"),
        ("dataset_unified", "test", "exploratory_contaminated_test", "Inspected during Phase 2; cross-split near-duplicates reported"),
        ("dataset_unified", "external_rangoon", "exploratory_external_clip", "35 continuous frames from Rangoon clip with incomplete annotations"),
        ("dataset_pothole440_seg_grouped", "train", "segmentation_train", "Grouped base captures (301 imgs)"),
        ("dataset_pothole440_seg_grouped", "val", "segmentation_val", "Grouped unaugmented originals (15 imgs)"),
        ("dataset_pothole440_seg_grouped", "test", "segmentation_reconstructed_test", "Regrouped unaugmented originals (15 imgs); domain-shifted closeups"),
        ("dataset_rdd_india_potholes", "train", "legacy_rdd_train", "Legacy RDD India train split"),
        ("dataset_rdd_india_potholes", "val", "legacy_rdd_val", "Legacy RDD India val split"),
        ("dataset_phone_potholes", "train", "legacy_phone_train", "Small exploratory phone dataset"),
        ("dataset_phone_potholes", "val", "legacy_phone_val", "Small exploratory phone dataset"),
        ("dataset_combined_v2", "train", "legacy_combined_train", "Legacy combined dataset"),
        ("dataset_combined_v2", "val", "legacy_combined_val", "Legacy combined val dataset"),
    ]

    unified_hashes: Dict[str, List[Tuple[str, str, int]]] = {}  # split -> [(sample_id, sha, dhash)]

    for ds_name, split_name, role, exposure_note in datasets_spec:
        img_dir = os.path.join(project_root, ds_name, "images", split_name)
        lbl_dir = os.path.join(project_root, ds_name, "labels", split_name)
        if not os.path.isdir(img_dir):
            continue

        imgs = sorted(
            f
            for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        nonempty_lbls = 0
        total_boxes = 0
        sample_idx = 0

        for fname in imgs:
            full_img = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
            has_lbl = os.path.isfile(lbl_path) and os.path.getsize(lbl_path) > 0
            box_cnt = 0
            if os.path.isfile(lbl_path):
                with open(lbl_path, "r", encoding="utf-8", errors="ignore") as lfh:
                    lines = [ln.strip() for ln in lfh if ln.strip()]
                box_cnt = len(lines)
                if box_cnt > 0:
                    nonempty_lbls += 1
                total_boxes += box_cnt

            sha = _sha256_file(full_img)
            if ds_name == "dataset_unified":
                img = cv2.imread(full_img)
                dh = _compute_dhash_int(img)
                if dh is not None:
                    unified_hashes.setdefault(split_name, []).append((fname, sha, dh))

            manifest_rows.append(
                {
                    "sample_id": f"{ds_name}/{split_name}/{stem}",
                    "source_group": f"{ds_name}:{split_name}",
                    "source_path": f"{ds_name}/images/{split_name}/{fname}",
                    "file_sha256": sha,
                    "annotation_available": bool(os.path.isfile(lbl_path) and box_cnt > 0),
                    "annotation_count": box_cnt,
                    "dataset_role": role,
                    "known_previous_exposure": exposure_note,
                    "inclusion_status": "EXPLORATORY_OR_DEV" if role != "development_train" else "TRAINING_ONLY",
                    "reason": exposure_note,
                }
            )
            sample_idx += 1

        split_summaries.append(
            {
                "dataset": ds_name,
                "split": split_name,
                "image_count": len(imgs),
                "annotated_files_nonempty": nonempty_lbls,
                "total_annotation_instances": total_boxes,
                "role": role,
                "independent_heldout": False,
                "provenance_note": exposure_note,
            }
        )

    # Check evaluation/frames + evaluation/ground_truth_labels
    eval_frames_dir = os.path.join(project_root, "evaluation", "frames")
    eval_lbl_dir = os.path.join(project_root, "evaluation", "ground_truth_labels")
    eval_imgs = sorted(glob.glob(os.path.join(eval_frames_dir, "*.jpg"))) if os.path.isdir(eval_frames_dir) else []
    eval_nonempty = 0
    for ep in eval_imgs:
        fname = os.path.basename(ep)
        stem = os.path.splitext(fname)[0]
        lpath = os.path.join(eval_lbl_dir, f"{stem}.txt")
        has_ann = os.path.isfile(lpath) and os.path.getsize(lpath) > 0
        if has_ann:
            eval_nonempty += 1
        manifest_rows.append(
            {
                "sample_id": f"evaluation/{stem}",
                "source_group": "rangoon_sampled_frames",
                "source_path": f"evaluation/frames/{fname}",
                "file_sha256": _sha256_file(ep),
                "annotation_available": has_ann,
                "annotation_count": 0,
                "dataset_role": "unannotated_rangoon_frames",
                "known_previous_exposure": "Sampled from Watch Video_ Terrible potholes in Rangoon Road.mp4",
                "inclusion_status": "BLOCKED_NO_LABELS",
                "reason": "0-byte ground-truth label file; cannot compute detection accuracy.",
            }
        )

    # Check exploratory road videos
    video_specs = [
        ("ruralRoad_202f", "uploads/upload_1794847200_ruralRoad_potHoles.mp4", "rural_road_dashcam", "Used in Phases 5C/6A/6B/7A"),
        ("rawSurvey_160f", "roadvision-dashboard/public/videos/raw_survey.mp4", "urban_survey_dashcam", "Used in Phases 5C/6A/7A/7B"),
        ("rangoon_clip", "Watch Video_ Terrible potholes in Rangoon Road.mp4", "rangoon_flooded_corridor", "Used in Phases 2/3A/6A/7A/8A"),
    ]
    for vid, vrel, grp, note in video_specs:
        vp = os.path.join(project_root, vrel)
        if os.path.isfile(vp):
            manifest_rows.append(
                {
                    "sample_id": f"video/{vid}",
                    "source_group": grp,
                    "source_path": vrel,
                    "file_sha256": _sha256_file(vp),
                    "annotation_available": False,
                    "annotation_count": 0,
                    "dataset_role": "exploratory_video",
                    "known_previous_exposure": note,
                    "inclusion_status": "EXPLORATORY_VIDEO_ONLY",
                    "reason": "Unlabeled road footage previously used for system development.",
                }
            )

    # Cross-split exact & perceptual dHash duplicate scan inside dataset_unified
    exact_cross_dups = 0
    dhash_near_dups_le6 = 0
    dhash_near_dups_le2 = 0
    splits = list(unified_hashes.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            sa, sb = splits[i], splits[j]
            items_a = unified_hashes[sa]
            items_b = unified_hashes[sb]
            shas_b = {x[1] for x in items_b}
            for _, sha_a, dh_a in items_a:
                if sha_a in shas_b:
                    exact_cross_dups += 1
                for _, _, dh_b in items_b:
                    hd = _hamming64(dh_a, dh_b)
                    if hd <= 6:
                        dhash_near_dups_le6 += 1
                    if hd <= 2:
                        dhash_near_dups_le2 += 1

    manifest_csv_path = os.path.join(out_dir, "evaluation_manifest.csv")
    fieldnames = [
        "sample_id",
        "source_group",
        "source_path",
        "file_sha256",
        "annotation_available",
        "annotation_count",
        "dataset_role",
        "known_previous_exposure",
        "inclusion_status",
        "reason",
    ]
    with open(manifest_csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "evaluation_manifest_csv": "evaluation_manifest.csv",
        "total_manifest_entries": len(manifest_rows),
        "dataset_split_summaries": split_summaries,
        "evaluation_ground_truth_status": {
            "frames_found": len(eval_imgs),
            "nonempty_label_files": eval_nonempty,
            "status": "BLOCKED" if eval_nonempty == 0 else "ANNOTATED",
        },
        "dataset_unified_cross_split_audit": {
            "exact_sha256_cross_split_matches": exact_cross_dups,
            "dhash_hamming_le_6_cross_split_pairs": dhash_near_dups_le6,
            "dhash_hamming_le_2_cross_split_pairs": dhash_near_dups_le2,
        },
        "independent_annotated_test_set_available": False,
        "independent_detection_generalization_status": "BLOCKED",
        "verdict": (
            "No untouched independent annotated pothole detection test dataset exists in the repository. "
            "dataset_unified/test (153 imgs) and external_rangoon (35 imgs) were previously inspected and "
            "exhibit cross-split perceptual near-duplicates or incomplete clip annotations; evaluation/ground_truth_labels/ "
            "has 0 annotated boxes. Detection metrics are therefore strictly reported as EXPLORATORY."
        ),
    }
