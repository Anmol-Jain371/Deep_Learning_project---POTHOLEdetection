"""
========================================================================================
RoadVision AI -- Phase 4A.5: Box-Prompted MobileSAM Benchmark & Annotation Assist
========================================================================================
Evaluates and compares across the 36 Phase 4A.4 development frames:
  - A: Existing Phase 2 YOLOv8n detector only
  - B: Phase 4A.4 YOLOv8n + ROI-guided YOLOv8n-seg with quality fallback (batched)
  - C1: YOLOv8n + Box-Prompted MobileSAM (full-frame embedding reuse)
  - C2: YOLOv8n + Box-Prompted MobileSAM (cropped-ROI prompting)

Also:
  - Benchmarks stage-level steady-state latency (excluding startup load time),
    peak GPU VRAM, and scaling vs proposal count (embedding reuse benefit).
  - Generates 4-panel comparative visualizations for all 36 frames.
  - Generates separate UNVERIFIED MobileSAM polygon/mask suggestions for the
    40-frame `dataset_dashcam_seg_dev/` without overwriting existing files.

STRICT RULES:
  - Zero training/fine-tuning; zero evaluation/tuning on Pothole440 grouped test split.
  - Never treats acceptance rate or SAM IoU prediction as segmentation accuracy.
  - Never computes dashcam mask IoU/Dice without verified pixel ground truth.
========================================================================================
"""

import os
import sys
import csv
import json
from collections import Counter
from typing import List, Dict, Any, Tuple

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.roi_guided_segmentation import (
    ROISegConfig,
    ROIGuidedSegmenter,
    DEFAULT_DET_WEIGHTS,
    DEFAULT_SEG_WEIGHTS,
)
from experiments.benchmark_roi_segmentation import (
    build_dev_frame_catalog,
    load_frame,
    compute_latency_stats,
)
from experiments.prompt_guided_segmentation import (
    PromptSegConfig,
    MobileSAMBoxPromptAdapter,
    DEFAULT_MOBILESAM_WEIGHTS,
)
from experiments.prepare_dashcam_annotation_set import mask_to_polygons

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(BASE_DIR, "runs", "benchmark")
VIZ_DIR = os.path.join(BENCH_DIR, "prompt_guided_seg_viz")
DEV_ANN_ROOT = os.path.join(BASE_DIR, "dataset_dashcam_seg_dev")


def draw_panel_masks(
    rgb: np.ndarray,
    proposals: List[Dict[str, Any]],
    union_mask: np.ndarray,
    mask_rgb: List[int],
    contour_rgb: Tuple[int, int, int],
    tag_prefix: str,
    score_key: str,
) -> np.ndarray:
    vis = rgb.copy()
    H, W = vis.shape[:2]
    if union_mask.any():
        tint = vis.copy()
        tint[union_mask] = mask_rgb
        vis = cv2.addWeighted(vis, 0.58, tint, 0.42, 0)
        cnts, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, contour_rgb, 2)

    for p in proposals:
        x1, y1, x2, y2 = [max(0, int(round(v))) for v in p["det_box_xyxy"]]
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if p["accepted"]:
            col = contour_rgb
            lbl = f"{tag_prefix} s:{p[score_key]:.2f}"
        else:
            col = (255, 190, 40)
            reason = p["rejection_reasons"][0].replace("_mask_prediction", "").replace("_", "-")[:14]
            lbl = f"FB:{reason}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, lbl, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.39, col, 1, cv2.LINE_AA)
    return vis


def render_4panel_prompt_comparison(
    entry: Dict[str, Any],
    ts: float,
    img_bgr: np.ndarray,
    roi_res: Dict[str, Any],
    sam_full_res: Dict[str, Any],
    sam_crop_res: Dict[str, Any],
) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    vis_det = rgb.copy()
    for p in roi_res["proposals"]:
        x1, y1, x2, y2 = [int(round(v)) for v in p["det_box_xyxy"]]
        cv2.rectangle(vis_det, (max(0, x1), max(0, y1)), (min(W - 1, x2), min(H - 1, y2)), (34, 197, 94), 2)
        cv2.putText(vis_det, f"{p['det_conf']:.2f}", (max(0, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (34, 197, 94), 1, cv2.LINE_AA)

    vis_roi = draw_panel_masks(
        rgb, roi_res["proposals"], roi_res["union_accepted_mask"], [16, 185, 129], (80, 255, 160), "ROI", "seg_conf"
    )
    vis_sam_f = draw_panel_masks(
        rgb, sam_full_res["proposals"], sam_full_res["union_accepted_mask"], [168, 85, 247], (216, 180, 254), "SAM", "sam_score"
    )
    vis_sam_c = draw_panel_masks(
        rgb, sam_crop_res["proposals"], sam_crop_res["union_accepted_mask"], [59, 130, 246], (147, 197, 253), "SAM-C", "sam_score"
    )

    fig, axes = plt.subplots(1, 4, figsize=(19.2, 4.1), dpi=130)
    ts_str = f"t={ts:.2f}s | " if ts >= 0 else ""
    sr = roi_res["summary"]
    ssf = sam_full_res["summary"]
    ssc = sam_crop_res["summary"]
    panels = [
        (vis_det, f"A: Det Only ({ts_str}n={sr['num_proposals']})"),
        (vis_roi, f"B: ROI YOLOv8n-seg (acc={sr['accepted_masks']}/{sr['num_proposals']}, {roi_res['timing_ms']['combined_total_ms']:.1f}ms)"),
        (vis_sam_f, f"C1: MobileSAM Full (acc={ssf['accepted_masks']}/{ssf['num_proposals']}, {sam_full_res['timing_ms']['combined_total_ms']:.1f}ms)"),
        (vis_sam_c, f"C2: MobileSAM Crop (acc={ssc['accepted_masks']}/{ssc['num_proposals']}, {sam_crop_res['timing_ms']['combined_total_ms']:.1f}ms)"),
    ]
    for ax, (im, ttl) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(ttl, fontsize=8.8)
        ax.axis("off")

    fig.suptitle(f"{entry['frame_id']} | {entry['source_video']} | {entry['notes']}", fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    out_p = os.path.join(VIZ_DIR, f"{entry['frame_id']}.png")
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    return out_p


def summarize_strategy(outputs: List[Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]], name: str, score_key: str) -> Dict[str, Any]:
    tot_prop = tot_acc = tot_rej = tot_empty = tot_fb = 0
    rej_counter = Counter()
    ratios_acc, inside_acc, cov_det_acc, bcontact_acc = [], [], [], []
    ratios_rej = []
    active_frame_covs = []
    excessive_gt35_frames = 0

    for _, img, res in outputs:
        s = res["summary"]
        tot_prop += s["num_proposals"]
        tot_acc += s["accepted_masks"]
        tot_rej += s["rejected_masks"]
        tot_empty += s["empty_masks"]
        tot_fb += s["bbox_fallbacks"]

        acc_cov_pct = float(res["union_accepted_mask"].mean() * 100.0)
        if s["accepted_masks"] > 0:
            active_frame_covs.append(acc_cov_pct)
        if acc_cov_pct > 35.0:
            excessive_gt35_frames += 1

        for p in res["proposals"]:
            if p["accepted"]:
                ratios_acc.append(p["area_ratio_to_box"])
                inside_acc.append(p["frac_inside_det_box"])
                cov_det_acc.append(p["det_box_coverage"])
                bcontact_acc.append(p["crop_boundary_contact"])
            else:
                if p["area_ratio_to_box"] > 0:
                    ratios_rej.append(p["area_ratio_to_box"])
                for r in p["rejection_reasons"]:
                    rej_counter[r] += 1

    return {
        "strategy_name": name,
        "total_proposals": tot_prop,
        "detection_preservation_rate": 1.0,
        "accepted_masks": tot_acc,
        "acceptance_rate_pct": round(100.0 * tot_acc / max(1, tot_prop), 2),
        "rejected_masks": tot_rej,
        "rejection_rate_pct": round(100.0 * tot_rej / max(1, tot_prop), 2),
        "empty_masks": tot_empty,
        "empty_mask_rate_pct": round(100.0 * tot_empty / max(1, tot_prop), 2),
        "bbox_fallbacks": tot_fb,
        "fallback_rate_pct": round(100.0 * tot_fb / max(1, tot_prop), 2),
        "excessive_accepted_cov_gt35pct_frames": excessive_gt35_frames,
        "mean_accepted_frame_cov_when_active_pct": round(float(np.mean(active_frame_covs)), 3) if active_frame_covs else 0.0,
        "accepted_geometry": {
            "mean_area_ratio_to_box": round(float(np.mean(ratios_acc)), 4) if ratios_acc else 0.0,
            "mean_containment_inside_box": round(float(np.mean(inside_acc)), 4) if inside_acc else 0.0,
            "mean_det_box_coverage": round(float(np.mean(cov_det_acc)), 4) if cov_det_acc else 0.0,
            "mean_boundary_contact": round(float(np.mean(bcontact_acc)), 4) if bcontact_acc else 0.0,
        },
        "rejected_nonempty_count": len(ratios_rej),
        "rejected_nonempty_mean_area_ratio": round(float(np.mean(ratios_rej)), 4) if ratios_rej else 0.0,
        "rejection_reason_counts": dict(rej_counter),
    }


def bucket_timing_scaling(timings_list: List[Dict[str, Any]], seg_key: str) -> Dict[str, Any]:
    buckets = {"0": [], "1": [], "2": [], "3": [], "4+": []}
    for row in timings_list:
        key = "4+" if row["n_prop"] >= 4 else str(row["n_prop"])
        buckets[key].append(row)
    out = {}
    for k, rows in buckets.items():
        if not rows:
            continue
        out[k] = {
            "samples": len(rows),
            "mean_n_prop": round(float(np.mean([r["n_prop"] for r in rows])), 2),
            "combined": compute_latency_stats([r["combined_total_ms"] for r in rows]),
            "seg_component": compute_latency_stats([r[seg_key] for r in rows]),
        }
    return out


def generate_dashcam_dev_mobilesam_suggestions(sam_adapter: MobileSAMBoxPromptAdapter) -> Dict[str, Any]:
    """Generate separate UNVERIFIED MobileSAM suggestions for the 40-frame dev set."""
    manifest_csv = os.path.join(DEV_ANN_ROOT, "annotation_manifest.csv")
    rows = list(csv.DictReader(open(manifest_csv, "r", encoding="utf-8")))
    sug_root = os.path.join(DEV_ANN_ROOT, "unverified_mobilesam_suggestions")
    os.makedirs(sug_root, exist_ok=True)

    sug_rows = []
    tot_props = tot_acc = tot_fb = 0

    for row in rows:
        sid = row["sample_id"]
        sess = row["session_id"]
        img_p = os.path.join(DEV_ANN_ROOT, row["frame_path"])
        img_bgr = cv2.imread(img_p)
        H, W = img_bgr.shape[:2]

        out = sam_adapter.process_frame(img_bgr, mode="full_frame_embed")
        s = out["summary"]
        tot_props += s["num_proposals"]
        tot_acc += s["accepted_masks"]
        tot_fb += s["bbox_fallbacks"]

        sess_dir = os.path.join(sug_root, sess)
        os.makedirs(sess_dir, exist_ok=True)
        mask_rel = f"unverified_mobilesam_suggestions/{sess}/{sid}_mobilesam_suggested_mask.png"
        json_rel = f"unverified_mobilesam_suggestions/{sess}/{sid}_mobilesam_suggestions.json"

        cv2.imwrite(os.path.join(DEV_ANN_ROOT, mask_rel), (out["union_accepted_mask"].astype(np.uint8) * 255))

        prop_sug = []
        for p in out["proposals"]:
            polys = mask_to_polygons(p["final_mask"]) if p["accepted"] else []
            prop_sug.append({
                "proposal_idx": p["idx"],
                "det_box_xyxy": p["det_box_xyxy"],
                "det_conf": p["det_conf"],
                "mobilesam_accepted": p["accepted"],
                "rejection_reasons": p["rejection_reasons"],
                "sam_iou_pred": p["sam_score"],
                "unverified_polygon_px": polys,
                "label_status": "UNVERIFIED_MODEL_SUGGESTION",
            })

        with open(os.path.join(DEV_ANN_ROOT, json_rel), "w", encoding="utf-8") as f:
            json.dump({
                "sample_id": sid,
                "session_id": sess,
                "source_video": row["source_video"],
                "frame_number": int(row["frame_number"]),
                "width": W,
                "height": H,
                "label_status": "UNVERIFIED_MODEL_SUGGESTION",
                "verified_ground_truth": False,
                "warning": "Optional MobileSAM mask suggestions only; requires human inspection of ALL visible potholes, detector misses, and negative frames.",
                "proposals": prop_sug,
            }, f, indent=2)

        sug_rows.append({
            "sample_id": sid,
            "session_id": sess,
            "source_video": row["source_video"],
            "frame_number": row["frame_number"],
            "mobilesam_suggestions_json": json_rel,
            "mobilesam_suggested_mask_png": mask_rel,
            "num_detector_proposals": s["num_proposals"],
            "num_mobilesam_accepted": s["accepted_masks"],
            "num_mobilesam_fallbacks": s["bbox_fallbacks"],
            "label_status": "UNVERIFIED_MODEL_SUGGESTION",
            "verified_ground_truth": False,
        })

    out_csv = os.path.join(DEV_ANN_ROOT, "mobilesam_suggestions_manifest.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sug_rows[0].keys()))
        w.writeheader()
        w.writerows(sug_rows)

    return {
        "frames_processed": len(sug_rows),
        "total_detector_proposals": tot_props,
        "mobilesam_accepted_suggestions": tot_acc,
        "mobilesam_fallbacks": tot_fb,
        "acceptance_rate_pct": round(100.0 * tot_acc / max(1, tot_props), 2),
        "manifest_csv": out_csv,
    }


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    catalog = build_dev_frame_catalog()
    frames_loaded = [(e, lf[1], lf[0]) for e in catalog for lf in [load_frame(e)]]  # (entry, ts, img_bgr)

    # Initialize both segmentation adapters (startup tracked separately)
    roi_seg = ROIGuidedSegmenter(DEFAULT_DET_WEIGHTS, DEFAULT_SEG_WEIGHTS, config=ROISegConfig(batch_roi=True))
    roi_seg.warmup(360, 640)

    sam_adapter = MobileSAMBoxPromptAdapter(
        DEFAULT_DET_WEIGHTS, DEFAULT_MOBILESAM_WEIGHTS, config=PromptSegConfig(mode="full_frame_embed")
    )
    sam_adapter.warmup(360, 640)

    def reset_vram():
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)

    def read_vram():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return (
                round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
                round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2),
            )
        return 0.0, 0.0

    # 1. Strategy B: ROI-guided YOLOv8n-seg (batched) across 36 frames
    reset_vram()
    roi_outputs = []
    roi_timings = []
    for entry, ts, img in frames_loaded:
        res_r = roi_seg.process_frame(img)
        roi_outputs.append((entry, img, res_r))
        roi_timings.append({"n_prop": res_r["summary"]["num_proposals"], **res_r["timing_ms"]})
    vram_roi_alloc, vram_roi_res = read_vram()

    # 2. Strategy C1: MobileSAM Full-Frame Embedding Reuse (3 passes over 36 frames = 108 timed passes)
    reset_vram()
    sam_full_outputs = []
    sam_full_timings = []
    for loop_i in range(3):
        for entry, ts, img in frames_loaded:
            res_sf = sam_adapter.process_frame(img, mode="full_frame_embed")
            sam_full_timings.append({"n_prop": res_sf["summary"]["num_proposals"], **res_sf["timing_ms"]})
            if loop_i == 0:
                sam_full_outputs.append((entry, img, res_sf))
    vram_sf_alloc, vram_sf_res = read_vram()

    # 3. Strategy C2: MobileSAM Cropped-ROI Prompting (1 full pass over 36 frames)
    reset_vram()
    sam_crop_outputs = []
    sam_crop_timings = []
    for entry, ts, img in frames_loaded:
        res_sc = sam_adapter.process_frame(img, mode="cropped_roi_prompt")
        sam_crop_outputs.append((entry, img, res_sc))
        sam_crop_timings.append({"n_prop": res_sc["summary"]["num_proposals"], **res_sc["timing_ms"]})
    vram_sc_alloc, vram_sc_res = read_vram()

    # 4. Render 36 4-panel visualizations & save per-frame CSV
    per_frame_rows = []
    all_proposals_compare = []
    for (entry, ts, img), (_, _, r_roi), (_, _, r_sf), (_, _, r_sc) in zip(
        frames_loaded, roi_outputs, sam_full_outputs, sam_crop_outputs
    ):
        viz_p = render_4panel_prompt_comparison(entry, ts, img, r_roi, r_sf, r_sc)
        sr, ssf, ssc = r_roi["summary"], r_sf["summary"], r_sc["summary"]
        tr, tsf, tsc = r_roi["timing_ms"], r_sf["timing_ms"], r_sc["timing_ms"]

        per_frame_rows.append({
            "frame_id": entry["frame_id"],
            "source_video": entry["source_video"],
            "session_id": entry["session_id"],
            "frame_number": entry["frame_number"],
            "timestamp_sec": ts,
            "num_proposals": sr["num_proposals"],
            "roi_seg_accepted": sr["accepted_masks"],
            "roi_seg_empty": sr["empty_masks"],
            "roi_seg_fallbacks": sr["bbox_fallbacks"],
            "roi_seg_cov_pct": round(float(r_roi["union_accepted_mask"].mean() * 100.0), 3),
            "roi_seg_ms": tr["combined_total_ms"],
            "sam_full_accepted": ssf["accepted_masks"],
            "sam_full_empty": ssf["empty_masks"],
            "sam_full_fallbacks": ssf["bbox_fallbacks"],
            "sam_full_cov_pct": round(float(r_sf["union_accepted_mask"].mean() * 100.0), 3),
            "sam_full_det_ms": tsf["detector_ms"],
            "sam_full_pre_ms": tsf["sam_preprocess_ms"],
            "sam_full_enc_ms": tsf["sam_encoder_ms"],
            "sam_full_dec_ms": tsf["sam_decoder_ms"],
            "sam_full_rec_ms": tsf["mask_reconstruct_ms"],
            "sam_full_qa_ms": tsf["quality_eval_ms"],
            "sam_full_total_ms": tsf["combined_total_ms"],
            "sam_crop_accepted": ssc["accepted_masks"],
            "sam_crop_fallbacks": ssc["bbox_fallbacks"],
            "sam_crop_cov_pct": round(float(r_sc["union_accepted_mask"].mean() * 100.0), 3),
            "sam_crop_total_ms": tsc["combined_total_ms"],
            "viz_path": viz_p,
        })

        for pr, psf, psc in zip(r_roi["proposals"], r_sf["proposals"], r_sc["proposals"]):
            all_proposals_compare.append({
                "frame_id": entry["frame_id"],
                "session_id": entry["session_id"],
                "proposal_idx": pr["idx"],
                "det_conf": pr["det_conf"],
                "roi_accepted": pr["accepted"],
                "roi_reasons": pr["rejection_reasons"],
                "roi_area_ratio": pr["area_ratio_to_box"],
                "sam_full_accepted": psf["accepted"],
                "sam_full_reasons": psf["rejection_reasons"],
                "sam_full_score": psf["sam_score"],
                "sam_full_area_ratio": psf["area_ratio_to_box"],
                "sam_full_inside": psf["frac_inside_det_box"],
                "sam_crop_accepted": psc["accepted"],
                "sam_crop_reasons": psc["rejection_reasons"],
                "sam_crop_score": psc["sam_score"],
            })

    csv_path = os.path.join(BENCH_DIR, "prompt_guided_segmentation_per_frame.csv")
    csv_cols = [k for k in per_frame_rows[0].keys() if k != "viz_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in per_frame_rows:
            w.writerow({k: r[k] for k in csv_cols})

    # 5. Annotation-assisted suggestions on 40-frame dataset_dashcam_seg_dev
    ann_summary = generate_dashcam_dev_mobilesam_suggestions(sam_adapter)

    # 6. Timing statistics
    sam_full_nonempty_dec = [r["sam_decoder_ms"] for r in sam_full_timings if r["n_prop"] > 0]
    sam_full_per_box_dec = [r["sam_decoder_per_box_ms"] for r in sam_full_timings if r["n_prop"] > 0]
    sam_full_nonempty_enc = [r["sam_encoder_ms"] for r in sam_full_timings if r["n_prop"] > 0]

    results_json = {
        "experiment": "Phase 4A.5 Box-Prompted MobileSAM Feasibility Study",
        "startup_load_ms_separate": sam_adapter.startup_load_ms,
        "sam_config": {k: v for k, v in vars(sam_adapter.cfg).items()},
        "dev_frames_count": len(per_frame_rows),
        "strategy_summaries": {
            "B_roi_guided_yolov8n_seg": summarize_strategy(roi_outputs, "ROI-Guided YOLOv8n-seg", "seg_conf"),
            "C1_mobilesam_full_frame_embed": summarize_strategy(sam_full_outputs, "Box-Prompted MobileSAM (Full-Frame Embed)", "sam_score"),
            "C2_mobilesam_cropped_roi_prompt": summarize_strategy(sam_crop_outputs, "Box-Prompted MobileSAM (Cropped ROI)", "sam_score"),
        },
        "performance_rtx2050": {
            "C1_mobilesam_full_frame_embed": {
                "samples": len(sam_full_timings),
                "detector": compute_latency_stats([r["detector_ms"] for r in sam_full_timings]),
                "sam_preprocess": compute_latency_stats([r["sam_preprocess_ms"] for r in sam_full_timings]),
                "sam_encoder_all_frames": compute_latency_stats([r["sam_encoder_ms"] for r in sam_full_timings]),
                "sam_encoder_when_active": compute_latency_stats(sam_full_nonempty_enc),
                "sam_decoder_when_active": compute_latency_stats(sam_full_nonempty_dec),
                "sam_decoder_per_box_when_active": compute_latency_stats(sam_full_per_box_dec),
                "mask_reconstruct": compute_latency_stats([r["mask_reconstruct_ms"] for r in sam_full_timings]),
                "quality_eval": compute_latency_stats([r["quality_eval_ms"] for r in sam_full_timings]),
                "combined_total": compute_latency_stats([r["combined_total_ms"] for r in sam_full_timings]),
                "peak_gpu_allocated_mb": vram_sf_alloc,
                "peak_gpu_reserved_mb": vram_sf_res,
                "scaling_by_num_proposals": bucket_timing_scaling(sam_full_timings, "sam_encoder_ms"),
            },
            "C2_mobilesam_cropped_roi_prompt": {
                "samples": len(sam_crop_timings),
                "sam_encoder_all_frames": compute_latency_stats([r["sam_encoder_ms"] for r in sam_crop_timings]),
                "sam_decoder_all_frames": compute_latency_stats([r["sam_decoder_ms"] for r in sam_crop_timings]),
                "combined_total": compute_latency_stats([r["combined_total_ms"] for r in sam_crop_timings]),
                "peak_gpu_allocated_mb": vram_sc_alloc,
                "peak_gpu_reserved_mb": vram_sc_res,
                "scaling_by_num_proposals": bucket_timing_scaling(sam_crop_timings, "sam_encoder_ms"),
            },
            "B_roi_guided_yolov8n_seg_batched": {
                "samples": len(roi_timings),
                "combined_total": compute_latency_stats([r["combined_total_ms"] for r in roi_timings]),
                "peak_gpu_allocated_mb": vram_roi_alloc,
                "peak_gpu_reserved_mb": vram_roi_res,
            },
        },
        "annotation_assist_40_frames": ann_summary,
        "per_frame_summary": per_frame_rows,
        "per_proposal_comparison": all_proposals_compare,
    }

    json_path = os.path.join(BENCH_DIR, "prompt_guided_segmentation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print("=== PHASE 4A.5 BENCHMARK COMPLETE ===")
    print("Startup load ms (excluded from steady state):", sam_adapter.startup_load_ms)
    for k, v in results_json["strategy_summaries"].items():
        print(f"  {k}: acc={v['accepted_masks']}/{v['total_proposals']} ({v['acceptance_rate_pct']}%), empty={v['empty_masks']}, fb={v['bbox_fallbacks']}")
    print("  C1 Full-Frame MobileSAM Latency:", results_json["performance_rtx2050"]["C1_mobilesam_full_frame_embed"]["combined_total"])
    print("  C2 Cropped-ROI MobileSAM Latency:", results_json["performance_rtx2050"]["C2_mobilesam_cropped_roi_prompt"]["combined_total"])
    print("  Annotation Assist 40 frames:", ann_summary)


if __name__ == "__main__":
    main()
