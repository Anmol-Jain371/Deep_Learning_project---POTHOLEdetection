# Phase 4A.4 — ROI Mask Quality Evaluator & Diagnostic Report

**Module:** `evaluate_roi_mask_quality()` in [experiments/roi_guided_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/roi_guided_segmentation.py)  
**Benchmark Diagnostics:** [roi_segmentation_comparison.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/roi_segmentation_comparison.json) (`per_proposal_diagnostics`)  
**Disclaimer:** All quality thresholds and rules documented below are **deterministic geometric heuristics**, **NOT calibrated probabilities of pixel accuracy**. Mask acceptance indicates bounded consistency with the enclosing detector box rather than verified boundary IoU or Dice.

---

## 1. Configured Acceptance & Rejection Rules (`ROISegConfig`)

For each ROI crop containing target detector box $D$ and candidate segmentation instances $\{(M_j, s_j)\}$, candidates are ranked by overlap IoU with $D$ (`mask_det_iou`), and the top associated mask $M^*$ is accepted **only if all checks pass**:

| Quality Criterion | Configured Rule | Failure Reason Tag | Physical Failure Prevented |
|---|---|---|---|
| **Empty Output** | `len(cands) > 0` and `mask_area > 0` | `empty_mask_prediction` | Missing mask in ROI |
| **Non-Zero Association** | `|M* ∩ D| > 0` | `no_overlap_with_detector_box` | Unrelated pothole elsewhere in padded crop |
| **Segmentation Confidence** | `seg_conf >= 0.25` | `low_seg_confidence` | Weak spurious mask noise |
| **Minimum Pixel Area** | `mask_area >= 36 px` | `below_min_mask_area` | Tiny degenerate dots |
| **Area Ratio vs. Detector Box** | `0.12 <= area(M*)/area(D) <= 1.35` | `undersized_vs_det_box` / `excessive_area_ratio` | Sliver clips or overflowing background flood |
| **Containment Inside Det Box** | `|M* ∩ D| / area(M*) >= 0.55` | `outside_detector_box_spill` | Masks bleeding into unrelated surrounding road |
| **Detector Box Coverage** | `|M* ∩ D| / area(D) >= 0.15` | `low_det_box_coverage` | Masks covering only a corner crack of `D` |
| **Crop-Boundary Contact** | Padded-border contact ratio `<= 0.30` | `excessive_crop_boundary_contact` | Runaway masks clipped artificially by ROI edge |
| **Connected Fragmentation** | `components (>=4 px) <= 4` | `excessive_fragmentation` | Shattered speckle predictions |
| **Association Ambiguity** | No 2nd valid candidate with `IoU >= 0.20` and `IoU_2 / IoU_1 >= 0.70` | `ambiguous_roi_association` | Split/double mask confusion inside single ROI |

Whenever any rule triggers, $M^*$ and all rejection reasons are logged in `per_proposal_diagnostics`, and the system returns `final_representation = "bbox_fallback"` with the preserved detector bounding box.

---

## 2. Diagnostic Outcomes Across Evaluations

### 2.1 Primary 36-Frame Development Benchmark (`98` Proposals)
- **Accepted ROI Masks:** `11 / 98` (`11.22%`) — mean mask-to-box area ratio `0.6192`, mean fraction inside detector box `0.9065`.
- **Rejected Masks & Bounding-Box Fallbacks:** `87 / 98` (`88.78%`), preserving `98 / 98` (`100%`) detections.
- **Rejection Reason Counts (non-exclusive; a single rejected mask may violate multiple rules):**
  - `empty_mask_prediction`: **`68`** (`69.39%` of all attempts — dominant failure mode)
  - `outside_detector_box_spill`: **`17`** (`17.35%`)
  - `excessive_crop_boundary_contact`: **`9`** (`9.18%`)
  - `excessive_area_ratio`: **`9`** (`9.18%`)
  - `ambiguous_roi_association`: **`2`** (`2.04%`)
  - `low_det_box_coverage`: **`1`** (`1.02%`)
  - `excessive_fragmentation`: **`1`** (`1.02%`)
- **Non-Empty Rejected Geometry (`19` masks):** Mean mask-to-detector-box area ratio was **`1.5518`** (`> 2.5×` higher than accepted masks' `0.6192`), confirming the evaluator cleanly caught overflowing masks before any could render or feed downstream geometry.

### 2.2 Full-Clip Video Runs (`demo_roi_segmentation_video.py`, `batch_roi=True`)
- **Unseen Rural Road (`202` frames, `914` raw proposals):** `34` accepted (`3.72%`), `880` fallbacks (`96.28%`); top reasons: `empty_mask_prediction` (`644`), `outside_detector_box_spill` (`230`), `excessive_crop_boundary_contact` (`155`), `excessive_area_ratio` (`114`), `low_det_box_coverage` (`17`), `ambiguous_roi_association` (`5`), `no_overlap_with_detector_box` (`3`).
- **Rangoon Road (`300` frames, `401` raw proposals):** `94` accepted (`23.44%`), `307` fallbacks (`76.56%`); top reasons: `empty_mask_prediction` (`135`), `outside_detector_box_spill` (`86`), `ambiguous_roi_association` (`80`), `excessive_crop_boundary_contact` (`65`), `excessive_area_ratio` (`45`), `excessive_fragmentation` (`34`), `low_det_box_coverage` (`21`), `no_overlap_with_detector_box` (`16`), `undersized_vs_det_box` (`1`).

---

## 3. Known Limitations of Geometric Quality Gating

1. **Dependence on Detector Box Tightness (`dev_rural_f0075.png`):** Because containment (`frac_inside_det_box`) and area ratio are measured relative to the detector box, if `YOLOv8n` produces an oversized or off-target box (e.g., overlapping grass shoulder), an invalid non-pothole mask inside that box can satisfy all geometric checks and be accepted.
2. **Valid-Pothole Over-Rejection Sensitivity:** Very tight rectangular detector boxes around curved craters can trigger `excessive_crop_boundary_contact` or `outside_detector_box_spill` if a slightly jagged pothole edge extends outside the detector box into the padded margin.
3. **High Empty-Mask Rate (`~69%`):** No geometric post-filter can recover pixel masks when `YOLOv8n-seg` fails to activate on oblique low-contrast dashcam crops; overcoming empty ROI masks requires dashcam domain-adapted segmentation training.
