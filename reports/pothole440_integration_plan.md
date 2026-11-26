# Phase 3B: Pothole440 Safe Integration Plan & Final Audit Decision

---

## 1. Final Audit Conclusion (Evidence-Based Selection)

Based strictly on the empirical evidence gathered in [`dataset_inventory.json`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_inspection/dataset_inventory.json), [`pothole440_dataset_audit.md`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/pothole440_dataset_audit.md), and [`pothole440_depth_encoding_report.md`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/pothole440_depth_encoding_report.md):

> **Selected Conclusion: C — Segmentation is feasible, but depth validation requires additional metadata.**

*(Specifically: pixel-wise segmentation masks are clean and convertible to YOLO instance segmentation after base-ID de-duplication, whereas **metric depth validation is infeasible** without scanner scale/intrinsics/volume metadata, and even **relative depth evaluation** is heavily constrained to the valid non-padded Go!SCAN patch because `99.5%` of depth images are `79.8%`-median padded with sample-varying constants `1–147`, `186` masks overlap padding, and `34/40` official test images share base captures with `train`.)*

---

## 2. Strict Integration Guardrails

1. **Do NOT merge `Pothole440` into `dataset_unified`:** Keep RoadVision's forward-dashcam RDD2022/phone training domain isolated from top-down close-up macro captures.
2. **Do NOT use the official `Pothole440/test` split as-is:** Because `34` of `40` test images share base pothole IDs with `train` (`43` shared between `train` and `val`), official split metrics reflect augmentation memorization.
3. **Do NOT attempt `k_disp` or `depth_exponent` calibration on Pothole440:** `MetricDepthCalibrator` pinhole assumptions (`H_cam`, `VFOV`, `horizon_ratio`) do not hold for top-down Go!SCAN captures and Pothole440 lacks metric `cm`/`mm` units.
4. **Respect Academic Licensing:** Keep `/Pothole440-main/` excluded in [`.gitignore`](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/.gitignore), restrict usage to non-commercial research, and cite Ali et al. (IEEE TII 2026, `DOI: 10.1109/TII.2026.3706348`) if any experiments use it.

---

## 3. Recommended Safe Integration Roadmap (If Approved Later)

### Track A — Clean Segmentation Track (Primary Immediate Use)
1. **Group-Aware Re-Splitting (`98` Base Captures):** Group all `440` triplets strictly by parsed base capture ID (`YYYY_MM_DD_pothole_NN`, `98` unique IDs) and create an uncontaminated grouped split (e.g., `68` train / `15` val / `15` test base IDs, excluding or flagging the `6` empty-mask augmented artifacts).
2. **YOLO Instance Segmentation Export:** Convert binary `{0, 255}` masks to normalized YOLO polygon labels via `cv2.findContours` into a separate standalone directory (e.g., `dataset_pothole440_seg_grouped/`).
3. **Model Experiment (Separate from Dashcam Production):** Optionally train/evaluate `YOLOv8n-seg` / `YOLOv8s-seg` on the grouped Pothole440 split purely as a **macro surface-delineation study** (crater shape & pixel area benchmark) without altering RoadVision's dashcam production weights.

### Track B — Masked Relative Depth Backbone Comparison (Secondary, Restricted Scope)
If comparing `MiDaS_small` vs. `Depth Anything V2` on Pothole440 prior to collecting field dashcam measurements:
1. **Scan-Validity Masking:** Detect and exclude the dominant constant padding region per depth map (plus morphological margin inside the Go!SCAN patch boundary) so step cliffs (`sample_06`, `sample_12`) never enter evaluation.
2. **Scale- & Shift-Invariant Metrics Only:** Evaluate strictly on the valid scan interior + grouped test captures using **Spearman rank correlation ($\rho$)**, **scale-and-shift aligned Pearson $r$** (accounting for inverse-depth sign inversion), and **affine-aligned MAE in native normalized units** — **never claiming cm/mm metric error**.

### Track C — Recommended Immediate Next Implementation Step
Before training any Pothole440 segmentation model or running MiDaS vs. Depth Anything V2 on masked interiors, the highest-value next step is either:
1. **Annotate the `40` held-out frames in `evaluation/ground_truth_labels/`** (or build a new untouched dashcam test set) so RoadVision's Phase 2 dashcam detectors have a certified untouched final benchmark, **OR**
2. **Run a masked relative-depth comparison script** (`MiDaS_small` vs. `Depth Anything V2` inside valid non-padded Go!SCAN regions on a base-ID-grouped Pothole440 subset) — keeping all results strictly labeled as **relative depth ordering in native units**.
