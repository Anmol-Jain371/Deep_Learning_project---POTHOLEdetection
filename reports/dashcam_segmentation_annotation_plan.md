# Phase 4A.4 — Dashcam Segmentation Annotation Plan (Development & Untouched Final Set)

**Prepared Development Dataset:** [dataset_dashcam_seg_dev/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_dashcam_seg_dev)  
**Generator Script:** [prepare_dashcam_annotation_set.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/prepare_dashcam_annotation_set.py)  
**Manifest Files:** [annotation_manifest.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_dashcam_seg_dev/annotation_manifest.csv), [annotation_manifest.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_dashcam_seg_dev/annotation_manifest.json)  
**Status:** **UNVERIFIED SUGGESTIONS ONLY (`PENDING_MANUAL_POLYGON_MASK`)** — Not Ground Truth

---

## 1. Development Dataset Inventory (`40` Session-Grouped Frames)

Frames are organized in separate session subdirectories (`frames/<session_id>/`) so future train/val splits can enforce **zero cross-split session overlap**:

| Session ID | Frames | Source Footage | Prior Model Exposure | Detector Proposals | Unverified ROI Mask Suggestions | BBox Fallbacks |
|---|---:|---|---|---:|---:|---:|
| `rural_road_unseen` | `12` | `uploads/upload_1794847200_ruralRoad_potHoles.mp4` (`640×360`) | **Unseen by Detector & Seg** | `44` | `1` | `43` |
| `rangoon_road_unseen_temporal` | `14` | `Watch Video_ Terrible potholes in Rangoon Road.mp4` (`640×360`) | **Unseen temporal frames** (single video) | `18` | `2` | `16` |
| `raw_survey_video` | `8` | `roadvision-dashboard/public/videos/raw_survey.mp4` (`1280×720`) | **Unseen raw frames** (`0` train MD5 matches) | `16` | `8` | `8` |
| `phone_val_session` | `6` | `dataset_phone_potholes/images/val` (`1280×720`) | **Detector unified pool provenance**; Seg unseen | `27` | `4` | `23` |
| **Total (`DEV_ANNOTATION_ONLY`)** | **`40`** | **4 Capture Sources** | — | **`105`** | **`15`** | **`90`** |

Each frame has paired metadata under:
- `frames/<session_id>/<sample_id>.png` — clean original RGB image
- `unverified_proposals/<session_id>/<sample_id>_proposals.json` — detector boxes + optional simplified contour polygons explicitly labeled `"UNVERIFIED_MODEL_SUGGESTION_DO_NOT_USE_AS_GT"`
- `unverified_mask_suggestions/<session_id>/<sample_id>_suggested_mask.png` — binary visualization of accepted ROI suggestions only (`0` elsewhere)

---

## 2. Manual Verification & Annotation Guidelines

Before any frame in `dataset_dashcam_seg_dev` can be used for fine-tuning or validation:
1. **Do Not Trust Auto-Suggestions:** Human annotators must independently verify *every* pothole across the entire drivable surface — not just inside the `105` detector boxes (recall from Phase 4A.4 that `YOLOv8n` missed `dt_05` / `dev_rangoon_f0020` cavities and emitted an imprecise shoulder box on `dev_rural_f0075`).
2. **True Cavity Boundary Definition:** Trace the depressed/eroded pothole rim in perspective projection; exclude intact surrounding pavement, traffic vehicles, curbs, grass shoulders, and specular glare shadows outside the depression.
3. **Water-Filled & Shadowed Regions:** Where standing water fills a clearly bounded depression, trace the recessed edge rim rather than only partial bright reflection islands.
4. **Update Verification Status:** Flip `mask_verification_status` to `VERIFIED_HUMAN_POLYGON` and `verified_ground_truth` to `true` only after two-pass visual inspection.

---

## 3. Recommended Separate Untouched Final Dashcam Test Set

Because all `40` frames in `dataset_dashcam_seg_dev` come from videos already inspected during Phase 4A.3/4A.4 development (and `phone_val_session` overlaps `dataset_unified` provenance), **`dataset_dashcam_seg_dev` must strictly remain a development/adaptation dataset**.

For unbiased final certification of any future domain-adapted dashcam segmentation model, we recommend constructing a **separate untouched final dashcam segmentation test set** (`dataset_dashcam_seg_final_test`) meeting these criteria:
- **Source Isolation:** Acquire `3–5` **brand-new** forward-facing dashcam / smartphone road survey videos recorded on completely separate road corridors, dates, and camera mounts (zero shared videos with `Rangoon Road`, `ruralRoad_potHoles`, `raw_survey`, `dataset_phone_potholes`, or `RDD2022`).
- **Target Sample Size:** `60–100` manually verified polygon-segmented frames (`>= 12–20` non-consecutive frames per video separated by `>= 2 seconds` to limit temporal correlation), stratified across dry daylight, shadowed, wet/water-filled, and multi-pothole urban/rural scenes, plus `~15–20` clean-road negative frames (shadows, manhole covers, tar-crack patches) to measure dashcam false-positive mask leakage.
- **Freeze & Lock:** Hash and freeze `dataset_dashcam_seg_final_test` before domain-adaptation fine-tuning begins, and evaluate on it **exactly once** under a pre-registered protocol.
