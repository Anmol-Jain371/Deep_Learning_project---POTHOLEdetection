# Phase 13 — Scientific Limitations & Unresolved Field Validation Requirements

1. **No Untouched Independent Annotated Detection Test Set:**  
   `evaluation/ground_truth_labels/*.txt` contains `40` empty (`0`-byte) annotation files. `dataset_unified/val`, `test`, and `external_rangoon` have prior tuning/comparison exposure and cross-split perceptual near-duplicates; `dataset_pothole440_seg_grouped/test` (`15` originals) contains domain-shifted overhead close-ups. Independent generalization remains **`BLOCKED`**.
2. **Uncalibrated Physical Depth & Volume:**  
   No straightedge/LiDAR/stereo physical ground truth exists in the repository (legacy `depth_ground_truth_manifest.csv` `P01–P15` failed reproducibility checks). V2 emits uncalibrated relative inverse-depth residuals ONLY (`is_metric_3d = False`); raw residuals are model/resolution-dependent and non-comparable across MiDaS and DAv2 or across different input resolutions without compatible affine/provenance contracts.
3. **Approximate Camera-Location GPS Only:**  
   GPS coordinates synchronize against vehicle/dashcam trajectory timestamps, NOT surveyed pothole center geometry. Cross-trip candidate clustering has only been evaluated on synthetic GPS fixtures (`field_validation_status = BLOCKED`).
4. **Session ByteTrack IDs ≠ Verified Unique Physical Potholes:**  
   Track fragmentation, occlusion, or boundary entry/exit can yield multiple session tracks for one cavity or merge nearby defects.
5. **Heuristic Evidence Quality vs. Physical Severity:**  
   Phase 9 quality (`HIGH_QUALITY` possible with `detection_quality < 0.50` + absent temporal evidence under deployed v1 weighting) and Phase 12 review-priority categories reflect evidence triage only—NOT calibrated probability, Pavement Condition Index (PCI), or civil-engineering structural repair urgency.
6. **Historical Geometry Snapshots (`42` files):**  
   Snapshots written before the `valid_ring_mask` fix retain `0`-cell `reference_mask_grid` fields; newly generated snapshots populate `reference_mask_grid` accurately.
