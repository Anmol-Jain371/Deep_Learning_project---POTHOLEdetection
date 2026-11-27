# Phase 13 — Negative Controls & Failure Analysis

**Run ID:** `phase13_full_20261127`  
**Registry Artifact:** `runs/evaluation/phase13/phase13_full_20261127/failure_cases.csv`

---

| Experiment / Case ID | Source / Fixture | Outcome | Category | Actionable Status |
|---|---|---|---|---|
| `NC_blank_video` | `synthetic_blank_8f.mp4` | 0 candidate tracks, 0 false depth failures | `negative_control_passed` | Verified clean zero-detection path |
| `NC_corrupted_gps` | `lat=999.0_bad.csv` | Rejected cleanly with `ValueError` | `negative_control_passed` | Input sanitization enforced |
| `NG_raised_and_low_support` | Synthetic raised hump & neighbor-occluded cases (`geometry_ablation.py`) | `100%` rejected (`10/10` invalid synthetic cases) by B/C/D vs `0%` (`0/10`) by legacy annular `A` | `negative_control_passed` | Proves necessity of RANSAC/support gates |
| `FC_p9_low_conf_high_quality` | `0f286074/snap_t15_g0_f000048` | `detection_quality=0.4133` + missing temporal scored `0.8516 (HIGH_QUALITY)` | `estimator_limitation` | Preserved in deployed v1; documented in `E_conservative_gated` |
| `FC_p10_legacy_empty_ref_mask` | `42` pre-fix geometry JSONs | `reference_mask_grid` had `0` True cells due to `valid_ring` key mismatch | `artifact_key_mismatch_fixed` | Fixed in `surface.py`; new snapshots record `>0` cells |
| `FC_independent_detection_eval` | `evaluation/ground_truth_labels/*.txt` | `40` files present, all `0` bytes | `blocked_missing_ground_truth` | Annotate held-out images externally to unblock |
| `FC_depth_ground_truth` | `depth_ground_truth_manifest.csv` (`P01–P15`) | Provenance-incomplete / un-reproducible via live pipeline | `blocked_missing_ground_truth` | Uncalibrated relative scores only in V2 |
| `FC_geospatial_multitrip` | Synthetic GPS only | `0` genuine multi-trip matched pairs on disk | `blocked_missing_ground_truth` | Field multi-trip validation blocked |
