# Phase 9 — Uncertainty-Aware Hazard Assessment

## 1. Implementation Summary

Implemented an explainable, non-probabilistic measurement-quality assessment module in [roadvision_v2/quality/assessment.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_v2/quality/assessment.py) and integrated it into [roadvision_v2/schemas.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_v2/schemas.py), [roadvision_v2/pipeline.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_v2/pipeline.py), [roadvision_api_v2/routes.py](file:///d:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_api_v2/routes.py), and the RoadVision 2.0 React dashboard (`Workspace`, `Hazard Explorer`, `Track Analysis`, `Reports`).

### Key Design Guarantees
- **Six separated quality components** (`[0, 1]` or `null` when unavailable): `detection_quality`, `geometric_reference_quality` (combines correlated RANSAC inlier ratio, sample count, quadrant coverage, MAD residual, and annular retention in one group), `cavity_extraction_quality`, `temporal_consistency`, `measurement_freshness`, and `environmental_risk`.
- **Heuristic composite quality index (`[0, 1]` or `null`)** normalized strictly over available evidence weight, explicitly marked `is_experimental_heuristic=true` and `is_calibrated_probability=false`.
- **Quality categories**: `HIGH_QUALITY`, `MODERATE_QUALITY`, `LOW_QUALITY`, `INSUFFICIENT_EVIDENCE`, `INVALID`.
- **Strict gating**: `FRESH_INVALID` and `EXPIRED` measurements always map to `INVALID` with `quality_index=null` regardless of high detector confidence. `CACHED_VALID` inherits source fresh geometric/cavity/environmental evidence, applies an age-based freshness penalty (`cached_freshness_penalty_max=0.35` over `0.45s`), and is never counted toward fresh temporal stability.
- **Temporal stability**: Computes `median_relative_score`, `mad_relative_score`, `iqr_relative_score`, `robust_temporal_variability`, and `ref_fit_mad_variability` strictly from compatible `(model_identity, depth_input_resolution, inference_precision)` `FRESH_VALID` observations (`>= 3` samples required; returns `null` otherwise).
- **Explainable rejection diagnostics**: Preserves raw pipeline codes (`insufficient_reference_samples`, `poor_spatial_support`, `low_inlier_ratio`, `unexpected_residual_polarity`, `suspected_water_reflection`, `EXPIRED`, etc.) alongside human-readable explanations.

## 2. Automated Verification

Executed `python -m unittest tests/test_phase9_quality_assessment.py -v` (`7/7` tests passed) covering high-quality vs noisy reference planes, insufficient samples, missing diagnostics, invalid/expired enforcement, cached aging, repeated cache non-inflation, temporal outliers, incompatible config isolation, track aggregation, strict JSON `null` serialization, and backward compatibility.

## 3. Exploratory Evaluation (Real Video Clips, RTX 2050 CUDA, `balanced` mode)

| Clip | Frames | Obs | Tracks | Quality Categories | Median Fresh QI | Sufficient Tracks | Top Rejection Reasons |
|------|--------|-----|--------|-------------------|-----------------|-------------------|-----------------------|
| `upload_1794546659_test_clip.mp4` | 25 | 25 | 1 | `HIGH_QUALITY`: 11, `MODERATE_QUALITY`: 8, `INVALID`: 6 | `0.8877` | `0 / 1` (only 2 `FRESH_VALID` vs 17 `CACHED_VALID`) | `unexpected_residual_polarity` (6) |
| `upload_1794546827_ruralRoad_potHoles.mp4` | 60 | 118 | 17 | `HIGH_QUALITY`: 58, `MODERATE_QUALITY`: 18, `INSUFFICIENT_EVIDENCE`: 18, `INVALID`: 24 | `0.8730` | `2 / 17` | `unexpected_residual_polarity` (27), `poor_spatial_support` (1) |

> **Scientific Limitation:** Heuristic quality indices and empirical MAD/IQR dispersions reflect internal model/RANSAC consistency only; they are **not** calibrated probabilities of correctness or physical cm uncertainty bounds, and thresholds were not tuned on an independent ground-truth validation benchmark.
