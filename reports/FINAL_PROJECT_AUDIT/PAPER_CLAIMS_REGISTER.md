# Scientific Claims Register (Categories A–F) — Phase 14 Manuscript Gate

---

## Category A — Supported and Ready to Report
1. **Architecture & Uncertainty-Preserving Data Contract:** RoadVision AI 2.0 separates relative inverse-depth estimation from unvalidated physical centimetre calibration (`RoadVisionV2Pipeline`), explicitly typing unavailable/invalid observations as `None` (`null`), labeling cached reads `CACHED_VALID` with bounded age expiration (`EXPIRED`), and executing single-GPU queued jobs cleanly on RTX 2050 (`4 GB` VRAM).
2. **Pothole440 Grouped Leakage Remediation:** Grouping Pothole440 by capture `base_id` (`68` train / `15` val / `15` test groups, excluding `52` test augmentations) eliminated cross-split augmentation leakage (`0` group overlap verified).
3. **Genuine CUDA Inference & Memory Footprint:** Under RTX 2050 `4 GB` VRAM constraints, sequential benchmarking measured YOLOv8n (`3.14M` params, `12.383 ms` predict mean, `68.49 MB` peak alloc) vs YOLOv8s (`11.14M` params, `25.710 ms`, `200.77 MB`), and MiDaS Small (`256`, `16.45 ms`) vs Depth Anything V2 Small (`266/336/392/518`, `25.80–147.01 ms`, `243–410 MB` peak alloc).

## Category B — Supported Only as Synthetic Validation
1. **Adaptive Vectorized MSAC/RANSAC vs OLS Reference Plane Estimation:** On 45 analytic synthetic road planes (flat, sloped, structured noise, 10–40% ring outliers, raised protrusions, low support, boundary truncation), OLS and RANSAC achieve `100%` (`45/45`) gate concordance vs legacy scalar annular `77.78%` (`35/45`), with MSAC/RANSAC achieving lower RMSE under outliers (`0.000131` vs OLS `0.001065`). *(Note: fix double-execution timing bug in `geometry_ablation.py` before citing `4.514 ms` speedup.)*
2. **Validity-Aware Temporal Filtering & Step Tracking:** On a synthetic step signal (`0.08 -> 0.22`) with injected invalid/expired gaps, `rolling_median_validity_ema` lowers step-to-step variance (`0.00574` vs raw `0.03500`) and preserves `None` across expired gaps (`2/2`) rather than hiding invalid intervals as numeric `0.0`.
3. **Relative 2.5D Snapshot Resampling Fidelity:** Downsampling synthetic signed residual grids (`max_grid_size=48`) preserves missing `NaN` holes (`9` null cells preserved), cross-section gaps, signed polarity (`ref - depth`), and non-zero `reference_mask_grid` (`1,141` ref cells, `304` cavity cells) with `0.000396` mean resampling difference and `0` extra depth inference calls.
4. **Geospatial Synchronization & Bounded-Diameter Clustering Mechanics:** Exact Haversine (`9.969 m` test pair), timestamp deduplication, gap rejection (`>2.0 s`), and bounded-diameter complete-linkage clustering (`POTENTIAL_MATCH`, `UNMATCHED`) verified deterministically on synthetic GPS fixtures.

## Category C — Supported Only as Exploratory Evidence
1. **YOLOv8n vs YOLOv8s Accuracy on `dataset_unified`:** Exploratory comparison on `dataset_unified/test` (`153` imgs: `mAP50=0.3643` vs `0.3581` at `conf=0.25`; `0.4865` at default `conf~0.001`) and `external_rangoon` (`35` imgs: `0.1560` vs `0.1728`), bounded by unresolved cross-split perceptual similarity (`328` pairs `dHash<=6`, `15` `<=2`) and missing source trip IDs.
2. **Pothole440 Grouped Test Segmentation (`N=15` Closeups):** YOLOv8n-seg reached `0.5977` mean IoU / `0.6763` mean Dice on `15` unaugmented closeups (`dataset_pothole440_seg_grouped/test`), evaluated as an auxiliary dashcam-domain transfer caveat (not deployed in V2 runtime).
3. **DAv2 vs MiDaS Fresh Valid Coverage (`N=42` Boxes, 17 Correlated Frames):** `DAv2 336 FP16` improved relative gate validity over `MiDaS 256` (`59.52%` vs `28.57%`) on sampled road clips while correlating strongly with DAv2 `518` (`Spearman = 0.9992`, internal relative consistency only).
4. **Budget-Aware Scheduling Trade-off on Frozen Clip (`150` Frames):** Reduced depth model executions from `150` to `80` (`46.67%` reduction) with `36.21%` fresh + `35.13%` cached = `71.34%` effective measurement availability (`18.63` simulated FPS) on a single 96%-active rural clip.
5. **Phase 9 Conservative vs Deployed Quality Categorization:** Conservative category gating lowers `HIGH_QUALITY` classifications (`105` vs `145` on the `obs[:300]` subset of job `0f286074`) when detector confidence is low or temporal history is absent.

## Category D — System Functionality Claims
1. End-to-end async Flask REST API (`/api/v2/*`), SQLite WAL persistence, atomic job queue & cooperative cancellation, single-GPU worker lock with V1, H.264 browser transcoding fallback handling, and React dashboard pages (verified cleanly via tests + API + Vite proxy; interactive Chrome walkthrough pending).
2. Live GPU smoke execution of `LIVE` (depth off & MiDaS on), `BALANCED` (DAv2 336 + E_budget_aware + temporal cache), and `SURVEY` (DAv2 518) with strict `100%` summary-to-frame reconciliation and clean negative/edge rejection (`400/404/409/415`).

## Category E — Claims Requiring Further Evidence (Currently BLOCKED)
1. **Independent Pothole Detection Generalization on Untouched Field Footage** *(requires annotating the `40` `0`-byte files in `evaluation/ground_truth_labels/` or a new multi-corridor dataset)*.
2. **Real-World Multi-Trip GPS Cross-Trip Pothole Deduplication Accuracy** *(requires real multi-pass synchronized dashcam GPS trips + fixing `is_synthetic` DB persistence)*.
3. **Civil-Engineering Alignment of Maintenance Review Priority** *(requires engineer-assigned PCI / priority labels)*.
4. **Live Interactive Chrome Console-Error-Free Sign-Off** *(requires running Chrome DevTools walkthrough & capturing screenshots for `15_Application_Screenshots/`)*.

## Category F — Claims That MUST NOT Appear in the Paper
1. **DO NOT CLAIM** `84.7% mAP50`, `92.4% Precision`, or V1 `YOLOv8s` 5.96 MB benchmark table numbers (`server.py`, `BenchmarkComparison.jsx`).
2. **DO NOT CLAIM** calibrated physical depth in **centimetres/millimetres**, physical pothole width/length, **litres of volume**, **kg of asphalt**, or **`0.56 cm MAE / 100% severity concordance`** (`depth_ground_truth_manifest.csv` uncertified).
3. **DO NOT CLAIM** zero dataset leakage across `dataset_unified` without noting the `328` `dHash<=6` cross-split pairs and missing trip IDs.
4. **DO NOT CLAIM** DAv2 `518` agreement (`Spearman 0.9992`) represents physical ground-truth depth accuracy.
5. **DO NOT CLAIM** `depth_call_reduction` (`46.67%`) directly equals end-to-end runtime speedup or generalized FPS across unseen videos.
6. **DO NOT CLAIM** heuristic quality indices (`0–1`) are calibrated probabilities or statistical confidence intervals.
7. **DO NOT CLAIM** ByteTrack session track counts (`unique_candidate_track_count`) or geospatial clusters equal verified unique physical potholes, or that vehicle GPS equals surveyed pothole coordinates.
8. **DO NOT CLAIM** maintenance `PRIORITY_REVIEW` represents structural severity or Pavement Condition Index (PCI).
