# Phase 5C Production Readiness Assessment & Final Recommendation

> [!WARNING]
> **Production Freeze Preserved:** No edits were made to [roadvision_engine.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/roadvision_engine.py), [server.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/server.py), dashboard UI, trained checkpoints, or calibration constants (`k_disp = 0.082`, `depth_exponent = 1.35`).

---

## 1. Readiness Decision

### Selected Recommendation: **Option B — Requires further runtime, validity or temporal robustness improvements** *(specifically validity-gating domain calibration & approach-distance normalization)*

We explicitly reject immediate promotion (**Option A**, even behind a production feature flag) and reject permanent shelving (**Option C**) based on direct Phase 5C evidence:

### Why NOT Option C (Purely Experimental)
- Algorithmic speed, engineering isolation, and lifecycle safety **already meet production engineering quality**:
  - `optimized_ransac` runs in `3.49 ms/region` mean across diverse video (`2.31 ms` median; standalone per-region CPU total `< 1.7 ms` on `ruralRoad`), keeping standalone end-to-end throughput at **`17.7–21.5 FPS`** on RTX 2050 (`~4–6 ms` above legacy).
  - Peak shadow memory is negligible (`3.4 MB` traced CPU overhead, `178.14 MB` peak GPU allocation).
  - **21/21 automated tests pass**, confirming deterministic `stratified_quadrant` sampling (`512` budget), crash-safe exception isolation (`NOT_EVALUATED`), staleness expiration (`EXPIRED -> None`), and clean track resets across `360p`/`720p` resolution changes.

### Why NOT Option A (Why Production Promotion Must Wait)
1. **Severe Urban 720p Rejection Rate (`7.6%` Valid on `rawSurvey_720p_160f`):**
   - While unseen rural video (`ruralRoad_202f`) yielded **`63.9%` valid (`441/690`) + `20.4%` stale (`141`)**, the `1280×720` urban asphalt sequence dropped to **`19/249` (`7.6%`) VALID** (`230 INVALID`), and `rangoonRoad_180f` dropped to **`47/214` (`22.0%`) VALID** (`110 EXPIRED`).
   - Across the entire 1,178-detection corpus, **`511` detections rejected on `unexpected_residual_polarity`** (`190` on `low_inlier_ratio`). Replacing legacy output directly today would silence alerts on nearly 92% of urban 720p detections until we distinguish true flat-mud puddle polarity reversals (as seen in [preview_rangoon_f60.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase5c_shadow_viz/preview_rangoon_f60.png)) from wide 720p box/background curvature causing `pos_strong_frac < 0.22` or negative MiDaS corners.
2. **Unresolved Vehicle Approach Drift (`ρ = 0.3441`):**
   - Relative P75 scores still climb strongly as vehicles approach (`Spearman ρ = 0.3441` vs box pixel area), because neither reference-MAD normalization nor frame-IQR normalization cancels perspective approach scaling. Thus any relative severity threshold (`MILD/MODERATE/SEVERE`) would remain camera-distance-dependent.
3. **Zero Verified Metric Ground Truth:**
   - Physical depth accuracy (`cm`, width, volume) remains unverified (`Pothole440` blocked). `BGR -> RGB` alignment increases valid acceptance (`40.82% -> 51.02%`) and conforms to ImageNet MiDaS weights, yet still outputs relative inverse depth only.

---

## 2. Recommended Next Engineering Step (Awaiting Explicit Approval)

Before requesting production integration approval:
1. **Audit & Calibrate `unexpected_residual_polarity` on `rawSurvey_720p` & `rangoonRoad`:** Test polygon-interior or inner-core trimmed cavity sampling (suppressing intact asphalt box corners) and soft quadratic road-crown tolerance so valid dry 720p potholes are preserved while true flat-water reversals stay rejected.
2. **Design Perspective-Normalized Relative Severity:** Investigate focal/box-`y2`-normalized relative indices (strictly labeled relative, not `cm`) so tracks remain stable under vehicle approach.
3. **Verify `BGR -> RGB` Fix Separately** once calibrated ground truth or polarity audit is complete.
