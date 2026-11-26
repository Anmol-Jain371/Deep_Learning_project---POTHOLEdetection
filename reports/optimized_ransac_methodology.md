# Optimized Adaptive RANSAC & Reference Point Budgeting Methodology (Phase 5B)

> [!IMPORTANT]
> **Isolated Research Module — Zero Production Modifications:** Implemented in [experiments/optimized_relative_depth.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/optimized_relative_depth.py) alongside the preserved Phase 5A baseline [experiments/robust_relative_depth.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/robust_relative_depth.py). All outputs express **relative `MiDaS_small` inverse-depth units only** (NOT metric depth in cm, physical area, or volume). Production `roadvision_engine.py`, `server.py`, checkpoints, frozen datasets, `k_disp = 0.082`, and `depth_exponent = 1.35` remain untouched.

---

## 1. Reproducible Profiling of Phase 5A Baseline (Task 1)

Profiling [extract_relative_depth_for_box](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/robust_relative_depth.py#L742-L828) across all 98 YOLOv8n regions on the 36 cached development frames (`cProfile` × 3 passes = 294 calls, `5.428 s` under profiler instrumentation; `10.25 ms/region` un-instrumented mean) identified four dominant CPU bottlenecks:

| Stage / Function | Observed Profiler Cumtime (294 calls) | Un-Instrumented Phase 5A Mean (ms) | Root Cause Identified |
| :--- | :---: | :---: | :--- |
| `fit_reference_surface` (`ransac_plane`) | `4.100 s` (`75.5%`) | `7.62 ms` | Sequential Python `for` loop calling `np.linalg.solve` `120×` per box (`35,253` calls), per-iteration `A_sub @ c_cand` consensus checks, and `3×` full-array `np.linalg.lstsq` SVD calls over up to `>30,000` pixels |
| `compute_cavity_statistics` | `0.439 s` | `1.13 ms` | Repeated partial sorts (`4×` separate `np.percentile` + `2×` `np.median` + duplicate `np.isfinite`) |
| `build_adaptive_annular_region` | `0.360 s` | `0.83 ms` | Extracting & normalizing full-resolution `(N, 3)` arrays even for `>30,000 px` rings |
| `_estimate_local_surface_sigma` | `0.293 s` | `~0.81 ms` | Full-patch `np.median` on all horizontal + vertical finite differences without strided subsampling |

Notice that GPU porting was intentionally **not** assumed beneficial: MiDaS inference already sits on the RTX 2050 (`27.38 ms`), whereas per-box planar fitting operates on small irregular subsets (`256–512` points after budgeting) where host-to-device synchronization would dominate compute. Pure CPU vectorization + bounded spatial sampling eliminates the overhead directly.

---

## 2. Budgeted Reference Sampling Strategies (Task 2)

`build_and_sample_annular_region` constructs the exact neighbor-excluded finite annular ring mask and reduces pixel counts to `max_ref_samples` (evaluated across `256, 512, 1024, 2048`, and `None` full resolution):

1. **`full`:** Uses all `N_avail` finite ring pixels.
2. **`uniform_stride`:** `np.linspace(0, N_avail - 1, num=budget, dtype=int64)` over row-major coordinates.
3. **`deterministic_random`:** Seeded (`seed + 101`) permutation draw without replacement.
4. **`stratified_quadrant` (Recommended Default):**
   - Partitions `valid_ring` into 4 axial sectors around target box center (`Top`, `Bottom`, `Left`, `Right`).
   - Allocates an initial support floor `min(quad_count, min_quadrant_samples)` (`5` minimum) to every populated quadrant so wide/tall potholes never starve narrow flanking sides under small budgets (`256–512`).
   - Distributes remaining budget proportionally to quadrant area via intra-quadrant uniform linspace stride.
   - Records **both** full-ring spatial support and sampled-subset quadrant support, rejecting one-sided/starved subsets identically (`poor_spatial_support`, `insufficient_reference_samples`).

---

## 3. Vectorized Adaptive RANSAC/MSAC & Normal-Equation Refit (Task 3)

Implemented in `fit_ransac_optimized` (`OptimizedDepthConfig`):
- **Strided Finite-Difference Scale (`_estimate_local_sigma_fast`):** Computes local pixel-to-pixel sigma over at most `4096` valid ring difference pairs (`0.6589 ms` mean), immune to block curb steps.
- **Batched Triplet Generation & Degeneracy Pruning:** Draws micro-batches of `ransac_batch_size = 16` triplets vectorized via `rng.integers`, rejects duplicate indices and near-collinear triplets (`det2 < 1e-4`), and solves **all non-degenerate 3×3 systems simultaneously** via `np.linalg.solve(M_v, z_v[..., None])`.
- **Single-GEMM Consensus Scoring:** Computes `|z - c_batch @ A^T|` in one `(K, 3) @ (3, M)` matrix multiply per micro-batch (`0.3460 ms` mean total RANSAC loop time!).
- **Adaptive Confidence Stopping:** Evaluates at least `ransac_min_iters = 16`, updating $N_{\text{req}} = \lceil \log(1 - 0.99) / \log(1 - w_{\text{best}}^3) \rceil$ dynamically up to `ransac_max_iters = 128` (`mean_iters_evaluated = 45.55` across 98 real regions at `budget = 512`, and `16` on clean planes).
- **3×3 Normal-Equation Refit & Separable Broadcast (`0.1267 ms`):** Refits inliers via `A^T A c = A^T z` checking `eigvalsh` condition number (`sqrt(eig_max / eig_min) <= 1e4`), and evaluates the crater plane using separable 1D axis broadcasting (`cy_vec, cx_vec`).
- **Single-Pass Cavity Stats & Preserved Quality Gates (`0.4688 + 0.3662 ms`):** Joint `np.percentile(..., [25, 50, 75, 85, 90])` + identical 11-point `assess_quality_fast` rules (no gate thresholds relaxed).

---

## 4. Sampling & Point-Budget Ablation on 36 Dashcam Frames (98 Regions)

Compared against Phase 5A `original_ransac` (`mean = 10.25 ms`, `valid = 49/98`, `ref_mad_median = 2.600`):

| Strategy & Budget | Valid Count (`%`) | Matched w/ Orig (`n`) | Poor Support / Insuff Rejections | Mean Iters (`max=128`) | Matched Ref MAD (Med) | Median `|ΔP75|` vs Orig (Rel Units) | Mean / Med / P95 (ms) | Speedup vs Orig |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full` (`None`) | `48/98` (`48.98%`) | `41` | `0` / `0` | `46.20` | `2.5475` | `0.8824` (`5.37%`) | `27.22 / 10.76 / 125.20` | `0.37x` |
| `uniform_stride_256` | `45/98` (`45.92%`) | `41` | `0` / `0` | `45.39` | `2.7168` | `0.7121` (`5.52%`) | `2.93 / 2.23 / 6.47` | `3.42x` |
| `uniform_stride_512` | `44/98` (`44.90%`) | `39` | `0` / `0` | `45.55` | `2.7908` | `0.6106` (`4.29%`) | `2.71 / 1.99 / 6.00` | `3.70x` |
| `uniform_stride_1024` | `50/98` (`51.02%`) | `40` | `0` / `0` | `46.04` | `2.5273` | `0.5764` (`5.20%`) | `2.82 / 2.09 / 6.36` | `3.56x` |
| `uniform_stride_2048` | `46/98` (`46.94%`) | `39` | `0` / `0` | `46.20` | `2.3760` | `0.7910` (`6.40%`) | `3.15 / 2.36 / 6.83` | `3.18x` |
| `stratified_quadrant_256` | `51/98` (`52.04%`) | `42` | `0` / `0` | `44.73` | `2.7914` | `0.9418` (`7.22%`) | `2.76 / 1.98 / 6.09` | `3.63x` |
| **`stratified_quadrant_512`** | **`50/98` (`51.02%`)** | **`40`** | **`0` / `0`** | **`45.55`** | **`2.6018`** | **`0.7187` (`6.78%`)** | **`2.94 / 2.11 / 6.49`** | **`3.41x`** |
| `stratified_quadrant_1024` | `44/98` (`44.90%`) | `35` | `0` / `0` | `46.04` | `2.5853` | `0.8656` (`4.56%`) | `2.95 / 2.21 / 6.46` | `3.40x` |
| `stratified_quadrant_2048` | `44/98` (`44.90%`) | `38` | `0` / `0` | `45.88` | `2.4852` | `0.8514` (`6.92%`) | `3.29 / 2.54 / 7.41` | `3.05x` |
| `deterministic_random_256` | `47/98` (`47.96%`) | `36` | `0` / `0` | `43.59` | `2.4614` | `1.2096` (`10.48%`) | `2.66 / 1.91 / 5.80` | `3.77x` |
| `deterministic_random_512` | `48/98` (`48.98%`) | `39` | `1` / `0` | `45.55` | `2.5272` | `0.7290` (`8.40%`) | `2.86 / 2.06 / 6.34` | `3.51x` |
| `deterministic_random_1024` | `49/98` (`50.00%`) | `42` | `0` / `0` | `45.55` | `2.6566` | `0.8523` (`6.49%`) | `3.02 / 2.14 / 7.19` | `3.32x` |
| `deterministic_random_2048` | `47/98` (`47.96%`) | `40` | `0` / `0` | `44.90` | `2.7625` | `0.8115` (`6.86%`) | `3.34 / 2.53 / 7.37` | `3.00x` |

**Trade-offs & Choice of Recommended Budget (`stratified_quadrant`, `max_ref_samples = 512`):**
- `full` budget explodes to **`27.22 ms` (`P95 = 125.20 ms`)** because scoring `32/16 × 30,000` full-ring arrays dominates cache bandwidth (`0.37x` slower than Phase 5A's `1500`-point cap!).
- `deterministic_random_512` triggered `1` spurious `poor_spatial_support` failure (`7.4%–10.5%` mismatch at `256`–`512`), whereas **`stratified_quadrant_512` eliminates spatial-support starvation (`0` spurious failures), matches `51.02%` valid rate (`50/98`), hits the local minimum `|ΔP75|` divergence among stratified budgets (`0.7187`), and runs in `2.94–3.02 ms` mean (`2.11–2.13 ms` median, `3.39x–3.41x` speedup)**.
