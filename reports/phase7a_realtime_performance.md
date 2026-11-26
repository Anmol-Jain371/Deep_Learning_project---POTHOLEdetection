# Phase 7A — Real-Time Deadline & Asynchronous Feasibility Report (Tasks 10, 11 & 12)

Hardware: **NVIDIA GeForce RTX 2050 (`4.0 GB` VRAM)**, Windows 11, Python `3.12.10`. Visuals: [phase7a_latency_and_invocation_rate.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_viz/phase7a_latency_and_invocation_rate.png), [phase7a_demo_preview_rural_f0045.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_viz/phase7a_demo_preview_rural_f0045.png), [phase7a_demo_preview_survey_f0021.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/phase7a_viz/phase7a_demo_preview_survey_f0021.png).

---

## 1. Full Pipeline End-to-End Timing & Proposal Scaling (Cached Always-On Grounding, `482f` Mixed `720p+360p` / `300f` `360p`)

Per-stage `DAv2 336 (fp16_weights)` on `482f` (`33%` `720p`): preproc `30.57 ms`, GPU infer `39.58 ms`, resize `2.18 ms`, RANSAC extract `8.67 ms` (`2.73 ms` on `360p` Rangoon). On `482f`:

| Policy | Mean / Med (ms) | P95 / P99 (ms) | Keyframe / Non-KF Med (ms) | Amortized FPS | `0 -> 1 -> 2 -> 3 -> 4+` Mean (ms) | Miss `>40ms` (`25 FPS`) | Miss `>33.3ms` (`30 FPS`) | `300f` `360p` Mean / Miss25 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MiDaS_AlwaysOn` | `107.7 / 69.1` | `184.3 / 240.4` | `99.9 / —` | `9.29` | `83 -> 92 -> 93 -> 78 -> 57` | `100.0%` | `100.0%` | `72.9 ms` / `100.0%` |
| `MiDaS_CandGated` | `89.4 / 59.9` | `174.1 / 230.1` | `88.0 / 27.9` | `11.19` | `21 -> 89 -> 90 -> 75 -> 56` | `86.7%` | `89.0%` | `67.7 ms` / `94.3%` |
| `DAv2_336_AlwaysOn` | `119.5 / 94.5` | `180.8 / 271.4` | `107.2 / —` | `8.37` | `101 -> 110 -> 117 -> 109 -> 88` | `100.0%` | `100.0%` | `90.7 ms` / `100.0%` |
| `DAv2_336_Fixed_k3` | `60.0 / 29.7` | `150.6 / 224.7` | `106.4 / 29.7` | `16.66` | `—` | `45.4%` | `53.1%` | `40.4 ms` / `32.7%` |
| **`DAv2_336_TrackAware`** | `77.9 / 84.5` | `166.7 / 267.2` | `105.4 / 28.9` | `12.84` | `—` | `62.2%` | `67.2%` | `36.4 ms` (`27.5 FPS`) / `27.0%` |
| **`DAv2_336_BudgetAware`** | `62.2 / 31.8` | **`143.1`** / `202.9` | `100.8 / 29.7` | `16.07` | `—` | `51.0%` | `57.7%` | **`35.8 ms` (`28.0 FPS`) / `26.0%`** |
| `DAv2_266_TrackAware` | `78.8 / —` | `198.8 / 313.2` | `95.2 / 29.4` | `12.69` | `—` | `64.1%` | `69.3%` | `33.1 ms` (`30.2 FPS`) / `31.0%` |
| `DAv2_518_AlwaysOn` | `229.1 / 220.7` | `280.7 / 368.0` | `220.7 / —` | `4.37` | `—` | `100.0%` | `100.0%` | `173.3 ms` / `100.0%` |

*(Why `4+` can average lower than `1`: `ruralRoad` `640×360` carries dense `4+` potholes with lower CPU decode/preprocess time than single-box `1280×720` `rawSurvey` frames.)*

---

## 2. Live End-to-End Synchronous Verification (`362` Live Frames: `ruralRoad_202f` + `rawSurvey_160f`)

Measured end-to-end with live video decode + YOLOv8n + ByteTrack + scheduler + GPU inference + RANSAC + `TrackDepthCache` + OpenCV HUD rendering (`0.49–0.66 ms`) + CSV logging (`0.01–0.02 ms`):

| Live Policy (`362` Frames) | Live Depth Calls | Wall Sec | Sustained Wall FPS | Mean / Med / P95 / P99 (ms) | KF / Non-KF Mean (ms) | Deadline Miss `25 / 30 FPS` | Sim. `25 FPS` Backlog | Peak GPU VRAM Alloc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B_midas_256_cand_gated` | `311` | `20.72 s` | `17.47 FPS` | `56.75 / 50.39 / 93.27 / 102.2` | `63.04 / 18.40` | `79.3% / 85.9%` | `362/362` | `346.9 MB` |
| `A_dav2_336_always_on` | `362` | `33.39 s` | `10.84 FPS` | `91.93 / 83.30 / 132.2 / 168.5` | `91.93 / —` | `100% / 100%` | `362/362` | `393.8 MB` |
| **`D_dav2_336_track_aware`** | `188` | `24.55 s` | `14.75 FPS` | `67.47 / 74.55 / 144.6 / 195.8` | `111.2 / 20.22` | `52.5% / 53.3%` | `361/362` | `418.7 MB` |
| **`E_dav2_336_budget_40ms`** | **`132`** | **`17.53 s`** | **`20.65 FPS`** | `48.06 / 24.57 / 120.1 / 155.0` | `97.36 / 19.76` | **`36.5% / 37.0%`** | `326/362` | `442.0 MB` |

*(Process CPU peak RSS helper returned `0.0 MB` due to Windows API query permissions and is omitted rather than estimated.)*

---

## 3. Bounded Asynchronous Depth Worker Feasibility (Task 11, `ruralRoad_202f`)

Testing `BoundedAsyncDepthWorker(DAv2_336 fp16_weights, maxsize=1, max_result_age_sec=0.35)` with zero artificial sleeps:
- **Submitted / Completed:** `95` submitted (`4` dropped queue-full), `95` completed, **`2` rejected as stale/out-of-order**, `93` consumed validly (`consumed_result_age_ms`: `40.0 ms` med, `80.0 ms` P95, `83.2 ms` P99).
- **Critical GPU Contention Finding:** Even without blocking Python on `depth.predict()`, running `DAv2 336` concurrently on the same RTX 2050 CUDA device contended with main-thread YOLOv8n (`640`), pushing **main-thread blocking latency to `54.52 ms` mean / `109.91 ms` P95 (`59.41%` > `40 ms`)**!

---

## 4. Deadline Conclusion

Scheduling boosts **sustained live throughput `+90.5%` (`10.84 -> 20.65 FPS`)** and slashes non-keyframe latency to **`~19.8–20.2 ms`**, yet **both synchronous keyframes (`97–111 ms`) and async single-GPU contention (`109.9 ms` P95) still violate strict real-time `25/30 FPS` per-frame deadlines (`36.5%–59.4%` deadline misses)**. Amortized throughput alone cannot establish hard real-time compliance.
