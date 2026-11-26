# Phase 4A.5 — Box-Prompted Segmentation Feasibility Report (MobileSAM)

**Status:** Complete (Feasibility Study — Zero Training, Production Files Untouched)  
**Adapter Module:** [experiments/prompt_guided_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/prompt_guided_segmentation.py)  
**Benchmark Script:** [experiments/benchmark_prompt_guided_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_prompt_guided_segmentation.py)  
**Unit Tests:** [tests/test_prompt_guided_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/tests/test_prompt_guided_segmentation.py) (`4/4` passed)  
**Benchmark Results:** [prompt_guided_segmentation_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_segmentation_results.json), [prompt_guided_segmentation_per_frame.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_segmentation_per_frame.csv)

---

## 1. Model Feasibility Verification (Task 1)

MobileSAM was evaluated as the primary lightweight box-prompted candidate. Because MobileSAM integrated cleanly with our existing Windows environment without dependency changes, secondary candidate SAM 2 Tiny was not downloaded.

| Criterion | Verified Finding (MobileSAM) |
|---|---|
| **Authoritative Source** | `ChaoningZhang/MobileSAM` (TinyViT image encoder distilled from ViT-H SAM) + Ultralytics official release asset (`https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt`) |
| **License & Usage** | **Apache License 2.0** (permissive for academic research and integration; Ultralytics wrapper subject to Ultralytics AGPL-3.0 / enterprise licensing if invoked via `ultralytics`) |
| **Checkpoint Integrity** | [mobile_sam.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/mobile_sam.pt): `40,728,226` bytes (`38.84 MiB`), `SHA256: 6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` |
| **Exact Parameter Count** | `10,130,092` total (`6,065,532` TinyViT image encoder, `6,220` prompt encoder, `4,058,340` mask decoder) |
| **Python / PyTorch / OS** | Verified natively on Windows with Python `3.12.10`, PyTorch `2.6.0+cu124`, CUDA `True` (`NVIDIA GeForce RTX 2050`, `4096 MiB`), `Ultralytics 8.4.211`, and already-installed `timm 1.0.31` |
| **Dependencies Modified** | **None** (`0` package upgrades or extra installs; `ultralytics.models.sam.Predictor` supports MobileSAM directly) |
| **Box-Prompt & Multi-Mask Support** | Native `[N, 4]` `xyxy` box prompt encoding at `1024×1024` top-left LetterBox (`center=False`), `multimask_output=True` returning `3` scale hypotheses per box alongside predicted mask IoU scores |
| **Startup Load Latency** | `379.26 ms` (measured separately from steady-state inference) |

---

## 2. Adapter Design & Embedding Reuse Architecture (Task 2)

Implemented in [MobileSAMBoxPromptAdapter](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/prompt_guided_segmentation.py):
1. **Detector Proposal Front-End:** Runs `Phase 2 YOLOv8n` (`imgsz=640`, `conf=0.25`, `iou=0.70`) once per frame to generate `N` bounding boxes.
2. **Full-Frame Embedding Reuse (`mode="full_frame_embed"`):**
   - Preprocesses frame via `LetterBox(1024, center=False)` and ImageNet normalization.
   - Runs TinyViT `image_encoder` **once per active frame**, caching the `(1, 256, 64, 64)` embedding.
   - Prompts `prompt_encoder` + `mask_decoder` across all `N` boxes simultaneously (`multimask_output=True`, `3N` masks at `256×256` logits).
3. **Cropped-ROI Prompting (`mode="cropped_roi_prompt"`):** Expands each box (`crop_pad_ratio=0.40`, `min_pad_px=24`) and runs MobileSAM encoder + decoder per crop for controlled latency/quality comparison.
4. **Coordinate Mapping & Multi-Mask Selection:** Inverts top-left LetterBox padding on `(3, 256, 256)` logits back to exact `(H, W)` resolution (`unpad_and_resize_sam_logits`), applies deterministic geometric consistency rules (`assess_sam_candidates_for_box`: SAM score `>= 0.25`, area `>= 36 px`, area ratio `0.12–1.35`, containment inside box `>= 0.55`, box coverage `>= 0.15`, border contact `<= 0.30`, components `<= 4`), and selects the valid candidate with highest SAM score.
5. **Fallback:** Rejected/empty candidates fall back to the filled detector box (`100%` detection preservation). **Disclaimer:** SAM predicted scores reflect internal mask confidence, *not* calibrated pothole probability.

---

## 3. RTX 2050 Steady-State Performance Benchmark (Task 5)

Measured after CUDA warmup with `torch.cuda.synchronize()` over `108` steady-state passes (`36` frames × `3` loops for `C1`) and `36` passes (`C2` & `B`), strictly **excluding** the `379.26 ms` startup load time:

### 3.1 Stage-by-Stage Breakdown (`C1: MobileSAM Full-Frame Embed`)

| Stage | Mean (ms) | Median (ms) | P95 (ms) | Notes |
|---|---:|---:|---:|---|
| **YOLOv8n Detector (`640`)** | `10.79` | `9.95` | `15.42` | Proposal front-end |
| **SAM Preprocessing (`1024` LetterBox)** | `5.54` | `6.07` | `7.04` | Only when `N > 0` |
| **TinyViT Image Encoder (All / Active)** | `78.66` / `88.50` | `88.46` / `88.49` | `88.76` / `88.76` | Fixed `~88.5 ms` cost per active frame |
| **SAM Prompt Decoder (Active / Per-Box)** | `19.14` / `6.74` | `18.70` / `6.51` | `39.40` / `8.81` | Amortized across `N` boxes |
| **Mask Unpadding & Reconstruction** | `2.42` | `1.46` | `7.21` | `256×256` logits → `(H, W)` |
| **Geometric Quality Assessment** | `14.07` | `8.70` | `45.51` | Evaluates `3N` candidate masks |
| **Combined Cascade (`C1`)** | **`128.50`** | **`133.48`** | **`191.48`** | **`7.78 FPS`** (excl. ByteTrack & MiDaS) |
| **Combined Cropped-ROI (`C2`)** | **`365.36`** | **`264.91`** | **`855.00`** | **`2.74 FPS`** (`247.73 ms` mean encoder) |
| **Phase 4A.4 ROI `YOLOv8n-seg` (`B`)** | **`55.37`** | **`48.13`** | **`112.34`** | **`18.06 FPS`** (`batch_roi=True`) |

- **GPU Memory Footprint (RTX 2050 `4 GB`):**
  - `C1` Full-Frame MobileSAM: `360.42 MB` peak allocated / `2,736.0 MB` peak reserved (PyTorch allocator caching alongside all loaded models).
  - `C2` Cropped-ROI MobileSAM: `368.88 MB` peak allocated / `2,736.0 MB` reserved.
  - `B` ROI `YOLOv8n-seg` (batched): `389.18 MB` peak allocated / `2,470.0 MB` reserved.

### 3.2 Scaling vs. Number of Proposals (Value of Embedding Reuse)

| Proposals / Frame | `C1` Encoder (ms) | `C1` Combined (ms / FPS) | `C2` Cropped Encoder (ms) | `C2` Combined (ms / FPS) |
|---|---:|---:|---:|---:|
| `0` (`n=12`) | `0.00` | `10.06` (`99.37 FPS`) | `0.00` | `17.41` (`57.45 FPS`) |
| `1` (`n=24`) | `88.45` | `116.56` (`8.58 FPS`) | `88.43` | `125.43` (`7.97 FPS`) |
| `2` (`n=21`) | `88.54` | `133.67` (`7.48 FPS`) | `177.19` | `243.77` (`4.10 FPS`) |
| `3` (`n=21`) | `88.49` | `144.63` (`6.91 FPS`) | `268.72` | `392.44` (`2.55 FPS`) |
| `4+` (`mean=5.5`, `n=30`) | `88.50` | `170.51` (`5.86 FPS`) | `508.96` | `762.64` (`1.31 FPS`) |

**Key Hardware Finding:** Image-embedding reuse dramatically flattens per-pothole inference cost (`6.74 ms/box` prompt decoder + constant `88.50 ms` encoder vs `~88.5 ms/crop` linear blowup in `C2`, achieving `4.47×` speedup at `4+` proposals). Nevertheless, **`7.78 FPS` (`128.50 ms`)** is **too slow for real-time on-device RoadVision video inference** once tracking and MiDaS depth are added.

---

## 4. Annotation-Assisted Workflow (`dataset_dashcam_seg_dev/`, Task 6)

Generated optional **unverified** MobileSAM polygon and mask suggestions across all `40` session-grouped frames in [dataset_dashcam_seg_dev/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_dashcam_seg_dev), stored separately under `unverified_mobilesam_suggestions/` and indexed in [mobilesam_suggestions_manifest.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/dataset_dashcam_seg_dev/mobilesam_suggestions_manifest.csv) (zero existing files overwritten):
- **Raw Yield:** `105` detector proposals → **`102` MobileSAM polygon suggestions (`97.14%`)**, `3` fallbacks (versus only `15/105` / `14.29%` ROI `YOLOv8n-seg` suggestions).
- **Required Human Review Protocol:** Every JSON/CSV entry is labeled `UNVERIFIED_MODEL_SUGGESTION` (`verified_ground_truth=False`). Human annotators must:
  1. Audit **every suggested polygon** for water-only trimming (expand masks to include outer eroded cavity rims) and false-positive prompts (reject concrete patches/shadows segmented by SAM).
  2. Manually annotate **missed detector potholes** across the full frame (MobileSAM cannot invent boxes for undetected cavities).
  3. Curate and inspect **genuine clean-road negative frames** to quantify false-positive segmentation leakage.
  4. Reserve a **completely separate, untouched final dashcam evaluation set** grouped by distinct road videos/capture sessions (zero overlap with `Rangoon Road`, `ruralRoad_potHoles`, `raw_survey`, or `dataset_phone_potholes`).

---

## 5. Final Recommendation

**Recommendation B — Prompt-guided segmentation is useful primarily as an annotation assistant.**

Zero-shot MobileSAM overcomes the `69.39%` empty-ROI failure rate of `YOLOv8n-seg` (`0%` empty predictions, `97.14%` polygon suggestion coverage on `dataset_dashcam_seg_dev/`) inside RTX 2050 VRAM limits (`360 MB` allocated), making it highly valuable for accelerating offline polygon dataset curation. However, its `~88.5 ms/frame` TinyViT encoding bottleneck (`7.78 FPS` steady-state) and class-agnostic semantic ambiguity (detailed in [segmentation_method_comparison.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/segmentation_method_comparison.md)) preclude real-time autonomous deployment. Production should remain detection-based until domain-adapted semantic dashcam segmentation is trained and evaluated on manually verified labels.
