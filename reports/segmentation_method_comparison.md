# Phase 4A.5 — Dashcam Segmentation Method Comparison & Semantic Error Analysis

**Benchmark Data:** [prompt_guided_segmentation_results.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_segmentation_results.json), [prompt_guided_segmentation_per_frame.csv](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_segmentation_per_frame.csv)  
**Side-by-Side Visualizations (`36` frames):** [runs/benchmark/prompt_guided_seg_viz/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz)  
**Benchmark Script:** [experiments/benchmark_prompt_guided_segmentation.py](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/experiments/benchmark_prompt_guided_segmentation.py)

> [!IMPORTANT]
> The `36` development frames were previously examined during Phase 4A.4 and lack pixel segmentation ground truth. Metrics below measure **mask availability, heuristic bounding-box consistency, and visual plausibility** — **NOT** verified pixel IoU or Dice accuracy. Zero tuning was performed on the completed Pothole440 grouped test split.

---

## 1. Controlled 3-Way Comparison Across `36` Development Frames (`98` Identical Proposals)

| Metric | A: Detector Only (`Phase 2 YOLOv8n`) | B: ROI-Guided `YOLOv8n-seg` (`batched`) | C1: Box-Prompted `MobileSAM` (Full Embed) | C2: Box-Prompted `MobileSAM` (Cropped ROI) |
|---|---:|---:|---:|---:|
| **Detection Preservation Rate** | `100%` (`98/98`) | `100%` (`98/98`) | `100%` (`98/98`) | `100%` (`98/98`) |
| **Accepted Mask Rate** | — (boxes only) | `11.22%` (`11/98`) | **`98.98%` (`97/98`)** | `95.92%` (`94/98`) |
| **Rejected Mask Rate** | — | `88.78%` (`87/98`) | `1.02%` (`1/98`) | `4.08%` (`4/98`) |
| **Empty Mask Frequency** | — | **`69.39%` (`68/98`)** | **`0.00%` (`0/98`)** | **`0.00%` (`0/98`)** |
| **BBox Fallback Frequency** | `100%` (`98/98`) | `88.78%` (`87/98`) | `1.02%` (`1/98`) | `4.08%` (`4/98`) |
| **Accepted Mean Area / Det Box** | `1.000` | `0.6191` | `0.3858` | `0.4191` |
| **Accepted Mean Box Containment** | `1.000` | `0.9065` | `0.9924` | `0.9903` |
| **Accepted Mean Border Contact** | — | `0.0213` | `0.0000` | `0.0023` |
| **Excessive Coverage (`> 35%` frame)** | `0 / 36` | `0 / 36` (post-QA) | `0 / 36` | `0 / 36` |
| **Mean Active Frame Mask Coverage** | — | `5.756%` | `4.745%` | `4.764%` |
| **Rejection Breakdown** | — | `empty: 68`, `spill: 17`, `border: 9`, `area: 9`, `ambig: 2`, `cov: 1`, `frag: 1` | `excessive_fragmentation: 1` | `excessive_fragmentation: 4` |

### Why Does `MobileSAM` Achieve `98.98%` Heuristic Acceptance vs `11.22%` for `YOLOv8n-seg`?
1. **Zero Empty Failures:** Unlike `YOLOv8n-seg` (which must classify pixels as `pothole` vs background and emits empty masks on `69.39%` of oblique dashcam crops), MobileSAM treats the box prompt as a **class-agnostic spatial prior** ("segment the prominent object inside this rectangle") and almost always returns a bounded shape (`0%` empty).
2. **Box-Prompt Containment Bias:** MobileSAM masks average `99.24%` containment inside the prompt box (`0.3858×` box area), meaning geometric consistency checks rarely reject them — **even when the segmented shape is NOT a pothole**.
3. **Full-Frame Embed vs Cropped-ROI Prompt (`C1` vs `C2`):** Re-encoding crops individually (`C2`) yields visually similar masks at `4.764%` vs `4.745%` active coverage (`0.4191` vs `0.3858` area ratio) while triggering slightly more fragmentation rejections (`4` vs `1`) and costing `2.84×` higher latency (`365.36 ms` vs `128.50 ms`).

---

## 2. Qualitative Semantic Failure Analysis (Task 4)

Visual inspection of comparative panels in [prompt_guided_seg_viz/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz) reveals critical semantic differences that heuristic geometry checks **cannot** detect:

1. **Class-Agnostic Segmentation of False-Positive Concrete Patches ([dev_rangoon_f0020.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz/dev_rangoon_f0020.png)):**
   - The `Phase 2 YOLOv8n` detector emitted a false-positive box (`0.37`) over an intact pale concrete road patch between muddy puddles. `ROI YOLOv8n-seg` correctly returned empty (`FB:empty`), whereas **MobileSAM (`s:0.97`) cleanly segmented the intact concrete patch inside the box**. Because that patch sits completely inside the prompt box, geometric QA accepted it (`acc=2/2`).
2. **Water-Reflection Lock vs. True Crater Rim ([dev_phone_frame_0005.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz/dev_phone_frame_0005.png), [dev_rangoon_f0440.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz/dev_rangoon_f0440.png)):**
   - On the large foreground crater in `frame_0005.jpg`, `ROI YOLOv8n-seg` captured both the water puddle and adjacent dark rocky cavity (`green mask`), whereas **MobileSAM latched solely onto the high-contrast reflective water pool**, under-segmenting the eroded left/bottom gravel rim of the actual pothole. Conversely, on distant bright water puddles (`dev_rural_f0015.png`, `dev_rural_f0060.png`, `dev_rural_f0135.png`), MobileSAM traced remarkably clean puddle shapes where `ROI YOLOv8n-seg` failed completely (`0/7` accepted on `dev_rural_f0135`).
3. **Grass, Roadside Shoulder & Spurious Box Failures ([dev_rural_f0075.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz/dev_rural_f0075.png)):**
   - Where `YOLOv8n` emitted a spurious right-edge box overlapping roadside grass, `ROI YOLOv8n-seg` painted a large grass region (`acc=1/3`). MobileSAM isolated only a tiny surface blob inside the spurious box (less visual grass sprawl, yet still geometrically "accepted" as `acc=3/3` inside a non-pothole box!).
4. **Shadowed Cavities & Cracks ([dev_rangoon_f0695.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz/dev_rangoon_f0695.png), [dev_phone_frame_0035.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz/dev_phone_frame_0035.png)):**
   - On bottom-edge shadowed cracks (`dev_rangoon_f0695`) and shadowed foreground craters (`dev_phone_frame_0035`), MobileSAM produced cohesive masks where `ROI YOLOv8n-seg` emitted empty or spilling outputs; on a shadow-only false-positive box in `frame_0035`, MobileSAM shattered into multiple components and was rightly caught by `excessive_fragmentation` (`FB:excessive-frag`).
5. **Multiple Adjacent & Overlapping Potholes ([dev_survey_f0020.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz/dev_survey_f0020.png), [dev_rangoon_f0270.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/prompt_guided_seg_viz/dev_rangoon_f0270.png)):**
   - When `YOLOv8n` emits two overlapping boxes on one crater (`dev_rangoon_f0270`), MobileSAM segments the full puddle+cracked rim (`acc=2/2`, `s:0.96`), recovering the upper left rocky shoulder clipped by `ROI YOLOv8n-seg` (`dev_rangoon_f0270`). However, overlapping detector boxes still cause duplicate masks over the same cavity unless box NMS is tightened or merged.

---

## 3. Summary Verdict Across Methods

- **`Phase 2 YOLOv8n` Detector Only (`A`):** Fastest (`~10.8 ms`), retains all detections, zero false semantic mask confidence, but provides coarse rectangular envelopes (`4.28%` pavement over-coverage in Phase 4A.3).
- **`ROI-Guided YOLOv8n-seg` (`B`):** Domain-specific semantic filter rejects false patches (`dev_rangoon_f0020`), runs at `18.06 FPS` (`55.37 ms`), and prevents whole-road collapse, **but suffers `69.39%` empty masks** on oblique/distant dashcam cavities.
- **`Box-Prompted MobileSAM` (`C1`):** Eliminates empty ROI failures (`0%` empty, `98.98%` visually coherent boundary extraction across water/cracks/distant puddles), **but is class-agnostic** (segments non-pothole concrete patches & water glares with `0.97` SAM score) and too slow (`7.78 FPS` / `128.50 ms`) for real-time inference. Its optimal role is **offline annotation assistance** to build manually verified dashcam segmentation training data.
