# Phase 4A.3 — Box-versus-Mask Representation Comparison Report

**Study Type:** Controlled Box-versus-Mask Representation Comparison (Matched `YOLOv8n-seg` Predictions)  
**Split:** Reconstructed Grouped Test Split (`dataset_pothole440_seg_grouped/test`, `15` unaugmented original `1200×800` images)  
**Checkpoint:** [best.pt](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/segment/pothole440_yolov8n_seg_v1/weights/best.pt) (`SHA256: 73f525322681468322dd129589279b016eeeab1197879a8876e3071e8e0d6d6d`)  
**Raw Data:** [pothole440_box_vs_mask.json](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_box_vs_mask.json)  
**Visualizations:** [runs/benchmark/pothole440_box_vs_mask_viz/](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_box_vs_mask_viz)

---

## 1. Experimental Design (Isolating Representation Value)

To isolate the geometric benefit of pixel-level segmentation masks over rectangular bounding boxes without confounding differences between separately trained detectors:
1. Both representations are derived from the **exact same forward pass and matched surviving predictions** of `YOLOv8n-seg` (`imgsz=640`, `iou_nms=0.70`, `retina_masks=True`) at primary `conf=0.50` (and logged at secondary `conf=0.25` for reference).
2. **Rectangular Box Mask:** Union of filled predicted axis-aligned `boxes.xyxy` rectangles clipped to `1200×800`.
3. **Segmentation Mask:** Union of bilinearly upscaled (`>= 0.50`) predicted instance segmentation masks.
4. Both masks are evaluated identically against the original `1200×800` Pothole440 binary PNG ground-truth masks. Notice that the `4` empty-prediction test images (`test_06`, `test_08`, `test_09`, `test_12`) yield `0` predicted pixels for both box and mask representations, leaving `11` active comparisons.

---

## 2. Aggregate Box-versus-Mask Results (`n=15` Test Images)

### 2.1 Primary Frozen Threshold (`conf=0.50`)

| Metric | Matched Box Mask | Matched Seg Mask | Absolute Delta (Seg − Box) | Relative Improvement / Change |
|---|---:|---:|---:|---:|
| **Macro Mean IoU** | `0.4650` | **`0.5996`** | `+0.1345` | **`+28.93%` relative IoU gain** |
| **Micro-Pooled IoU** | `0.4973` | **`0.6015`** | `+0.1042` | **`+20.95%` relative IoU gain** |
| **Macro Mean Dice** | `0.5648` | **`0.6564`** | `+0.0916` | **`+16.22%` relative Dice gain** |
| **Micro-Pooled Dice** | `0.6643` | **`0.7512`** | `+0.0869` | **`+13.08%` relative Dice gain** |
| **Total FP Pavement Area** | `549,107` px | **`68,896` px** | `-480,211` px | **`-87.45%` FP pavement reduction** |
| **FP Pavement (% of GT Background)** | `4.277%` | **`0.537%`** | `-3.740 pp` | **`8.0×` cleaner background exclusion** |
| **FP Pavement (% of Total Pixels)** | `3.813%` | **`0.478%`** | `-3.335 pp` | — |
| **Total FN Cavity Area** | **`512,265` px** | `581,133` px | `+68,868` px | `+13.44%` FN cavity increase |
| **FN Cavity (% of Total GT Cavity)** | **`32.791%`** | `37.200%` | `+4.409 pp` | Tight mask trims edge/pale islands |
| **Images Where Seg IoU > Box IoU** | — | — | **`11 / 15`** (`11 / 11` detected) | — |

### 2.2 Secondary Reference (`conf=0.25`)

At `conf=0.25` (`3` empty-prediction images, `12` detected), the identical structural pattern holds:
- **Macro IoU:** `0.4885` (Box) → `0.6343` (Seg) (`+0.1458`, **`+29.85%`** relative)
- **Micro IoU:** `0.5294` (Box) → `0.6534` (Seg) (`+0.1240`, **`+23.42%`** relative)
- **Macro Dice:** `0.5999` (Box) → `0.6984` (Seg) (`+0.0985`, **`+16.42%`** relative)
- **Micro Dice:** `0.6923` (Box) → `0.7904` (Seg) (`+0.0981`, **`+14.16%`** relative)
- **FP Pavement Area:** `615,966` px (`4.798%` bg) → `75,258` px (`0.586%` bg), reducing false-positive pavement contamination by **`540,708` pixels (`-87.78%` relative)** while increasing FN cavity area by `83,234` pixels (`+20.35%`).

---

## 3. Per-Image Box-vs-Mask Comparison (`conf=0.50`)

| # | Filename | Inst | GT % | Box IoU | Seg IoU | IoU Gain | Box Dice | Seg Dice | Box FP (px) | Seg FP (px) | Box FN (px) | Seg FN (px) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | `2022_05_22_pothole_16.png` | `2` | `17.184` | `0.7390` | `0.8683` | `+0.1293` | `0.8499` | `0.9295` | `56,304` | `8,244` | `1,452` | `14,564` |
| 02 | `2022_05_23_pothole_1.png` | `1` | `4.494` | `0.3527` | `0.6248` | `+0.2721` | `0.5215` | `0.7691` | `79,164` | `25,632` | `0` | `172` |
| 03 | `2022_05_25_pothole_6.png` | `2` | `10.937` | `0.5950` | `0.9308` | `+0.3358` | `0.7461` | `0.9642` | `68,800` | `5,428` | `1,587` | `2,214` |
| 04 | `2022_06_01_pothole_11.png` | `1` | `10.087` | `0.7273` | `0.8971` | `+0.1698` | `0.8421` | `0.9458` | `35,255` | `5,633` | `762` | `4,906` |
| 05 | `2022_06_01_pothole_19.png` | `1` | `3.416` | `0.6624` | `0.9190` | `+0.2566` | `0.7969` | `0.9578` | `15,744` | `1,139` | `641` | `1,608` |
| 06 | `2022_06_01_pothole_24.png` | `0` | `20.798` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0` | `0` | `199,664` | `199,664` |
| 07 | `2022_06_03_pothole_32.png` | `1` | `15.785` | `0.7723` | `0.8795` | `+0.1071` | `0.8715` | `0.9359` | `42,741` | `5,512` | `1,491` | `13,417` |
| 08 | `2022_06_03_pothole_34.png` | `0` | `6.224` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0` | `0` | `59,754` | `59,754` |
| 09 | `2022_06_03_pothole_6.png` | `0` | `7.261` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0` | `0` | `69,710` | `69,710` |
| 10 | `2022_06_03_pothole_9.png` | `1` | `23.404` | `0.6042` | `0.9456` | `+0.3414` | `0.7533` | `0.9721` | `145,722` | `9,228` | `892` | `3,491` |
| 11 | `2022_06_04_pothole_13.png` | `1` | `10.158` | `0.5611` | `0.6447` | `+0.0836` | `0.7188` | `0.7840` | `40,459` | `1,419` | `20,100` | `33,734` |
| 12 | `2022_05_22_pothole_15.png` | `0` | `14.096` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0` | `0` | `135,318` | `135,318` |
| 13 | `2022_05_22_pothole_24.png` | `1` | `3.236` | `0.6644` | `0.8631` | `+0.1988` | `0.7983` | `0.9265` | `15,238` | `3,867` | `304` | `915` |
| 14 | `2022_06_02_pothole_3.png` | `2` | `10.270` | `0.6486` | `0.7077` | `+0.0591` | `0.7869` | `0.8289` | `27,210` | `1,855` | `16,994` | `27,502` |
| 15 | `2022_06_03_pothole_33.png` | `1` | `5.380` | `0.6483` | `0.7128` | `+0.0645` | `0.7866` | `0.8323` | `22,470` | `939` | `3,596` | `14,164` |

---

## 4. Scientific Interpretation for RoadVision Geometry

1. **Massive Elimination of Intact-Pavement Contamination (`-87.45%` FP):** Because potholes are non-convex and frequently elongated or diagonal (see [bvm_10_2022_06_03_pothole_9.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_box_vs_mask_viz/bvm_10_2022_06_03_pothole_9.png) and [bvm_02_2022_05_23_pothole_1.png](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/runs/benchmark/pothole440_box_vs_mask_viz/bvm_02_2022_05_23_pothole_1.png)), filled bounding boxes sweep large areas of surrounding intact asphalt (`549,107` FP px vs only `981,070` TP px for seg). Replacing filled boxes with predicted masks improves IoU on **every single detected test pothole (`11/11`)**, by up to `+0.3414` IoU (`+56.5%` relative on `test_10`).
2. **Boundary Tightness vs Interior Recall Trade-Off:** Because boxes act as an outer envelope, box FN cavity area (`512,265` px) is lower than segmentation FN area (`581,133` px). On pale interior islands (`test_11`, `test_14`, `test_15`), segmentation masks occasionally under-segment lighter inner debris (`+68,868` extra FN px across all 15 frames), whereas rectangular boxes blindly include those pixels along with tens of thousands of surrounding intact pavement pixels.
3. **Implication for Annular Depth Sampling & Area Estimation:** Any future cavity area or depth extraction strictly benefits from segmentation masks **only when the detection itself does not miss or collapse** — motivating the domain-transfer check in [pothole440_dashcam_transfer_report.md](file:///D:/DeepLearning/Deep_Learning_project---POTHOLEdetection/reports/pothole440_dashcam_transfer_report.md).
