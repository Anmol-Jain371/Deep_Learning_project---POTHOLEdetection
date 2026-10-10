import cv2
import torch
import numpy as np
import os
import sys

from roadvision_engine import RoadVisionPipeline, MetricDepthCalibrator

def main():
    video_path = "Watch Video_ Terrible potholes in Rangoon Road.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"Inspecting {video_path}: {W}x{H}, {total_frames} frames")

    pipeline = RoadVisionPipeline(yolo_weights="runs/detect/pothole_combined_v2/weights/best.pt", device="cpu")
    calibrator = MetricDepthCalibrator(frame_height=H, frame_width=W, camera_height_m=1.25, vfov_deg=60.0, horizon_ratio=0.40)

    sampled_depths = []
    raw_deltas = []
    z_roads = []
    sample_records = []
    frame_idx = 0

    while cap.isOpened() and frame_idx < 150: # Sample first 150 frames
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 2 != 0:
            continue

        results = pipeline.yolo.predict(frame, conf=0.12, iou=0.45, imgsz=800, verbose=False)
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        # Get depth map
        input_batch = pipeline.depth_transform(frame).to("cpu")
        with torch.no_grad():
            depth_pred = pipeline.midas(input_batch)
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze()
        depth_map = depth_pred.numpy()

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            # Donut disparity extraction
            raw_disp_delta = pipeline.depth_extractor.extract_depth_disparity(depth_map, x1, y1, x2, y2, margin=25)
            
            # Pinhole geometry
            dy = max(15.0, float(y2 - calibrator.horizon_y))
            Z_road = (calibrator.H_cam * calibrator.fy) / dy
            
            # Unclipped and clipped cm
            unclipped_cm = raw_disp_delta * (Z_road ** 1.35) * calibrator.k_disp
            clipped_cm = float(np.clip(unclipped_cm, 0.0, 35.0))

            # Donut details
            ry1, ry2 = max(0, y1 - 25), min(H, y2 + 25)
            rx1, rx2 = max(0, x1 - 25), min(W, x2 + 25)
            crater_patch = depth_map[y1:y2, x1:x2]
            outer_region = depth_map[ry1:ry2, rx1:rx2]
            donut_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
            donut_mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
            surrounding_road = outer_region[donut_mask]

            p15_crater = np.percentile(crater_patch, 15) if crater_patch.size > 0 else 0
            med_crater = np.median(crater_patch) if crater_patch.size > 0 else 0
            p85_crater = np.percentile(crater_patch, 85) if crater_patch.size > 0 else 0
            med_donut = np.median(surrounding_road) if surrounding_road.size > 0 else 0

            sampled_depths.append(clipped_cm)
            raw_deltas.append(raw_disp_delta)
            z_roads.append(Z_road)

            sample_records.append({
                "frame": frame_idx,
                "box": (x1, y1, x2, y2),
                "y2": y2,
                "dy": dy,
                "Z_road": Z_road,
                "med_donut": med_donut,
                "p15_crater": p15_crater,
                "med_crater": med_crater,
                "p85_crater": p85_crater,
                "raw_disp_delta": raw_disp_delta,
                "unclipped_cm": unclipped_cm,
                "clipped_cm": clipped_cm
            })

    cap.release()

    sampled_depths = np.array(sampled_depths)
    raw_deltas = np.array(raw_deltas)
    z_roads = np.array(z_roads)

    print("\n" + "="*80)
    print("DETAILED DEPTH DISTRIBUTION AUDIT (First 150 Frames)")
    print("="*80)
    print(f"Total Detections Evaluated: {len(sampled_depths)}")
    print(f"Estimated Depth (clipped cm):")
    print(f"  Min:    {np.min(sampled_depths):.2f} cm")
    print(f"  25th%:  {np.percentile(sampled_depths, 25):.2f} cm")
    print(f"  Median: {np.median(sampled_depths):.2f} cm")
    print(f"  75th%:  {np.percentile(sampled_depths, 75):.2f} cm")
    print(f"  90th%:  {np.percentile(sampled_depths, 90):.2f} cm")
    print(f"  Max:    {np.max(sampled_depths):.2f} cm")
    print(f"  Mean:   {np.mean(sampled_depths):.2f} cm")

    mild_count = np.sum(sampled_depths < 4.0)
    mod_count = np.sum((sampled_depths >= 4.0) & (sampled_depths < 8.0))
    sev_count = np.sum(sampled_depths >= 8.0)
    saturated_35 = np.sum(sampled_depths >= 34.9)

    print(f"\nSeverity Categories:")
    print(f"  MILD (<4.0 cm):      {mild_count} ({mild_count/len(sampled_depths)*100:.1f}%)")
    print(f"  MODERATE (4.0-7.9cm): {mod_count} ({mod_count/len(sampled_depths)*100:.1f}%)")
    print(f"  SEVERE (>=8.0 cm):   {sev_count} ({sev_count/len(sampled_depths)*100:.1f}%)")
    print(f"  Saturated at 35.0cm: {saturated_35} ({saturated_35/len(sampled_depths)*100:.1f}%)")

    print(f"\nUnderlying Variables:")
    print(f"  Raw Disparity Delta: min={raw_deltas.min():.2f}, med={np.median(raw_deltas):.2f}, max={raw_deltas.max():.2f}")
    print(f"  Z_road (metres):     min={z_roads.min():.2f}m, med={np.median(z_roads):.2f}m, max={z_roads.max():.2f}m")

    print(f"\nFirst 12 Sample Detections:")
    print(f"{'Frm':<4} {'[x1,y1,x2,y2]':<20} {'y2':<4} {'dy':<5} {'Z_rd(m)':<8} {'Donut':<7} {'P15_c':<7} {'Med_c':<7} {'dDisp':<6} {'Unclip(cm)':<11} {'Clip(cm)'}")
    for r in sample_records[:12]:
        b_str = f"[{r['box'][0]},{r['box'][1]},{r['box'][2]},{r['box'][3]}]"
        print(f"{r['frame']:<4} {b_str:<20} {r['y2']:<4} {r['dy']:<5.1f} {r['Z_road']:<8.2f} {r['med_donut']:<7.1f} {r['p15_crater']:<7.1f} {r['med_crater']:<7.1f} {r['raw_disp_delta']:<6.1f} {r['unclipped_cm']:<11.1f} {r['clipped_cm']:.1f}")

if __name__ == "__main__":
    main()
