import cv2
import torch
import numpy as np

from roadvision_engine import RoadVisionPipeline, MetricDepthCalibrator

pipeline = RoadVisionPipeline(yolo_weights="runs/detect/pothole_combined_v2/weights/best.pt", device="cpu")
cap = cv2.VideoCapture("Watch Video_ Terrible potholes in Rangoon Road.mp4")

# Inspect frame 2, 4, 6, 8, 10
frames_to_check = [2, 6, 10, 20]
calibrator = MetricDepthCalibrator(frame_height=360, frame_width=640, camera_height_m=1.25, vfov_deg=60.0, horizon_ratio=0.40)

frame_idx = 0
while cap.isOpened() and frame_idx < 25:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx not in frames_to_check:
        continue

    H, W = frame.shape[:2]
    results = pipeline.yolo.predict(frame, conf=0.12, iou=0.45, imgsz=800, verbose=False)
    if len(results) == 0 or results[0].boxes is None:
        continue

    input_batch = pipeline.depth_transform(frame).to("cpu")
    with torch.no_grad():
        depth_pred = pipeline.midas(input_batch)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze()
    depth_map = depth_pred.numpy()

    boxes = results[0].boxes.xyxy.cpu().numpy()
    print(f"\n--- FRAME {frame_idx} ({len(boxes)} boxes) ---")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        bw = x2 - x1
        bh = y2 - y1

        # Current annular donut (margin 25 all around)
        ry1, ry2 = max(0, y1 - 25), min(H, y2 + 25)
        rx1, rx2 = max(0, x1 - 25), min(W, x2 + 25)
        crater_patch = depth_map[y1:y2, x1:x2]
        outer_region = depth_map[ry1:ry2, rx1:rx2]
        donut_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
        donut_mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
        surrounding_road = outer_region[donut_mask]

        old_road = np.median(surrounding_road)
        old_p15 = np.percentile(crater_patch, 15)
        old_delta = max(0.0, old_road - old_p15)

        # Lateral road reference: left rim and right rim at the SAME rows y1..y2
        margin_x = max(10, int(bw * 0.25))
        left_rim = depth_map[y1:y2, max(0, x1 - margin_x):x1]
        right_rim = depth_map[y1:y2, x2:min(W, x2 + margin_x)]

        # Top rim (road behind) and bottom rim (road in front)
        margin_y = max(10, int(bh * 0.25))
        top_rim = depth_map[max(0, y1 - margin_y):y1, x1:x2]
        bot_rim = depth_map[y2:min(H, y2 + margin_y), x1:x2]

        print(f"Box {i} [{x1},{y1},{x2},{y2}] (w={bw}, h={bh}):")
        print(f"  Top rim (far):     median={np.median(top_rim):.1f}" if top_rim.size > 0 else "  Top rim: None")
        print(f"  Bottom rim (near): median={np.median(bot_rim):.1f}" if bot_rim.size > 0 else "  Bottom rim: None")
        print(f"  Left rim:          median={np.median(left_rim):.1f}" if left_rim.size > 0 else "  Left rim: None")
        print(f"  Right rim:         median={np.median(right_rim):.1f}" if right_rim.size > 0 else "  Right rim: None")
        print(f"  Crater patch:      min={crater_patch.min():.1f}, P15={old_p15:.1f}, median={np.median(crater_patch):.1f}, max={crater_patch.max():.1f}")
        print(f"  Old donut median:  {old_road:.1f} -> OLD DELTA: {old_delta:.1f}")

        # Row-by-row baseline comparison:
        # At each row y, road baseline = mean(left_rim[row], right_rim[row])
        # cavity depth at row y = road baseline(row) - crater[row]
        row_deltas = []
        for r_idx in range(y2 - y1):
            cur_y = y1 + r_idx
            l_val = depth_map[cur_y, max(0, x1 - margin_x):x1]
            r_val = depth_map[cur_y, x2:min(W, x2 + margin_x)]
            rim_vals = []
            if l_val.size > 0: rim_vals.extend(l_val.flatten())
            if r_val.size > 0: rim_vals.extend(r_val.flatten())
            if len(rim_vals) > 0:
                row_road = np.median(rim_vals)
                row_crater = depth_map[cur_y, x1:x2]
                # Pothole cavity floor is deeper (farther -> lower disparity)
                row_floor = np.percentile(row_crater, 25)
                row_deltas.append(row_road - row_floor)

        if row_deltas:
            row_delta_med = np.median(row_deltas)
            row_delta_p75 = np.percentile(row_deltas, 75)
            row_delta_max = np.max(row_deltas)
            print(f"  ROW-ALIGNED LATERAL DELTA: median={row_delta_med:.2f}, P75={row_delta_p75:.2f}, max={row_delta_max:.2f}")

cap.release()
