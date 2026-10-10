import cv2
import torch
import numpy as np

from roadvision_engine import RoadVisionPipeline, MetricDepthCalibrator

pipeline = RoadVisionPipeline(yolo_weights="runs/detect/pothole_combined_v2/weights/best.pt", device="cpu")
cap = cv2.VideoCapture("Watch Video_ Terrible potholes in Rangoon Road.mp4")

calibrator = MetricDepthCalibrator(frame_height=360, frame_width=640, camera_height_m=1.25, vfov_deg=60.0, horizon_ratio=0.40)

def extract_planar_detrended_disp(depth_map, x1, y1, x2, y2, margin=20):
    H, W = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5:
        return 0.0

    # Expand margin
    mx = max(10, int(bw * 0.25))
    my = max(10, int(bh * 0.25))
    ry1, ry2 = max(0, y1 - my), min(H, y2 + my)
    rx1, rx2 = max(0, x1 - mx), min(W, x2 + mx)

    # Outer region and mask excluding inner crater
    outer_region = depth_map[ry1:ry2, rx1:rx2]
    h_out, w_out = outer_region.shape
    donut_mask = np.ones((h_out, w_out), dtype=bool)
    donut_mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False

    # Coordinates of road reference pixels
    ys, xs = np.where(donut_mask)
    if len(ys) < 20:
        return 0.0

    z_vals = outer_region[ys, xs]

    # Fit a 1D or 2D plane: z(y, x) = a*y + b*x + c
    # Note: local road disparity is predominantly a function of y (vertical slope)
    A = np.column_stack([ys, xs, np.ones_like(ys)])
    try:
        plane_coeffs, residuals, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
        a, b, c = plane_coeffs
    except Exception:
        return 0.0

    # For every pixel in the inner crater, compute (expected_road - actual_depth)
    inner_patch = depth_map[y1:y2, x1:x2]
    iy, ix = np.indices(inner_patch.shape)
    # Global coordinates in outer_region frame
    iy_outer = (y1 - ry1) + iy
    ix_outer = (x1 - rx1) + ix

    expected_road = a * iy_outer + b * ix_outer + c
    cavity_depth = expected_road - inner_patch

    # Positive cavity_depth means actual disparity is LOWER than road -> sunken hole
    # We take the 80th percentile of positive depression or max of median
    # To be robust to specular water or debris, take 85th percentile of cavity_depth
    dep_85 = np.percentile(cavity_depth, 85)
    return float(max(0.0, dep_85))


def extract_lateral_rim_disp(depth_map, x1, y1, x2, y2, margin=20):
    H, W = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5:
        return 0.0

    mx = max(10, int(bw * 0.30))
    # Left and right rims at the SAME rows y1..y2
    left_rim = depth_map[y1:y2, max(0, x1 - mx):x1]
    right_rim = depth_map[y1:y2, x2:min(W, x2 + mx)]

    # At each row y, road baseline = median of available lateral rim pixels
    row_depressions = []
    for r in range(bh):
        row_y = y1 + r
        rim_pixels = []
        if left_rim.size > 0:
            rim_pixels.extend(left_rim[r, :])
        if right_rim.size > 0:
            rim_pixels.extend(right_rim[r, :])
        if len(rim_pixels) == 0:
            continue
        row_road = np.median(rim_pixels)
        row_crater = depth_map[row_y, x1:x2]
        # Pothole cavity floor is deeper (lower disparity)
        # Take the 20th percentile of crater disparity at this row
        row_floor = np.percentile(row_crater, 20)
        row_depressions.append(max(0.0, row_road - row_floor))

    if not row_depressions:
        return 0.0
    # Take 80th percentile of row depressions
    return float(np.percentile(row_depressions, 80))

print("Testing new extraction methods on Rangoon Road video...")
all_old = []
all_planar = []
all_lateral = []

frame_idx = 0
while cap.isOpened() and frame_idx < 100:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    if frame_idx % 2 != 0: continue

    H, W = frame.shape[:2]
    results = pipeline.yolo.predict(frame, conf=0.12, iou=0.45, imgsz=800, verbose=False)
    if len(results) == 0 or results[0].boxes is None: continue

    input_batch = pipeline.depth_transform(frame).to("cpu")
    with torch.no_grad():
        depth_pred = pipeline.midas(input_batch)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze()
    depth_map = depth_pred.numpy()

    boxes = results[0].boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        dy = max(15.0, float(y2 - calibrator.horizon_y))
        Z_road = (calibrator.H_cam * calibrator.fy) / dy
        z_scale = (Z_road ** 1.35) * calibrator.k_disp

        old_d = pipeline.depth_extractor.extract_depth_disparity(depth_map, x1, y1, x2, y2)
        planar_d = extract_planar_detrended_disp(depth_map, x1, y1, x2, y2)
        lateral_d = extract_lateral_rim_disp(depth_map, x1, y1, x2, y2)

        old_cm = np.clip(old_d * z_scale, 0, 35)
        planar_cm = np.clip(planar_d * z_scale, 0, 35)
        lateral_cm = np.clip(lateral_d * z_scale, 0, 35)

        all_old.append(old_cm)
        all_planar.append(planar_cm)
        all_lateral.append(lateral_cm)

cap.release()

all_old = np.array(all_old)
all_planar = np.array(all_planar)
all_lateral = np.array(all_lateral)

print(f"\nEvaluated {len(all_old)} detections across 100 frames:")
print("\n1. OLD METHOD (Annular Donut median vs P15):")
print(f"   Median depth: {np.median(all_old):.1f} cm | Mean: {np.mean(all_old):.1f} cm | Max: {np.max(all_old):.1f} cm")
print(f"   Mild (<4cm): {np.sum(all_old < 4)} ({np.sum(all_old < 4)/len(all_old)*100:.1f}%)")
print(f"   Moderate (4-7.9cm): {np.sum((all_old>=4)&(all_old<8))} ({np.sum((all_old>=4)&(all_old<8))/len(all_old)*100:.1f}%)")
print(f"   Severe (>=8cm): {np.sum(all_old >= 8)} ({np.sum(all_old >= 8)/len(all_old)*100:.1f}%)")
print(f"   Saturated at 35cm: {np.sum(all_old >= 34.9)} ({np.sum(all_old >= 34.9)/len(all_old)*100:.1f}%)")

print("\n2. PLANAR DETRENDED METHOD (Plane fitting z = a*y + b*x + c):")
print(f"   Median depth: {np.median(all_planar):.1f} cm | Mean: {np.mean(all_planar):.1f} cm | Max: {np.max(all_planar):.1f} cm")
print(f"   Mild (<4cm): {np.sum(all_planar < 4)} ({np.sum(all_planar < 4)/len(all_planar)*100:.1f}%)")
print(f"   Moderate (4-7.9cm): {np.sum((all_planar>=4)&(all_planar<8))} ({np.sum((all_planar>=4)&(all_planar<8))/len(all_planar)*100:.1f}%)")
print(f"   Severe (>=8cm): {np.sum(all_planar >= 8)} ({np.sum(all_planar >= 8)/len(all_planar)*100:.1f}%)")
print(f"   Saturated at 35cm: {np.sum(all_planar >= 34.9)} ({np.sum(all_planar >= 34.9)/len(all_planar)*100:.1f}%)")

print("\n3. ROW-ALIGNED LATERAL RIM METHOD (Left/Right rim at same y):")
print(f"   Median depth: {np.median(all_lateral):.1f} cm | Mean: {np.mean(all_lateral):.1f} cm | Max: {np.max(all_lateral):.1f} cm")
print(f"   Mild (<4cm): {np.sum(all_lateral < 4)} ({np.sum(all_lateral < 4)/len(all_lateral)*100:.1f}%)")
print(f"   Moderate (4-7.9cm): {np.sum((all_lateral>=4)&(all_lateral<8))} ({np.sum((all_lateral>=4)&(all_lateral<8))/len(all_lateral)*100:.1f}%)")
print(f"   Severe (>=8cm): {np.sum(all_lateral >= 8)} ({np.sum(all_lateral >= 8)/len(all_lateral)*100:.1f}%)")
print(f"   Saturated at 35cm: {np.sum(all_lateral >= 34.9)} ({np.sum(all_lateral >= 34.9)/len(all_lateral)*100:.1f}%)")
