import cv2
import torch
import numpy as np
import os
import copy
from collections import defaultdict

from roadvision_engine import RoadVisionPipeline, MetricDepthCalibrator, TemporalEMASmoother

def extract_planar_detrended_disp(depth_map, x1, y1, x2, y2):
    H, W = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5:
        return 0.0

    mx = max(10, int(bw * 0.25))
    my = max(10, int(bh * 0.25))
    ry1, ry2 = max(0, y1 - my), min(H, y2 + my)
    rx1, rx2 = max(0, x1 - mx), min(W, x2 + mx)

    outer_region = depth_map[ry1:ry2, rx1:rx2]
    h_out, w_out = outer_region.shape
    donut_mask = np.ones((h_out, w_out), dtype=bool)
    donut_mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False

    ys, xs = np.where(donut_mask)
    if len(ys) < 20:
        return 0.0

    z_vals = outer_region[ys, xs]
    A = np.column_stack([ys, xs, np.ones_like(ys)])
    try:
        plane_coeffs, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
        a, b, c = plane_coeffs
    except Exception:
        return 0.0

    inner_patch = depth_map[y1:y2, x1:x2]
    iy, ix = np.indices(inner_patch.shape)
    iy_outer = (y1 - ry1) + iy
    ix_outer = (x1 - rx1) + ix

    expected_road = a * iy_outer + b * ix_outer + c
    cavity_depth = expected_road - inner_patch

    # 85th percentile of cavity depression isolates deepest crater floor
    dep_85 = np.percentile(cavity_depth, 85)
    return float(max(0.0, dep_85))


def run_comparison():
    video_path = "Watch Video_ Terrible potholes in Rangoon Road.mp4"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    pipeline = RoadVisionPipeline(yolo_weights="runs/detect/pothole_combined_v2/weights/best.pt", device="cpu")
    calibrator = MetricDepthCalibrator(frame_height=H, frame_width=W, camera_height_m=1.25, vfov_deg=60.0, horizon_ratio=0.40)

    smoother_old = TemporalEMASmoother()
    smoother_new = TemporalEMASmoother()

    HORIZON_Y = int(H * 0.15)
    frame_idx = 0

    per_frame_old_cm = []
    per_frame_new_cm = []

    print(f"Running side-by-side full video comparison on {total_frames} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        stride = 2
        if frame_idx % stride != 0:
            continue

        results = pipeline.yolo.track(
            frame, persist=True, tracker="road_tracker.yaml",
            conf=0.12, iou=0.45, imgsz=800, verbose=False
        )

        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        if getattr(results[0].boxes, 'id', None) is not None and results[0].boxes.id is not None:
            raw_track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            raw_track_ids = [i + 1 for i in range(len(boxes))]

        # Get depth map
        input_batch = pipeline.depth_transform(frame).to("cpu")
        with torch.no_grad():
            depth_pred = pipeline.midas(input_batch)
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze()
        depth_map = depth_pred.numpy()

        for box, tid, conf in zip(boxes, raw_track_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            if y2 <= HORIZON_Y:
                continue

            dy = max(15.0, float(y2 - calibrator.horizon_y))
            Z_road = (calibrator.H_cam * calibrator.fy) / dy
            z_scale = (Z_road ** 1.35) * calibrator.k_disp

            # 1. Old disparity delta
            old_delta = pipeline.depth_extractor.extract_depth_disparity(depth_map, x1, y1, x2, y2)
            old_cm = float(np.clip(old_delta * z_scale, 0.0, 35.0))
            per_frame_old_cm.append(old_cm)

            # 2. New planar detrended disparity delta
            new_delta = extract_planar_detrended_disp(depth_map, x1, y1, x2, y2)
            new_cm = float(np.clip(new_delta * z_scale, 0.0, 35.0))
            per_frame_new_cm.append(new_cm)

            # Specular water
            patch = frame[y1:y2, x1:x2]
            is_water = pipeline.water_detector.inspect_patch(patch)

            smoother_old.update(tid, old_cm, y2, is_water, float(conf))
            smoother_new.update(tid, new_cm, y2, is_water, float(conf))

        if frame_idx % 100 == 0:
            print(f"  Processed frame {frame_idx}/{total_frames}...")

    cap.release()

    # Analyze unique tracked potholes (>= 2 observations)
    old_verified = {tid: d for tid, d in smoother_old.records.items() if d.get("observations", 1) >= 2}
    new_verified = {tid: d for tid, d in smoother_new.records.items() if d.get("observations", 1) >= 2}

    print("\n" + "="*80)
    print("FULL VIDEO BENCHMARK COMPARISON (1,005 Frames)")
    print("="*80)

    old_depths = [d["smooth_cm"] for d in old_verified.values()]
    new_depths = [d["smooth_cm"] for d in new_verified.values()]

    print(f"\n1. PER-FRAME ESTIMATED DEPTH DISTRIBUTION ({len(per_frame_old_cm)} detections):")
    print(f"{'Metric':<25} {'OLD (Annular Donut)':<22} {'NEW (Planar Detrended)'}")
    print("-" * 75)
    print(f"{'Minimum':<25} {np.min(per_frame_old_cm):<22.1f} {np.min(per_frame_new_cm):.1f} cm")
    print(f"{'25th Percentile':<25} {np.percentile(per_frame_old_cm, 25):<22.1f} {np.percentile(per_frame_new_cm, 25):.1f} cm")
    print(f"{'Median':<25} {np.median(per_frame_old_cm):<22.1f} {np.median(per_frame_new_cm):.1f} cm")
    print(f"{'75th Percentile':<25} {np.percentile(per_frame_old_cm, 75):<22.1f} {np.percentile(per_frame_new_cm, 75):.1f} cm")
    print(f"{'90th Percentile':<25} {np.percentile(per_frame_old_cm, 90):<22.1f} {np.percentile(per_frame_new_cm, 90):.1f} cm")
    print(f"{'Maximum':<25} {np.max(per_frame_old_cm):<22.1f} {np.max(per_frame_new_cm):.1f} cm")
    print(f"{'Mean':<25} {np.mean(per_frame_old_cm):<22.1f} {np.mean(per_frame_new_cm):.1f} cm")
    print(f"{'Saturated (>=34.9 cm)':<25} {np.sum(np.array(per_frame_old_cm) >= 34.9)} ({np.mean(np.array(per_frame_old_cm) >= 34.9)*100:.1f}%)"
          f"       {np.sum(np.array(per_frame_new_cm) >= 34.9)} ({np.mean(np.array(per_frame_new_cm) >= 34.9)*100:.1f}%)")

    print(f"\n2. UNIQUE VERIFIED POTHOLE AUDIT ({len(old_verified)} Unique Hazards):")
    print(f"{'Track ID':<10} {'Observations':<14} {'OLD Smooth Depth':<18} {'OLD Severity':<16} {'NEW Smooth Depth':<18} {'NEW Severity'}")
    print("-" * 90)

    for tid in sorted(old_verified.keys()):
        o_data = old_verified[tid]
        n_data = new_verified.get(tid, {})
        o_cm = o_data.get("smooth_cm", 0.0)
        n_cm = n_data.get("smooth_cm", 0.0)
        o_sev = o_data.get("severity", "MILD")
        n_sev = n_data.get("severity", "MILD")
        obs = o_data.get("observations", 1)
        print(f"#{tid:<9} {obs:<14} {o_cm:<18.1f} {o_sev:<16} {n_cm:<18.1f} {n_sev}")

    old_sev_counts = defaultdict(int)
    new_sev_counts = defaultdict(int)
    for d in old_verified.values(): old_sev_counts[d["severity"]] += 1
    for d in new_verified.values(): new_sev_counts[d["severity"]] += 1

    print("\n3. FINAL MUNICIPAL AUDIT SEVERITY COUNTS:")
    print(f"  OLD Method: {dict(old_sev_counts)}")
    print(f"  NEW Method: {dict(new_sev_counts)}")

if __name__ == "__main__":
    run_comparison()
