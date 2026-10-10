"""
========================================================================================
📸 Dataset V2: Systematic Diversity Extractor for Rangoon Road Footage
========================================================================================
Extracts 35 diverse, sharp frames evenly distributed across the 1,005 frames (40.2s).
Within each ~28-frame window (~1.1s), selects the sharpest frame (highest Laplacian variance)
to eliminate motion blur and capture distinct road states:
  - Near-field craters
  - Distant potholes & road perspective
  - Shadowed road vs bright sunlit asphalt
  - Clean asphalt transition zones (negative samples)
  - Severe multi-pothole clusters
========================================================================================
"""

import os
import sys
import cv2
import numpy as np

def main():
    video_path = "Watch Video_ Terrible potholes in Rangoon Road.mp4"
    out_dir = "dataset_v2_candidates"
    os.makedirs(out_dir, exist_ok=True)

    # Clean out any old candidate files in out_dir
    for f in os.listdir(out_dir):
        fp = os.path.join(out_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_count = 35
    window_size = total_frames / target_count

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print(f"Video: {video_path} ({total_frames} frames @ {fps:.1f} FPS, {W}x{H})")
    print(f"Extracting {target_count} diverse frames (1 per ~{window_size:.1f} frames / ~{window_size/fps:.2f}s)...")

    catalog = []

    for i in range(target_count):
        start_f = int(i * window_size)
        end_f = int(min(total_frames, (i + 1) * window_size))

        best_frame = None
        best_f_idx = start_f
        best_sharpness = -1.0
        best_brightness = 0.0

        # Scan candidate frames inside this window (step by 2 to find peak sharpness)
        for f_idx in range(start_f, end_f, 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = float(np.mean(gray))

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_brightness = brightness
                best_f_idx = f_idx
                best_frame = frame.copy()

        if best_frame is not None:
            rank = i + 1
            filename = f"rangoon_v2_{rank:02d}_f{best_f_idx:04d}.jpg"
            save_path = os.path.join(out_dir, filename)
            cv2.imwrite(save_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            timestamp_s = best_f_idx / fps

            # Categorize the frame by its visual environment and timestamp
            # Lighting
            if best_brightness < 125:
                light_tag = "Shadowed / Low-Contrast"
            elif best_brightness > 145:
                light_tag = "Bright Daylight"
            else:
                light_tag = "Balanced Road Daylight"

            # Road Zone Analysis
            if best_f_idx < 150:
                zone_desc = "Initial Approach (Variable Surface, Scattered Depressions)"
            elif 150 <= best_f_idx < 400:
                zone_desc = "Severe Pothole Cluster & Water/Mud Cavities"
            elif 400 <= best_f_idx < 600:
                zone_desc = "Mixed Asphalt with Patched & Rough Road Transition"
            elif 600 <= best_f_idx < 850:
                zone_desc = "Deep Impact Craters in Forward Driving Corridor"
            else:
                zone_desc = "Exit Corridor (Wide View, Distant Cavities, Clean Edges)"

            catalog.append({
                "rank": rank,
                "filename": filename,
                "path": save_path,
                "frame_idx": best_f_idx,
                "timestamp": f"{timestamp_s:.2f}s",
                "sharpness": round(best_sharpness, 1),
                "brightness": round(best_brightness, 1),
                "lighting": light_tag,
                "zone_desc": zone_desc
            })

    cap.release()

    # Write Markdown catalog
    catalog_md = os.path.join(out_dir, "CANDIDATE_FRAMES_CATALOG.md")
    with open(catalog_md, "w", encoding="utf-8") as f:
        f.write("# 📋 Dataset V2: 35 Selected Candidate Frames from Rangoon Road\n\n")
        f.write(f"**Source Video:** `{video_path}` (1,005 frames @ 25 FPS)\n")
        f.write(f"**Total Frames Extracted:** {len(catalog)} non-consecutive frames\n")
        f.write(f"**Target Quality:** Peak sharpness within each 1.1s temporal window (zero motion blur)\n")
        f.write(f"**Storage Directory:** `{out_dir}/`\n\n")
        f.write("| # | Filename | Time (s) | Frame # | Lighting Condition | Road Zone / Scene Type | Sharpness |\n")
        f.write("|:---:|:---|:---:|:---:|:---|:---|:---:|\n")
        for c in catalog:
            f.write(f"| {c['rank']:02d} | `{c['filename']}` | {c['timestamp']} | {c['frame_idx']} | {c['lighting']} | {c['zone_desc']} | {c['sharpness']} |\n")

    print(f"\n[OK] Successfully extracted {len(catalog)} frames to: {out_dir}")
    print(f"Catalog saved to: {catalog_md}")

if __name__ == "__main__":
    main()
