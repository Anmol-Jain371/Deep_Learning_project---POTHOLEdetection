import os
import json

candidates_dir = "dataset_v2_candidates"
labels_dir = os.path.join(candidates_dir, "labels")
manifest_path = os.path.join(candidates_dir, "review_manifest.json")

jpg_files = sorted([f for f in os.listdir(candidates_dir) if f.lower().endswith((".jpg", ".jpeg"))])
txt_files = sorted([f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")])

print(f"1. Number of JPG images: {len(jpg_files)}")
print(f"2. Number of TXT label files: {len(txt_files)}")

# 3. 1-to-1 Mapping check
missing_labels = []
for jf in jpg_files:
    stem = os.path.splitext(jf)[0]
    expected_txt = f"{stem}.txt"
    if expected_txt not in txt_files:
        missing_labels.append(jf)

mapping_ok = len(missing_labels) == 0 and len(jpg_files) == len(txt_files)
print(f"3. 1-to-1 mapping status: {'PASS (Every image has exactly one TXT label)' if mapping_ok else 'FAIL: ' + str(missing_labels)}")

# 4-11. Validation of boxes
total_rows = 0
per_frame_counts = []
malformed_rows = []
invalid_class_ids = []
out_of_range_coords = []
invalid_dimensions = []
outside_image_boxes = []

for idx, jf in enumerate(jpg_files, 1):
    stem = os.path.splitext(jf)[0]
    txt_path = os.path.join(labels_dir, f"{stem}.txt")
    
    box_count = 0
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        box_count = len(lines)
        total_rows += box_count
        
        for line_num, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                malformed_rows.append((jf, line_num, line))
                continue
            
            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                malformed_rows.append((jf, line_num, line))
                continue
                
            if cls_id != 0:
                invalid_class_ids.append((jf, line_num, cls_id))
            if not (0.0 <= xc <= 1.0) or not (0.0 <= yc <= 1.0):
                out_of_range_coords.append((jf, line_num, xc, yc))
            if w <= 0.0 or h <= 0.0:
                invalid_dimensions.append((jf, line_num, "w/h <= 0", w, h))
            if w > 1.0 or h > 1.0:
                invalid_dimensions.append((jf, line_num, "w/h > 1", w, h))
                
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0
            
            if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                outside_image_boxes.append((jf, line_num, round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)))
                
    per_frame_counts.append((idx, jf, box_count))

print(f"4. Total annotation rows across all 35 TXT files: {total_rows}")

# 12. Review manifest statistics
manifest_stats = {}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    revs = m.get("reviews", {})
    manifest_stats["reviewed"] = len(revs)
    manifest_stats["clean"] = sum(1 for r in revs.values() if r.get("is_clean") or r.get("num_boxes", 0) == 0)
    manifest_stats["pothole_frames"] = sum(1 for r in revs.values() if not r.get("is_clean") and r.get("num_boxes", 0) > 0)
    manifest_stats["total_boxes"] = sum(r.get("num_boxes", 0) for r in revs.values())

print("\n--- DETAILED INTEGRITY CHECKS ---")
print(f"6. Malformed label rows: {len(malformed_rows)} {malformed_rows if malformed_rows else ''}")
print(f"7. Class ID other than 0: {len(invalid_class_ids)} {invalid_class_ids if invalid_class_ids else ''}")
print(f"8. xc, yc outside [0,1]: {len(out_of_range_coords)} {out_of_range_coords if out_of_range_coords else ''}")
print(f"9. Width/height <= 0: {len([x for x in invalid_dimensions if '<=' in x[2]])}")
print(f"10. Width/height > 1: {len([x for x in invalid_dimensions if '>' in x[2]])}")
print(f"11. Calculated x1/y1/x2/y2 outside image: {len(outside_image_boxes)} {outside_image_boxes if outside_image_boxes else ''}")

print("\n--- PER-FRAME BREAKDOWN (5. All 35 frames) ---")
for item in per_frame_counts:
    status = "Clean (0 boxes)" if item[2] == 0 else f"{item[2]} pothole box(es)"
    print(f"  Frame #{item[0]:02d} | {item[1]} -> {status}")

print("\n--- 12. MANIFEST STATISTICS ---")
print(f"  Reviewed Frames : {manifest_stats.get('reviewed', 0)} / {len(jpg_files)}")
print(f"  Clean Frames    : {manifest_stats.get('clean', 0)}")
print(f"  Pothole Frames  : {manifest_stats.get('pothole_frames', 0)}")
print(f"  Manifest Boxes  : {manifest_stats.get('total_boxes', 0)}")

# Conclusion
all_passed = (
    len(jpg_files) == 35 and
    len(txt_files) == 35 and
    mapping_ok and
    len(malformed_rows) == 0 and
    len(invalid_class_ids) == 0 and
    len(out_of_range_coords) == 0 and
    len(invalid_dimensions) == 0 and
    len(outside_image_boxes) == 0 and
    manifest_stats.get("reviewed", 0) == 35
)

print("\n" + "=" * 50)
print(f"OVERALL VALIDATION RESULT: {'PASS' if all_passed else 'FAIL'}")
print("=" * 50)
