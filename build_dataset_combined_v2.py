"""
========================================================================================
📦 Dataset Combined V2 Builder
========================================================================================
Combines:
  1. Dataset V1 (dataset_phone_potholes/): 70 images (56 train / 14 val)
  2. Dataset V2 (dataset_v2_candidates/): 35 images (28 train / 7 val)

Produces:
  dataset_combined_v2/
  ├── images/
  │   ├── train/ (84 images)
  │   └── val/   (21 images)
  ├── labels/
  │   ├── train/ (84 TXT files)
  │   └── val/   (21 TXT files)
  └── data.yaml

Strictly read-only with respect to sources (copies only).
========================================================================================
"""

import os
import shutil
import glob
import yaml

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
V1_DIR = os.path.join(BASE_DIR, "dataset_phone_potholes")
V2_DIR = os.path.join(BASE_DIR, "dataset_v2_candidates")
OUT_DIR = os.path.join(BASE_DIR, "dataset_combined_v2")

def main():
    print("Building Dataset Combined V2...")

    # Create destination directories
    img_train_dir = os.path.join(OUT_DIR, "images", "train")
    img_val_dir = os.path.join(OUT_DIR, "images", "val")
    lbl_train_dir = os.path.join(OUT_DIR, "labels", "train")
    lbl_val_dir = os.path.join(OUT_DIR, "labels", "val")

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Copy Dataset V1 (56 train / 14 val)
    v1_train_imgs = glob.glob(os.path.join(V1_DIR, "images", "train", "*.*"))
    v1_val_imgs = glob.glob(os.path.join(V1_DIR, "images", "val", "*.*"))

    for img_path in v1_train_imgs:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(V1_DIR, "labels", "train", f"{stem}.txt")
        shutil.copy2(img_path, os.path.join(img_train_dir, fname))
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(lbl_train_dir, f"{stem}.txt"))
        else:
            # Empty label if missing
            open(os.path.join(lbl_train_dir, f"{stem}.txt"), "w").close()

    for img_path in v1_val_imgs:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        lbl_path = os.path.join(V1_DIR, "labels", "val", f"{stem}.txt")
        shutil.copy2(img_path, os.path.join(img_val_dir, fname))
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(lbl_val_dir, f"{stem}.txt"))
        else:
            open(os.path.join(lbl_val_dir, f"{stem}.txt"), "w").close()

    print(f"Copied V1: {len(v1_train_imgs)} train, {len(v1_val_imgs)} val")

    # 2. Split and Copy Dataset V2 (35 images: 28 train / 7 val)
    # Val indices: every 5th frame [5, 10, 15, 20, 25, 30, 35]
    val_indices = {5, 10, 15, 20, 25, 30, 35}

    v2_jpgs = sorted([f for f in os.listdir(V2_DIR) if f.lower().endswith((".jpg", ".jpeg"))])
    v2_labels_dir = os.path.join(V2_DIR, "labels")

    v2_train_count = 0
    v2_val_count = 0

    for idx, fname in enumerate(v2_jpgs, 1):
        stem = os.path.splitext(fname)[0]
        src_img = os.path.join(V2_DIR, fname)
        src_lbl = os.path.join(v2_labels_dir, f"{stem}.txt")

        is_val = idx in val_indices
        dst_img_dir = img_val_dir if is_val else img_train_dir
        dst_lbl_dir = lbl_val_dir if is_val else lbl_train_dir

        shutil.copy2(src_img, os.path.join(dst_img_dir, fname))
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(dst_lbl_dir, f"{stem}.txt"))
        else:
            open(os.path.join(dst_lbl_dir, f"{stem}.txt"), "w").close()

        if is_val:
            v2_val_count += 1
        else:
            v2_train_count += 1

    print(f"Copied V2: {v2_train_count} train, {v2_val_count} val")

    # 3. Create data.yaml
    data_yaml_path = os.path.join(OUT_DIR, "data.yaml")
    yaml_content = {
        "path": OUT_DIR.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "pothole"
        }
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

    print(f"Created: {data_yaml_path}")

    # 4. Verification & Box Count Telemetry
    def count_boxes(dir_path):
        total = 0
        txts = glob.glob(os.path.join(dir_path, "*.txt"))
        for tp in txts:
            with open(tp, "r", encoding="utf-8") as fp:
                total += sum(1 for line in fp if line.strip())
        return total, len(txts)

    train_boxes, train_lbl_count = count_boxes(lbl_train_dir)
    val_boxes, val_lbl_count = count_boxes(lbl_val_dir)
    train_img_count = len(os.listdir(img_train_dir))
    val_img_count = len(os.listdir(img_val_dir))

    print("\n--- FINAL DATASET COMBINED V2 SUMMARY ---")
    print(f"1. Total Images         : {train_img_count + val_img_count}")
    print(f"2. Train Images         : {train_img_count}")
    print(f"3. Validation Images    : {val_img_count}")
    print(f"4. Total Bounding Boxes : {train_boxes + val_boxes}")
    print(f"5. Train Bounding Boxes : {train_boxes}")
    print(f"6. Val Bounding Boxes   : {val_boxes}")
    print(f"7. Number of V1 Images  : {len(v1_train_imgs) + len(v1_val_imgs)}")
    print(f"8. Number of V2 Images  : {v2_train_count + v2_val_count}")
    print(f"9. Class Names          : {yaml_content['names']}")
    print(f"10. 1-to-1 Mapping      : Train {train_img_count} == {train_lbl_count}, Val {val_img_count} == {val_lbl_count}")

if __name__ == "__main__":
    main()
