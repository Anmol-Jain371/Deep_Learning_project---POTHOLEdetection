"""
========================================================================================
🏗️ RoadVision AI — RDD2022 India (D40 Pothole) Dataset Builder
========================================================================================
Extracts and converts the official RDD2022 India subset from CRDDC (Sekilab):
1. Filters XML annotations for class D40 (Potholes)
2. Converts Pascal VOC (xmin, ymin, xmax, ymax) -> YOLO (class xc yc w h)
3. Extracts corresponding images directly from RDD2022/India.zip
4. Integrates with local smartphone dataset V1 (dataset_phone_potholes/)
5. Preserves 100% of Rangoon Road as completely blind, unseen external test data.
========================================================================================
"""

import os
import sys
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
import random
import yaml

def build_dataset(
    zip_path: str = "RDD2022/India.zip",
    v1_dir: str = "dataset_phone_potholes",
    output_dir: str = "dataset_rdd_india_potholes",
    val_ratio: float = 0.15,
    seed: int = 42
):
    random.seed(seed)
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Archive not found: {zip_path}")
        
    print("=" * 80)
    print("ROADVISION AI: RDD2022 INDIA (D40) DATASET BUILDER")
    print(f"Archive: {zip_path}")
    print(f"Output : {output_dir}")
    print("=" * 80)

    # Cleanly remove existing output directory to prevent stale/duplicate files
    if os.path.exists(output_dir):
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Prepare directories
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    extracted_samples = []
    total_d40_boxes = 0

    print("\n1. Scanning and parsing RDD2022 India annotations...")
    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = set(z.namelist())
        xml_files = [f for f in all_files if f.startswith("India/train/annotations/xmls/") and f.endswith(".xml")]
        print(f"   Found {len(xml_files)} total training XML annotations.")

        for xml_path in xml_files:
            with z.open(xml_path) as f:
                tree = ET.parse(f)
                root = tree.getroot()

                # Get image dimensions
                size_elem = root.find("size")
                if size_elem is not None:
                    width = float(size_elem.find("width").text)
                    height = float(size_elem.find("height").text)
                else:
                    # Fallback standard RDD India smartphone resolution
                    width, height = 720.0, 720.0

                boxes = []
                for obj in root.findall("object"):
                    cname = obj.find("name").text.strip()
                    if cname == "D40":  # Pothole
                        bnd = obj.find("bndbox")
                        xmin = float(bnd.find("xmin").text)
                        ymin = float(bnd.find("ymin").text)
                        xmax = float(bnd.find("xmax").text)
                        ymax = float(bnd.find("ymax").text)

                        # Clamp coordinates
                        xmin = max(0.0, min(width, xmin))
                        xmax = max(0.0, min(width, xmax))
                        ymin = max(0.0, min(height, ymin))
                        ymax = max(0.0, min(height, ymax))

                        bw = xmax - xmin
                        bh = ymax - ymin
                        if bw > 1 and bh > 1:
                            xc = (xmin + xmax) / 2.0 / width
                            yc = (ymin + ymax) / 2.0 / height
                            w_norm = bw / width
                            h_norm = bh / height
                            boxes.append((0, xc, yc, w_norm, h_norm))

                if boxes:
                    # Corresponding image path
                    base_name = os.path.splitext(os.path.basename(xml_path))[0]
                    img_path_in_zip = f"India/train/images/{base_name}.jpg"
                    if img_path_in_zip in all_files:
                        extracted_samples.append({
                            "type": "rdd2022",
                            "img_zip_path": img_path_in_zip,
                            "filename": f"rdd_in_{base_name}.jpg",
                            "boxes": boxes
                        })
                        total_d40_boxes += len(boxes)

        print(f"   Extracted {len(extracted_samples)} RDD2022 India images with {total_d40_boxes} D40 pothole boxes.")

        # 2. Add Local Dataset V1 (dataset_phone_potholes) if available
        v1_count = 0
        v1_boxes_count = 0
        if os.path.exists(v1_dir):
            print("\n2. Integrating local phone dataset V1 (dataset_phone_potholes)...")
            v1_imgs_dir = os.path.join(v1_dir, "images")
            v1_lbls_dir = os.path.join(v1_dir, "labels")

            for root_d, _, files in os.walk(v1_imgs_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        full_img = os.path.join(root_d, f)
                        base = os.path.splitext(f)[0]
                        lbl_file = os.path.join(v1_lbls_dir, f"{base}.txt")
                        if not os.path.exists(lbl_file):
                            # Try looking inside subdirs
                            for s_root, _, s_files in os.walk(v1_lbls_dir):
                                if f"{base}.txt" in s_files:
                                    lbl_file = os.path.join(s_root, f"{base}.txt")
                                    break

                        if os.path.exists(lbl_file):
                            boxes = []
                            with open(lbl_file, "r") as lf:
                                for line in lf:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        cid = int(parts[0])
                                        xc, yc, w, h = map(float, parts[1:5])
                                        boxes.append((0, xc, yc, w, h))
                            if boxes:
                                extracted_samples.append({
                                    "type": "local_v1",
                                    "full_img_path": full_img,
                                    "filename": f"v1_{f}",
                                    "boxes": boxes
                                })
                                v1_count += 1
                                v1_boxes_count += len(boxes)

            print(f"   Integrated {v1_count} local V1 phone images with {v1_boxes_count} pothole boxes.")

        # 3. Train / Val Split (Strict Zero-Leakage Guarantee)
        # All local V1 sequence frames go to TRAIN to prevent temporal leakage between adjacent frames.
        # Validation is drawn strictly from independent RDD2022 India road images.
        print("\n3. Creating zero-leakage split (all local V1 in train, val strictly from RDD2022)...")
        rdd_samples = [s for s in extracted_samples if s["type"] == "rdd2022"]
        local_samples = [s for s in extracted_samples if s["type"] == "local_v1"]

        random.shuffle(rdd_samples)
        n_val = int(len(rdd_samples) * val_ratio)
        val_samples = rdd_samples[:n_val]
        train_samples = rdd_samples[n_val:] + local_samples

        print(f"   Training images  : {len(train_samples)} ({len(train_samples) - len(local_samples)} RDD + {len(local_samples)} Local V1)")
        print(f"   Validation images: {len(val_samples)} (100% independent RDD2022 India)")
        print(f"   Total images     : {len(extracted_samples)}")
        print("   Zero temporal leakage: No V1 sequential frames in validation!")

        # 4. Extract and Write Files
        print("\n4. Writing dataset files to disk...")
        train_box_total = 0
        val_box_total = 0

        def save_sample(sample, target_img_dir, target_lbl_dir):
            nonlocal train_box_total, val_box_total
            img_out = os.path.join(target_img_dir, sample["filename"])
            lbl_out = os.path.join(target_lbl_dir, os.path.splitext(sample["filename"])[0] + ".txt")

            # Extract image
            if sample["type"] == "rdd2022":
                with z.open(sample["img_zip_path"]) as img_in, open(img_out, "wb") as img_f:
                    shutil.copyfileobj(img_in, img_f)
            else:
                shutil.copy2(sample["full_img_path"], img_out)

            # Write YOLO label
            with open(lbl_out, "w") as lf:
                for b in sample["boxes"]:
                    lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

            return len(sample["boxes"])

        for s in train_samples:
            train_box_total += save_sample(s, train_img_dir, train_lbl_dir)

        for s in val_samples:
            val_box_total += save_sample(s, val_img_dir, val_lbl_dir)

    # 5. Write data.yaml
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    abs_out = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = {
        "path": abs_out,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pothole"},
        "nc": 1
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("\n" + "=" * 80)
    print("DATASET CONSTRUCTION COMPLETE: RDD2022 INDIA + LOCAL V1")
    print("=" * 80)
    print(f"Configuration YAML : {data_yaml_path}")
    print(f"Total Images       : {len(extracted_samples)}")
    print(f"  - Train Images   : {len(train_samples)} ({train_box_total} pothole boxes)")
    print(f"  - Val Images     : {len(val_samples)} ({val_box_total} pothole boxes)")
    print(f"Total Bounding Boxes: {train_box_total + val_box_total}")
    print(f"External Benchmark : Rangoon Road (1,005 frames @ 25 FPS) - 100% BLIND TEST DATA")
    print("=" * 80)

if __name__ == "__main__":
    build_dataset()
