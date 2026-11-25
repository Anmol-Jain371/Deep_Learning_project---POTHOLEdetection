"""
========================================================================================
RoadVision AI -- Unified Zero-Leakage Dataset Builder
========================================================================================
Consolidates non-contaminated pothole datasets into a strict 3-way split:
  - Train (~80% RDD2022 India + all local phone V1 frames incl. clean-road negatives)
  - Val   (~10% RDD2022 India, strictly unseen during gradient updates)
  - Test  (~10% RDD2022 India, strictly held-out for final paper metrics)

And preserves Rangoon Road frames (`rangoon_*` from dataset_combined_v2) strictly
isolated in `external_test` so the Rangoon Road benchmark video has ZERO leakage.

Verifies MD5 image hashes across every split to guarantee zero duplicate leakage.
========================================================================================
"""

import os
import sys
import hashlib
import shutil
import random
import yaml
from collections import Counter

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RDD_DIR = os.path.join(BASE_DIR, "dataset_rdd_india_potholes")
PHONE_DIR = os.path.join(BASE_DIR, "dataset_phone_potholes")
COMBINED_V2_DIR = os.path.join(BASE_DIR, "dataset_combined_v2")
OUT_DIR = os.path.join(BASE_DIR, "dataset_unified")


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_label_lines(lbl_path: str):
    if not os.path.exists(lbl_path):
        return []
    lines = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Normalize all pothole annotations to class 0
                xc, yc, w, h = map(float, parts[1:5])
                if w > 0 and h > 0:
                    lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return lines


def collect_samples():
    seen_hashes = {}
    rdd_samples = []
    phone_samples = []
    rangoon_external = []

    # 1. Collect RDD2022 India samples from dataset_rdd_india_potholes
    for split in ["train", "val"]:
        img_dir = os.path.join(RDD_DIR, "images", split)
        lbl_dir = os.path.join(RDD_DIR, "labels", split)
        if not os.path.exists(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            h = md5_file(img_path)
            if h in seen_hashes:
                continue
            seen_hashes[h] = fname
            labels = load_label_lines(lbl_path)
            sample = {
                "filename": fname,
                "img_path": img_path,
                "labels": labels,
                "hash": h,
                "source": "rdd_india" if fname.startswith("rdd_") else "phone_v1"
            }
            if sample["source"] == "rdd_india":
                rdd_samples.append(sample)
            else:
                phone_samples.append(sample)

    # 2. Add any remaining phone samples (e.g., clean-road negative frames with empty labels)
    for split in ["train", "val"]:
        img_dir = os.path.join(PHONE_DIR, "images", split)
        lbl_dir = os.path.join(PHONE_DIR, "labels", split)
        if not os.path.exists(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            h = md5_file(img_path)
            if h in seen_hashes:
                continue
            seen_hashes[h] = fname
            labels = load_label_lines(lbl_path)
            phone_samples.append({
                "filename": f"v1_{fname}",
                "img_path": img_path,
                "labels": labels,
                "hash": h,
                "source": "phone_v1"
            })

    # 3. Collect Rangoon Road labeled frames ONLY for isolated external_test (never train/val)
    for split in ["train", "val"]:
        img_dir = os.path.join(COMBINED_V2_DIR, "images", split)
        lbl_dir = os.path.join(COMBINED_V2_DIR, "labels", split)
        if not os.path.exists(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.startswith("rangoon_"):
                continue
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            h = md5_file(img_path)
            if h in seen_hashes:
                continue
            seen_hashes[h] = fname
            labels = load_label_lines(lbl_path)
            rangoon_external.append({
                "filename": fname,
                "img_path": img_path,
                "labels": labels,
                "hash": h,
                "source": "rangoon_external"
            })

    return rdd_samples, phone_samples, rangoon_external


def write_split(samples, split_name):
    img_out_dir = os.path.join(OUT_DIR, "images", split_name)
    lbl_out_dir = os.path.join(OUT_DIR, "labels", split_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    box_count = 0
    neg_count = 0
    for s in samples:
        dst_img = os.path.join(img_out_dir, s["filename"])
        dst_lbl = os.path.join(lbl_out_dir, os.path.splitext(s["filename"])[0] + ".txt")
        shutil.copy2(s["img_path"], dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.writelines(s["labels"])
        if len(s["labels"]) == 0:
            neg_count += 1
        box_count += len(s["labels"])

    return len(samples), box_count, neg_count


def main(seed: int = 42):
    random.seed(seed)

    if os.path.exists(OUT_DIR):
        print(f"Cleaning previous {OUT_DIR}...")
        shutil.rmtree(OUT_DIR)

    rdd_samples, phone_samples, rangoon_external = collect_samples()

    # Deterministic 80 / 10 / 10 split of RDD2022 India
    random.shuffle(rdd_samples)
    n_total_rdd = len(rdd_samples)
    n_val = int(round(n_total_rdd * 0.10))
    n_test = int(round(n_total_rdd * 0.10))

    val_samples = rdd_samples[:n_val]
    test_samples = rdd_samples[n_val:n_val + n_test]
    train_rdd = rdd_samples[n_val + n_test:]

    # Keep all sequential phone_v1 frames strictly in train to prevent temporal leakage
    train_samples = train_rdd + phone_samples

    # Verify zero MD5 overlap between any pair of splits
    split_hashes = {
        "train": {s["hash"] for s in train_samples},
        "val": {s["hash"] for s in val_samples},
        "test": {s["hash"] for s in test_samples},
        "external_rangoon": {s["hash"] for s in rangoon_external},
    }
    names = list(split_hashes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_hashes[names[i]] & split_hashes[names[j]]
            assert len(overlap) == 0, f"Leakage detected between {names[i]} and {names[j]}: {len(overlap)}"

    tr_n, tr_b, tr_neg = write_split(train_samples, "train")
    va_n, va_b, va_neg = write_split(val_samples, "val")
    te_n, te_b, te_neg = write_split(test_samples, "test")
    ex_n, ex_b, ex_neg = write_split(rangoon_external, "external_rangoon")

    # Relative path data.yaml for portability across environments
    yaml_content = {
        "path": "dataset_unified",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
        "nc": 1
    }
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)

    # Separate YAML for external Rangoon Road frame evaluation
    ext_yaml_content = {
        "path": "dataset_unified",
        "train": "images/train",
        "val": "images/external_rangoon",
        "names": {0: "pothole"},
        "nc": 1
    }
    ext_yaml_path = os.path.join(OUT_DIR, "data_external_rangoon.yaml")
    with open(ext_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(ext_yaml_content, f, sort_keys=False, default_flow_style=False)

    print("=" * 75)
    print("UNIFIED ZERO-LEAKAGE DATASET BUILT SUCCESSFULLY")
    print("=" * 75)
    print(f"Train            : {tr_n} imgs ({tr_b} boxes, {tr_neg} negative frames) [{len(train_rdd)} RDD + {len(phone_samples)} Phone]")
    print(f"Val              : {va_n} imgs ({va_b} boxes, {va_neg} negative frames) [100% RDD India]")
    print(f"Test (Held-Out)  : {te_n} imgs ({te_b} boxes, {te_neg} negative frames) [100% RDD India]")
    print(f"External Rangoon : {ex_n} imgs ({ex_b} boxes, {ex_neg} negative frames) [Zero-leakage OOD test]")
    print(f"MD5 Overlap Check: PASSED (0 duplicates across all 4 splits)")
    print(f"Primary YAML     : {yaml_path}")
    print(f"External YAML    : {ext_yaml_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
