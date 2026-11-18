"""
========================================================================================
🚗 RoadVision AI — PyTorch Dataset for Faster R-CNN (RDD2022 India D40 Potholes)
========================================================================================
Parses Pascal VOC XML annotations from RDD2022 India subset and prepares PyTorch tensors:
  - Input: BGR/RGB image tensor (C, H, W) normalized to [0, 1]
  - Target: dict containing:
      - 'boxes': FloatTensor of shape (N, 4) in [xmin, ymin, xmax, ymax] absolute coords
      - 'labels': Int64Tensor of shape (N,) (1 for Pothole, 0 is Background)
      - 'image_id': tensor([idx])
      - 'area': FloatTensor of shape (N,)
      - 'iscrowd': Uint8Tensor of shape (N,)
========================================================================================
"""

import os
import glob
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F


class RDDPotholeDataset(Dataset):
    """
    Dataset loader directly parsing RDD2022 India training directory for D40 (pothole) labels.
    Ensures absolute Pascal VOC coordinate format compatible with Torchvision Object Detection.
    """

    def __init__(
        self,
        root_dir: str = "RDD2022/India/train",
        transforms: Optional[Any] = None,
        filter_empty: bool = True
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, "images")
        self.xml_dir = os.path.join(root_dir, "annotations", "xmls")

        if not os.path.exists(self.xml_dir):
            raise FileNotFoundError(f"XML annotation folder not found: {self.xml_dir}")

        all_xmls = sorted(glob.glob(os.path.join(self.xml_dir, "*.xml")))
        self.samples: List[Dict[str, Any]] = []

        print(f"🔍 Indexing D40 (Pothole) annotations from {len(all_xmls)} XMLs...")
        for xml_path in all_xmls:
            base_name = os.path.splitext(os.path.basename(xml_path))[0]
            img_path = os.path.join(self.img_dir, f"{base_name}.jpg")

            if not os.path.exists(img_path):
                continue

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                boxes = []
                for obj in root.findall("object"):
                    cname = obj.find("name").text.strip()
                    if cname == "D40":  # Pothole
                        bnd = obj.find("bndbox")
                        xmin = float(bnd.find("xmin").text)
                        ymin = float(bnd.find("ymin").text)
                        xmax = float(bnd.find("xmax").text)
                        ymax = float(bnd.find("ymax").text)

                        # Enforce valid box geometry
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])

                if filter_empty and len(boxes) == 0:
                    continue

                self.samples.append({
                    "img_path": img_path,
                    "boxes": boxes,
                    "base_name": base_name
                })
            except Exception:
                continue

        print(f"✅ Loaded {len(self.samples)} valid road images containing potholes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        w, h = img.size

        raw_boxes = sample["boxes"]
        boxes = []
        for b in raw_boxes:
            xmin = max(0.0, min(float(w), b[0]))
            ymin = max(0.0, min(float(h), b[1]))
            xmax = max(0.0, min(float(w), b[2]))
            ymax = max(0.0, min(float(h), b[3]))
            if (xmax - xmin) > 1.0 and (ymax - ymin) > 1.0:
                boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones((len(boxes),), dtype=torch.int64)  # 1 = Pothole
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area_tensor,
            "iscrowd": torch.zeros((len(boxes_tensor),), dtype=torch.int64)
        }

        # Basic transform: PIL -> Tensor
        img_tensor = F.to_tensor(img)

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    """Custom collator handling variable box counts per image."""
    return tuple(zip(*batch))

