"""
Created on 30/12/2024

@author: Aryan

Filename: KITTIMultiModalDataset.py
Relative Path: src/train/KITTIMultiModalDataset.py
"""

import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


class KITTIMultiModalDataset(Dataset):
    """
    Example dataset for TorchVision's pretrained detection models.
    Ignores calibration/Velodyne, and just provides bounding boxes + labels.
    """

    def __init__(
        self,
        coco_dir: Path,
        split: str,            # 'train' or 'val'
        image_size=(640, 640),  # optional resizing
        num_classes=9          # e.g. 8 KITTI classes + 1 for background
    ):
        self.coco_dir = coco_dir
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes

        # Load the splitted JSON that has images & annotations
        # e.g. data/coco/train/train_image.json and data/coco/train/train_annotations.json
        with open(self.coco_dir / split / f"{split}_image.json", "r") as f:
            self.image_data = json.load(f)
        with open(self.coco_dir / split / f"{split}_annotations.json", "r") as f:
            self.ann_data = json.load(f)

        # Convert list of images, list of annotations, etc.
        self.images = self.image_data["images"]  # list of dict
        self.annotations = self.ann_data["annotations"]  # list of dict
        # Build an image_id -> list of annotation dicts
        self.image_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            self.image_id_to_anns[img_id].append(ann)

        # For transformations (resize, convert to tensor, etc.)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): CxHxW
            target (dict): {
               'boxes': FloatTensor[N, 4],
               'labels': LongTensor[N],
               'image_id': tensor([img_id]),
               ...
            }
        """
        image_info = self.images[idx]
        img_id = image_info["id"]
        img_path = image_info["file_name"]  # e.g. data/kitti/...png

        # Load image
        full_img_path = Path(image_info["file_name"])
        image = Image.open(full_img_path).convert("RGB")
        image = self.transform(image)  # => CxHxW

        # Get the original width/height (before we resized)
        orig_w = image_info["width"]
        orig_h = image_info["height"]

        # Retrieve all annotations for this image
        anns = self.image_id_to_anns.get(img_id, [])

        boxes = []
        labels = []
        for ann in anns:
            # ann['bbox'] is [x, y, w, h] in original scale
            x, y, w, h = ann["bbox"]
            # Convert to [x_min, y_min, x_max, y_max]
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

            # We might want to scale these to the new image_size
            # since we did transforms.Resize(...). Let's do that:
            scale_x = self.image_size[0] / orig_w
            scale_y = self.image_size[1] / orig_h
            x_min = x_min * scale_x
            y_min = y_min * scale_y
            x_max = x_max * scale_x
            y_max = y_max * scale_y

            # category_id (1..8), TorchVision detection needs 1..(num_classes-1)
            label = ann["category_id"]  # e.g. 1..8
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # If no annotations, keep empty placeholders
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,        # shape [N, 4]
            "labels": labels,      # shape [N]
            "image_id": torch.tensor([img_id]),
        }

        return image, target
