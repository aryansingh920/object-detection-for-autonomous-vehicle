import os
import sys
import json
import random
import platform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import cv2
import numpy as np

class KittiCocoDataset(Dataset):
    """
    A simple PyTorch Dataset that reads:
      - {split}_image.json for image paths
      - {split}_annotations.json for bounding boxes, categories
    and returns images + target dict that can be used by YOLOv5/YOLOv8 style training.
    """

    def __init__(self, image_json_path, annotation_json_path, transform=None):
        super().__init__()
        self.transform = transform

        # Load image info
        with open(image_json_path, 'r') as f:
            image_data = json.load(f)
        self.images = image_data["images"]  # list of dicts

        # Load annotations
        with open(annotation_json_path, 'r') as f:
            ann_data = json.load(f)
        self.categories = ann_data["categories"]
        annotations = ann_data["annotations"]  # list

        # Build a dictionary: image_id -> list of annotations
        self.image_id_to_anns = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            self.image_id_to_anns[img_id].append(ann)

        # Create a category id -> name map if needed
        self.cat_id_to_name = {}
        for cat in self.categories:
            self.cat_id_to_name[cat["id"]] = cat["name"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Info for this image
        img_info = self.images[idx]
        image_id = img_info["id"]
        img_path = img_info["file_name"]  # Full path stored in the JSON

        # Read image (with PIL or OpenCV)
        try:
            # Let's do PIL example
            with Image.open(img_path).convert("RGB") as img:
                image_np = np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"[WARNING] Failed to open image {img_path}: {e}")
            # Return a dummy or raise exception
            raise e

        # Convert from HxWxC to a torch tensor: shape [C,H,W]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        # Get all bounding boxes + labels
        ann_list = self.image_id_to_anns.get(image_id, [])
        boxes = []
        labels = []
        for ann in ann_list:
            # bbox is [x_min, y_min, width, height]
            x, y, w, h = ann["bbox"]
            # YOLO typically uses [x_center, y_center, w, h], normalized
            # but let's just keep [x_min, y_min, x_max, y_max] for demonstration
            x2, y2 = x + w, y + h
            boxes.append([x, y, x2, y2])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            # If no bounding boxes, create a dummy
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Basic transform if needed
        if self.transform is not None:
            # For example, you might do random resizing, flipping, etc.
            # Here we'll skip for brevity.
            pass

        # Construct target dictionary
        target = {
            "boxes": boxes,          # shape [N, 4]
            "labels": labels,        # shape [N]
            "image_id": torch.tensor([image_id], dtype=torch.int64)
        }

        return image_tensor, target

