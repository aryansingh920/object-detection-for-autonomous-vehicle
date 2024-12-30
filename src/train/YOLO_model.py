"""
Created on 30/12/2024

@author: Aryan

Filename: KITTIMultiModal.py

Relative Path: src/train/KITTIMultiModal.py
"""

import json
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from tqdm import tqdm
# from src.train.multi import KITTIMultiModalDataset


class KITTIMultiModalDataset(Dataset):
    def __init__(self, coco_dir, split, config):
        self.coco_dir = Path(coco_dir) / split
        self.config = config
        self.split = split
        self.load_annotations()
        self.transforms = Compose(
            [Resize((config['image_height'], config['image_width'])), ToTensor()])

    def load_annotations(self):
        with open(self.coco_dir / f"{self.split}_image.json", "r") as f:
            self.image_data = json.load(f)

    def __len__(self):
        return len(self.image_data['images'])

    def __getitem__(self, idx):
        image_info = self.image_data['images'][idx]
        image_path = image_info['file_name']
        image = Image.open(image_path).convert("RGB")
        width, height = image.size  # Get the image dimensions

        annotations = [ann for ann in self.image_data['annotations']
                       if ann['image_id'] == image_info['id']]

        valid_annotations = []
        for ann in annotations:
            x_min, y_min, box_width, box_height = ann['bbox']
            x_max = x_min + box_width
            y_max = y_min + box_height

            # Ensure bounding box is within image dimensions and has positive area
            if box_width > 0 and box_height > 0 and x_max <= width and y_max <= height:
                valid_annotations.append(ann)
            # else:
                # logger.warning(
                #     f"Invalid bbox {ann['bbox']} in image {image_path}. Skipping.")

        if not valid_annotations:
            # Fallback if no valid annotations exist
            valid_annotations = [{"bbox": [0, 0, 1, 1], "category_id": 0}]

        # Convert to tensors
        boxes = torch.tensor([ann['bbox']
                              for ann in valid_annotations], dtype=torch.float32)
        labels = torch.tensor([ann['category_id']
                               for ann in valid_annotations], dtype=torch.int64)

        return self.transforms(image), {"boxes": boxes, "labels": labels}

# File: YOLO_model.py (Not required, using torchvision model)
