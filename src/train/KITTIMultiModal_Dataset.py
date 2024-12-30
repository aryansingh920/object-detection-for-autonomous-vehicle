"""
Created on 30/12/2024

@author: Aryan

Filename: KITTIMultiModal_Dataset.py

Relative Path: src/train/KITTIMultiModal_Dataset.py
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from config.config import YOLOConfig
from PIL import Image
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm
import os


class KITTIMultiModalDataset(Dataset):
    """Dataset class for loading KITTI multi-modal data."""

    def __init__(
        self,
        coco_dir: Path,
        split: str,
        config: YOLOConfig
    ):
        self.coco_dir = Path(coco_dir)
        self.split = split
        self.config = config

        # Load annotations
        self.load_annotations()
        self.setup_transforms()

    def load_annotations(self):
        """Load COCO format annotations for all modalities."""
        # Load image annotations
        image_json_path = self.coco_dir / \
            self.split / f"{self.split}_image.json"
        with open(image_json_path, 'r') as f:
            self.image_data = json.load(f)

        # Load calibration data
        calib_json_path = self.coco_dir / \
            self.split / f"{self.split}_calib.json"
        with open(calib_json_path, 'r') as f:
            self.calib_data = json.load(f)

        # Load Velodyne data
        velodyne_json_path = self.coco_dir / \
            self.split / f"{self.split}_velodyne.json"
        with open(velodyne_json_path, 'r') as f:
            self.velodyne_data = json.load(f)

        # Load annotations
        annotations_json_path = self.coco_dir / \
            self.split / f"{self.split}_annotations.json"
        if not annotations_json_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {annotations_json_path}")
        with open(annotations_json_path, 'r') as f:
            self.annotations = json.load(f)

        # Create index mappings
        self.create_index_mappings()

    def create_index_mappings(self):
        """Create mappings between different modalities."""
        self.image_to_calib = {
            img['id']: calib['id']
            for img, calib in zip(self.image_data['images'], self.calib_data['calibration'])
        }
        self.image_to_velodyne = {
            img['id']: vel['id']
            for img, vel in zip(self.image_data['images'], self.velodyne_data['velodyne'])
        }

    def setup_transforms(self):
        """Set up data augmentation transforms."""
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.config.defaults['image_height'], self.config.defaults['image_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.defaults.get('mean', [0.485, 0.456, 0.406]),
                                 std=self.config.defaults.get('std', [0.229, 0.224, 0.225]))
        ])

    def __len__(self) -> int:
        return len(self.image_data['images'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get multi-modal data item."""
        # Get image data
        image_info = self.image_data['images'][idx]
        image_id = image_info['id']

        # Load image
        image = self.load_image(image_info['file_name'])

        # Load corresponding calibration data
        calib_id = self.image_to_calib.get(image_id)
        calib = self.load_calibration(
            self.calib_data['calibration'][calib_id]['file_name']
        )

        # Load corresponding Velodyne data
        velodyne_id = self.image_to_velodyne.get(image_id)
        velodyne = self.load_velodyne(
            self.velodyne_data['velodyne'][velodyne_id]['file_name']
        )

        # Get annotations for this image
        annotations = [
            ann for ann in self.annotations['annotations']
            if ann['image_id'] == image_id
        ]

        # Convert annotations to YOLO format
        targets = self.convert_annotations(annotations, image_info)

        return {
            'image': image,
            'calib': calib,
            'velodyne': velodyne,
            'targets': targets
        }

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        full_image_path = self.coco_dir / self.split / image_path
        image = Image.open(full_image_path).convert('RGB')
        image = self.transform(image)
        return image

    def load_calibration(self, calib_path: str) -> torch.Tensor:
        """Load and preprocess calibration data."""
        full_calib_path = self.coco_dir / self.split / calib_path
        with open(full_calib_path, 'r') as f:
            calib = json.load(f)
        # Convert calibration data to tensor (this is a placeholder, adjust as needed)
        calib_tensor = torch.tensor(
            calib['calibration_matrix'], dtype=torch.float32)
        return calib_tensor

    def load_velodyne(self, velodyne_path: str) -> torch.Tensor:
        """Load and preprocess Velodyne point cloud data."""
        full_velodyne_path = self.coco_dir / self.split / velodyne_path
        with open(full_velodyne_path, 'r') as f:
            velodyne = json.load(f)
        # Convert Velodyne data to tensor (this is a placeholder, adjust as needed)
        velodyne_points = np.array(velodyne['points'], dtype=np.float32)
        velodyne_tensor = torch.from_numpy(velodyne_points)
        return velodyne_tensor

    def convert_annotations(
        self,
        annotations: list,
        image_info: dict
    ) -> torch.Tensor:
        """Convert COCO annotations to YOLO format."""
        yolo_targets = []
        img_width = image_info['width']
        img_height = image_info['height']

        for ann in annotations:
            # COCO format: [x_min, y_min, width, height]
            bbox = ann['bbox']
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            class_id = ann['category_id']
            yolo_targets.append([class_id, x_center, y_center, width, height])

        if yolo_targets:
            yolo_tensor = torch.tensor(yolo_targets, dtype=torch.float32)
        else:
            yolo_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return yolo_tensor
