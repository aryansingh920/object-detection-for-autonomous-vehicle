"""
Created on 30/12/2024

@author: Aryan

Filename: YOLO_model.py

Relative Path: src/train/YOLO_model.py
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


class YOLOModel(nn.Module):
    def __init__(self, num_classes: int, modalities: list):
        super(YOLOModel, self).__init__()
        # Define your model architecture here
        self.num_classes = num_classes
        # Example: Simple multi-modal input
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.calib_branch = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU()
        )
        self.velodyne_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU()
        )
        # Adjust based on your architecture
        self.fc = nn.Linear(16 * 2 + 16, num_classes)

    def forward(self, images, calib, velodyne):
        img_features = self.image_branch(images)
        img_features = torch.flatten(img_features, start_dim=1)
        calib_features = self.calib_branch(calib)
        velodyne_features = self.velodyne_branch(velodyne)
        combined = torch.cat(
            [img_features, calib_features, velodyne_features], dim=1)
        output = self.fc(combined)
        return output
