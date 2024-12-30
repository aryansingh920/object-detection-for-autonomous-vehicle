"""
Created on 29/12/2024

@author: Aryan

Filename: config.py

Relative Path: src/config/config.py
"""

from typing import Any, Dict, Optional
import yaml


class Config:
    # Define paths (Update these paths as per your directory structure)
    kitti_base_path = r"data/kitti"
    coco_base_path = r"data/coco"

    # Preprocessing settings
    preprocess = False
    train_val_ratio = 0.8
    normalize = False
    target_size_for_normalization = (1242, 375)

    # Training settings
    train = True

    # Model paths
    saved_model_path = r"saved_model"
    output_path = r"output"


class YOLOConfig:
    """Hyperparameter configuration for YOLO training."""

    def __init__(self, config_path: Optional[str] = None):
        # Default hyperparameters
        self.defaults = {
            # Model architecture
            'num_classes': 8,  # KITTI classes
            'input_channels': 3,  # RGB images
            'backbone': 'darknet53',

            # Training parameters
            'batch_size': 16,
            'num_epochs': 1,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,

            # Multi-modal fusion parameters
            'fusion_type': 'early',  # Options: 'early', 'late', 'middle'
            'calib_features': 64,
            'velodyne_features': 128,

            # Data augmentation
            'mosaic_prob': 0.5,
            'mixup_prob': 0.3,
            'flip_prob': 0.5,
            'hsv_prob': 0.5,

            # Loss weights
            'bbox_loss_weight': 1.0,
            'obj_loss_weight': 1.0,
            'cls_loss_weight': 1.0,
            'calib_loss_weight': 0.5,
            'velodyne_loss_weight': 0.5,

            # Learning rate scheduling
            'lr_scheduler': 'cosine',  # Options: 'step', 'cosine', 'linear'
            'warmup_epochs': 3,
            'lr_decay_rate': 0.1,
            'lr_decay_steps': [60, 80],

            # Early stopping
            'patience': 10,
            'min_delta': 0.001
        }

        # Load custom config if provided
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """Load custom configuration from YAML file."""
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
            self.defaults.update(custom_config)

    def save_config(self, save_path: str) -> None:
        """Save current configuration to YAML file."""
        with open(save_path, 'w') as f:
            yaml.dump(self.defaults, f)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration dictionary."""
        return self.defaults

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.defaults.update(updates)
