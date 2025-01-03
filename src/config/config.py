"""
Created on 29/12/2024

@author: Aryan

Filename: config.py

Relative Path: src/config/config.py
"""

from pathlib import Path


class Config:
    # General
    preprocess = False          # Whether to run the preprocessing
    normalize = False           # Whether to normalize the dataset
    validate = True             # Whether to run validation
    train = False                # Whether to run training

    kitti_base_path = "data/kitti"
    coco_base_path = "data/coco"
    validated_image_path = "verified"
    data_root = Path(coco_base_path)

    # For splitting
    train_val_ratio = 0.8
    target_size_for_normalization = (1242, 375)
    target = target_size_for_normalization

    # Training hyperparameters
    batch_size = 1
    num_classes = 9  # 8 KITTI classes + 1 background
    lr = 1e-3
    num_epochs = 1

    # W&B logging
    use_wandb = True
    wandb_project_name = "kitti_frcnn"

    model_name = "fasterrcnn_resnet50_fpn"

    # Device can be 'auto', 'cuda', 'cpu', or 'mps'
    device = "cpu"  # will auto-detect GPU/MPS if possible
