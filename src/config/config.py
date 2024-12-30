"""
Created on 29/12/2024

@author: Aryan

Filename: config.py

Relative Path: src/config/config.py
"""

class Config:
    # Define paths (Update these paths as per your directory structure)
    kitti_base_path = r"data/kitti"
    coco_base_path = r"data/coco"

    # Preprocessing settings
    train_val_ratio = 0.8
    normalize = True
    target_size_for_normalization = (1242, 375)

    # Model paths
    saved_model_path = r"saved_model"
    output_path = r"output"


class Hyperparameters:
    # Training hyperparameters
    batch_size = 16
    block_size = 32
    n_embd = 32
    n_head = 2
    n_layer = 2
    dropout = 0.0
    learning_rate = 1e-3
    max_iters = 10000
    eval_interval = 100
    eval_iters = 50

    # Model architecture
    model_arch = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, etc.

    # Training settings
    epochs = 50
    img_size = 640
