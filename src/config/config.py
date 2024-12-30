"""
Created on 29/12/2024

@author: Aryan

Filename: config.py

Relative Path: src/config/config.py
"""

class Config:
    """
    Centralized configuration for the object detection pipeline.
    """
    # Dataset settings
    dataset_type = "kitti"  # Options: "kitti" or "nuscenes"
    data_root = "data"  # Path to the dataset root directory
    # Name of the COCO-format annotation file
    output_json = "output/annotations.json"

    # YOLO dataset YAML path
    dataset_yaml = "input/coco/coco_datasets/kitti_coco_updated.yaml"

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
