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
    validate = False             # Whether to run validation
    train = True                # Whether to run training

    kitti_base_path = "data/kitti"
    coco_base_path = "data/coco"
    yolo_base_path = "data/yolo"
    validated_image_path = "data/verified"
    output = "output"
    data_root = Path(coco_base_path)

    # For splitting
    train_val_ratio = 0.8
    target_size_for_normalization = (1242, 375)
    target = target_size_for_normalization


class Hyperparameters:
    def __init__(
        self,
        model_name="yolov8n",  # Default model name
        model_path="models/yolov8n.pt",  # Default model path
        data="data/yolo/yolo.yaml",        # Dataset YAML file path
        epochs=1,                # Number of epochs
        imgsz=max(Config.target),                # Input image size
        batch=16,                 # Batch size
        lr0=0.01,                 # Initial learning rate
        momentum=0.937,           # Momentum for SGD optimizer
        weight_decay=0.0005,      # Weight decay
        warmup_epochs=3.0,        # Warmup epochs
        warmup_momentum=0.8,      # Warmup momentum
        warmup_bias_lr=0.1,       # Warmup bias learning rate
        optimizer="SGD",          # Optimizer: 'SGD' or 'Adam'
        augment=True,             # Data augmentation
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.lr0 = lr0
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.optimizer = optimizer
        self.augment = augment


def hyperparameters_yolov8m():

    return Hyperparameters(
        model_name="yolov8m",  # Use YOLOv8 medium model
        model_path="models/yolov8m.pt",  # Use YOLOv8 medium model
        data="data/yolo/yolo.yaml",     # Dataset YAML file path
        epochs=2,                     # Increased for better convergence
        imgsz=1280,                     # Higher resolution for detailed detection
        batch=4,                       # Balanced batch size for memory and performance
        lr0=0.0025,                     # Lower learning rate for better stability
        momentum=0.937,                 # Default momentum for SGD
        weight_decay=0.0005,            # Prevent overfitting
        warmup_epochs=-.0,              # Default warmup period
        warmup_momentum=0.9,            # Slightly increased momentum during warmup
        warmup_bias_lr=0.2,             # Increased learning rate for bias during warmup
        optimizer="Adam",             # Adam optimizer for faster convergence
        augment=True,                   # Enable data augmentation
    )


def hyperparameters_yolov5m():
    return Hyperparameters(
        model_name="yolov5m",  # Use YOLOv5 medium model
        model_path="models/yolov5m.pt",  # Use YOLOv5 medium model
        data="data/yolo/yolo.yaml",     # Dataset YAML file path
        epochs=120,                     # Increased epochs for improved mAP
        imgsz=1280,                      # Balanced image size for performance
        batch=16,                       # Batch size suitable for memory constraints
        lr0=0.003,                      # Learning rate optimized for YOLOv5m
        momentum=0.94,                  # Slightly higher momentum for smoother updates
        weight_decay=0.0001,            # Reduced weight decay to allow better fitting
        warmup_epochs=2.5,              # Shorter warmup period for YOLOv5
        warmup_momentum=0.85,           # Standard warmup momentum
        warmup_bias_lr=0.15,            # Lower bias learning rate for stability
        optimizer="Adam",               # Adam for faster convergence
        augment=True,                   # Enable data augmentation
    )
