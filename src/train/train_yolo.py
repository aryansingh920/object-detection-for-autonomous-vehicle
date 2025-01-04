"""
Created on 03/01/2025

@author: Aryan

Filename: train_yolo.py

Relative Path: src/train/train_yolo.py
"""
import torch
from ultralytics import YOLO
from config.config import Hyperparameters
import wandb


def main(hyperparams: Hyperparameters, use_wandb: bool = False, device: str = "cpu", rect: bool = True):
    # 1. Automatically select device

    if device == "auto":
        if torch.backends.mps.is_available():
            # for Apple Silicon (M1/M2) with Metal Performance Shaders
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"  # for NVIDIA GPUs
        else:
            device = "cpu"
    else:
        device = device

    print(f"Using device: {device}")
    # print("Hyperparameters: ", hyperparams.__dict__)

    # 2. Load a pre-trained YOLOv8 model
    model = YOLO(hyperparams.model_path)

    if use_wandb:
        if not wandb.run:
            wandb.init(
                project="yolo_training",
                name=hyperparams.model_name.lower(),
                config=vars(hyperparams)
            )

    # 3. Train with hyperparameters
    model.train(
        data=hyperparams.data,               # Path to the YAML file
        epochs=hyperparams.epochs,           # Number of training epochs
        imgsz=hyperparams.imgsz,             # Input image size
        batch=hyperparams.batch,             # Batch size
        lr0=hyperparams.lr0,                 # Initial learning rate
        momentum=hyperparams.momentum,       # Momentum
        weight_decay=hyperparams.weight_decay,  # Weight decay
        warmup_epochs=hyperparams.warmup_epochs,  # Warmup epochs
        warmup_momentum=hyperparams.warmup_momentum,  # Warmup momentum
        warmup_bias_lr=hyperparams.warmup_bias_lr,    # Warmup bias learning rate
        optimizer=hyperparams.optimizer,     # Optimizer: 'SGD' or 'Adam'
        augment=hyperparams.augment,         # Data augmentation
        device=device,                       # Selected device
        rect=rect,                           # Rectangular training
        project="models/yolo_training"
    )

    # 5. Save the trained model
    # Final model
    model.save(f"models/custom/{hyperparams.model_name}_final.pt")
    print("Model saved as final model")

    if use_wandb:
        wandb.finish()
