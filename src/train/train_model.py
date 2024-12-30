"""
Created on 29/12/2024

@author: Aryan

Filename: train_model.py

Relative Path: src/train/train_model.py
"""

import os
from ultralytics import YOLO
from tqdm import tqdm
import wandb
from config.config import Config
import torch


def init_wandb(config):
    """
    Initialize Weights & Biases tracking.
    """
    wandb.init(
        project=f"yolov8-{config.dataset_type.lower()}",
        name=f"{config.dataset_type}_experiment",
        config={
            "architecture": config.model_arch,
            "dataset": config.dataset_type,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.img_size,
            "learning_rate": config.learning_rate,
            "data_root": config.data_root
        }
    )


def train_model(config):
    """
    Train YOLOv8 on the dataset using the provided configuration.
    Includes W&B tracking and progress visualization.
    """
    try:
        # Step 1: Initialize W&B
        init_wandb(config)

        # Step 2: Load the YOLO model
        print(f"Loading YOLOv8 model: {config.model_arch}")
        model = YOLO(config.model_arch)

        # Step 3: Train the model
        print("Starting training...")


        results = model.train(
            data=config.dataset_yaml,  # Point to the updated COCO dataset YAML file
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.img_size,
            lr0=config.learning_rate,
            name=f"{config.dataset_type}_experiment",
            project=wandb.run.project,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=8,  # Number of worker threads
            exist_ok=True,  # Overwrite existing experiment
            verbose=True,  # Print verbose output
        )

        # Log final metrics to W&B
        wandb.log({
            "final_mAP": results.maps,
            "best_epoch": results.best_epoch,
            "final_precision": results.fitness,
        })

        print("Training finished!")
        print("Results:", results)

    except Exception as e:
        print(f"Error during training: {e}")
        # Log error to W&B if initialized
        if wandb.run:
            wandb.log({"error": str(e)})
            wandb.finish()
        raise e
    finally:
        wandb.finish()

def main():
    try:
        # Update the dataset YAML path in the configuration
        #   Config.dataset_yaml = "input/coco/coco_datasets/kitti_coco_updated.json"  # Point to the updated dataset

        # Start the training process
        train_model(Config)
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
