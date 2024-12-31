"""
Created on 30/12/2024

@author: Aryan

Filename: main.py

Relative Path: src/main.py
"""

import torch  # <--- ADD THIS IMPORT
from pathlib import Path
from preprocessing.main import KITTIToCOCOConverter
from train.YOLO_trainer import TorchVisionTrainer
from config.config import Config, YOLOConfig as ConfigYOLOConfig


def main():
    # Step 1: Preprocess datasets (convert only KITTI)
    # if Config.preprocess:
    #     preprocess_datasets()

    # Step 2: Train model
    if Config.train:
        train_model()


def preprocess_datasets():
    print("\n=== Step 1: Preprocessing Datasets ===")
    converter = KITTIToCOCOConverter(
        kitti_root=Config.kitti_base_path,
        coco_output=Config.coco_base_path
    )
    converter.parse_kitti_data()
    converter.organize_data()
    converter.split_data(train_ratio=Config.train_val_ratio)
    # if Config.normalize:
    #     converter.normalize_and_standardize_dataset(
    #         target_size=Config.target_size_for_normalization
    #     )
    converter.save_data()


def train_model():
    print("\n=== Step 2: Training Model ===")
    trainer = TorchVisionTrainer(
        data_root=Path("data/coco"),    # location of train/ val/ folders
        batch_size=4,
        num_classes=9,  # 8 KITTI classes + 1 for background
        lr=1e-3,
        num_epochs=2,
        device="mps" if torch.backends.mps.is_available() else "cuda"
    )
    trainer.train()


if __name__ == "__main__":
    main()
