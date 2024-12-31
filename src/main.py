"""
Created on 30/12/2024

@author: Aryan

Filename: main.py
Relative Path: src/main.py
"""

import torch
from pathlib import Path
from preprocessing.main import KITTIToCOCOConverter
from train.YOLO_trainer import TorchVisionTrainer
from config.config import Config





def main():
    # Step 1: Preprocess datasets (convert only KITTI) if needed
    if Config.preprocess:
        preprocess_datasets()

    # Step 2: Train model if needed
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

    if Config.normalize:
        converter.normalize_and_standardize_dataset(
            target_size=Config.target_size_for_normalization
        )
    converter.save_data()


def train_model():
    print("\n=== Step 2: Training Model ===")
    # Create data_root=Path("data/coco") or use Config.coco_base_path
    config = Config()
    # config.data_root = Path(config.coco_base_path)

    # Build trainer from config
    trainer = TorchVisionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
