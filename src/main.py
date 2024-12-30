"""
Created on 30/12/2024

@author: Aryan

Filename: main.py

Relative Path: src/main.py
"""

from pathlib import Path
from preprocessing.main import KITTIToCOCOConverter
from src.train.YOLO_trainer import KITTIMultiModalDataset, YOLOTrainer
from config.config import Config, YOLOConfig as ConfigYOLOConfig
# Rename to avoid conflict
from src.train.YOLO_trainer import YOLOConfig as TrainYOLOConfig


def main():
    # Step 1: Preprocess datasets (convert only KITTI)
    if Config.preprocess:
        preprocess_datasets()

    # Step 2: Train model
    if Config.train:
        train_model()


def preprocess_datasets():
    print("\n=== Step 1: Preprocessing Datasets ===")
    converter = KITTIToCOCOConverter(
        kitti_root=Config.kitti_base_path, coco_output=Config.coco_base_path)
    converter.parse_kitti_data()
    converter.organize_data()
    converter.split_data(train_ratio=Config.train_val_ratio)
    if Config.normalize:
        converter.normalize_and_standardize_dataset(
            target_size=Config.target_size_for_normalization)
    converter.save_data()


def train_model():
    print("\n=== Step 2: Training Model ===")
    yolo_config = ConfigYOLOConfig()  # Instance of config.config.YOLOConfig
    trainer = YOLOTrainer(
        config=yolo_config,  # Pass the YOLOConfig instance
        data_dir=Path(Config.coco_base_path),
        output_dir=Path(Config.output_path)
    )
    trainer.train()


if __name__ == "__main__":
    main()
