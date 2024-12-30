"""
Created on 30/12/2024

@author: Aryan

Filename: main.py

Relative Path: src/main.py
"""

from pathlib import Path
from preprocessing.main import KITTIToCOCOConverter
from train.YOLO_trainer import YOLOTrainer
from config.config import Config, YOLOConfig as ConfigYOLOConfig



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
    # yolo_config = ConfigYOLOConfig()  # Instance of config.config.YOLOConfig
    # trainer = YOLOTrainer(
    #     config=yolo_config,  # Pass the YOLOConfig instance
    #     data_dir=Path(Config.coco_base_path),
    #     output_dir=Path(Config.output_path)
    # )
    # trainer.train()
    config = {
        "learning_rate": 0.001,
        "batch_size": 128,
        "num_epochs": 1,
        "image_height": 1242,
        "image_width": 375,
        "wandb_project": "YOLO-Training"
    }
    trainer = YOLOTrainer(
        config=config,
        data_dir=Path(Config.coco_base_path),
        output_dir=Path(Config.output_path),
        # batch_size=4,       # Set batch size
        # num_epochs=1      # Set number of epochs
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
