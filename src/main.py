"""
Created on 30/12/2024

@author: Aryan

Filename: main.py
Relative Path: src/main.py
"""
import os
from pathlib import Path

import torch

from preprocessing.main import KITTIToCOCOConverter
from preprocessing.validate_dataset import DataValidator
# from train.YOLO_trainer import main
from config.config import Config

import argparse




def main():
    # Step 1: Preprocess datasets (convert only KITTI) if needed
    if Config.preprocess:
        preprocess_datasets()

    # Step 1.1: Validate dataset
    if Config.validate:
        validate_dataset()

    # Step 3: Train model if needed
    if Config.train:
        train()


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


def validate_dataset():
    dataset_path = Config.coco_base_path
    for dir in ["train", "val"]:
        validator = DataValidator(dataset_path=dataset_path, split=dir)
        validator.validate_dataset()
        print("Dataset path: ", dataset_path)
        validator.save_random_images_with_bboxes_without_calibration(
            split=dir,
            output_dir=Config.validated_image_path,
            num_images=5
        )


def train():
    print("\n=== Step 2: Training Model ===")
    dataset_path = Config.coco_base_path
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train_dir", default="data/coco/train", help="Path to train split directory")
    # parser.add_argument("--val_dir", default="data/coco/val", help="Path to val split directory")
    # parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    # parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size")
    # parser.add_argument("--num_classes", type=int, default=8, help="Number of categories in dataset")
    # args = parser.parse_args()

    # main(
    #     train_dir=f"{dataset_path}/train",
    #     val_dir=f"{dataset_path}/val",
    #     epochs=1,
    #     batch_size=2,
    #     num_classes=9
    # )


if __name__ == "__main__":
    main()
