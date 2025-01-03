"""
Created on 03/01/2025

@author: Aryan

Filename: pipeline.py

Relative Path: src/pipeline.py
"""

import os
from pathlib import Path

import torch

from preprocessing.main import KITTIToCOCOConverter
from preprocessing.validate_dataset import DataValidator
from preprocessing.coco_to_yolo import coco_to_yolo
from train.train_yolo import main
from config.config import Config, Hyperparameters, hyperparameters_yolov5m, hyperparameters_yolov8m

import argparse


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

    data_dir = Config.coco_base_path
    output_dir = Config.yolo_base_path
    coco_to_yolo(data_dir, output_dir)


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
    main(hyperparams=hyperparameters_yolov8m(),
         use_wandb=True,
         device="auto",
         rect=True,
        )
