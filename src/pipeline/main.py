"""
Created on 29/12/2024

@author: Aryan

Filename: main.py

Relative Path: src/pipeline/main.py
"""

import os
from preprocessing.kitti_to_coco import kitti_to_coco
from preprocessing.merge_coco import merge_coco_datasets
from preprocessing.copy_images import copy_dataset_images
from preprocessing.normalize_dataset import normalize_and_standardize_dataset
from preprocessing.validate_dataset import validate_coco_dataset
from preprocessing.update_json_paths import update_image_paths as update_json_paths
from train.train_model import main as trainer
import torch

def preprocess_datasets(kitti_base_path, output_dir):
    """Step 1: Convert KITTI to COCO format."""
    print("Starting preprocessing...")

    # Paths for outputs
    kitti_coco_path = os.path.join(output_dir, "coco_datasets", "kitti_coco.json")
    merged_coco_path = os.path.join(output_dir, "coco_datasets", "merged_dataset.json")

    # Convert KITTI to COCO
    print("Converting KITTI dataset to COCO format...")
    kitti_to_coco(kitti_base_path, kitti_coco_path)

    # For single-dataset pipeline, set merged path to KITTI COCO path
    print("Processing KITTI dataset only. Skipping merging...")
    merged_coco_path = kitti_coco_path

    print("Preprocessing completed.")
    return merged_coco_path


def process_data(merged_coco_path, images_dir, output_dir):
    json_file_path = r"input/coco/coco_datasets/kitti_coco.json"
    images_dir_update_json_paths_training = r"data/kitti/data_object_image_2/training/image_2/normalized_images"

    """Step 2: Copy images and normalize dataset."""
    print("Starting data processing...")

    # Paths for outputs
    output_images_dir = os.path.join(output_dir, "images", "original_images")
    normalized_coco_path = os.path.join(output_dir, "coco_datasets", "normalized_dataset.json")

    # Copy images
    print("Copying KITTI images to a unified directory...")
    copy_dataset_images(os.path.join(images_dir, "kitti"), output_images_dir)

    # Normalize dataset
    print("Normalizing the COCO dataset...")
    # normalize_and_standardize_dataset(
    #     merged_coco_path, images_dir_update_json_paths_training, normalized_coco_path)

    # Update JSON paths
    print("Updating JSON paths...")

    update_json_paths(json_file_path, images_dir_update_json_paths_training,
                      images_dir_update_json_paths_training)

    print("Data processing completed.")
    return normalized_coco_path


def validate_data(normalized_coco_path):
    """Step 3: Validate the normalized dataset."""
    print("Validating the normalized COCO dataset...")
    is_valid = validate_coco_dataset(normalized_coco_path)
    if is_valid:
        print("Validation passed!")
    else:
        print("Validation failed. Please check the dataset.")


# Future functions for training, evaluation, etc.
def train_model():
    """Step 4: Placeholder for model training."""
    # print("Training pipeline not implemented yet.")
    trainer()

def evaluate_model():
    """Step 5: Placeholder for model evaluation."""
    print("Evaluation pipeline not implemented yet.")
