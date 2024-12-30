"""
Created on 29/12/2024

@author: Aryan

Filename: copy_images.py

Relative Path: src/preprocessing/copy_images.py
"""

import os
import shutil
from tqdm import tqdm


def copy_dataset_images(kitti_images_dir, output_images_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    print("Starting to copy KITTI images...")
    total_copied = 0

    # Copy KITTI images
    kitti_training_dir = os.path.join(kitti_images_dir, "training", "image_2")
    if os.path.exists(kitti_training_dir):
        files = [f for f in os.listdir(kitti_training_dir) if f.endswith(('.png', '.jpg'))]
        for file in tqdm(files, desc="Copying KITTI images"):
            dest_file = f"kitti_{file}"
            shutil.copy(os.path.join(kitti_training_dir, file), os.path.join(output_images_dir, dest_file))
            total_copied += 1

    print(f"\nTotal KITTI images copied: {total_copied}")
    return total_copied


if __name__ == "__main__":
    kitti_images_dir = r"<path_to_kitti_images>"
    output_images_dir = r"<path_to_output_images>"
    copy_dataset_images(kitti_images_dir, output_images_dir)
