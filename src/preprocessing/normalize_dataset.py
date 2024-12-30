"""
Created on 29/12/2024

@author: Aryan

Filename: normalize_dataset.py

Relative Path: src/preprocessing/normalize_dataset.py
"""

import os
import json
from PIL import Image
from tqdm import tqdm


def normalize_and_standardize_dataset(merged_json_path, images_dir, output_json_path, target_size=(640, 640)):
    """
    Normalizes and standardizes the dataset by resizing images and normalizing bounding box coordinates.

    Parameters:
    - merged_json_path: str, path to the merged COCO JSON file.
    - images_dir: str, path to the directory containing original images.
    - output_json_path: str, path to save the normalized JSON file.
    - target_size: tuple, target size (width, height) for resizing images.
    """
    try:
        # Create output directory for normalized images
        output_images_dir = os.path.join(images_dir, "normalized_images")
        os.makedirs(output_images_dir, exist_ok=True)

        with open(merged_json_path, 'r') as f:
            data = json.load(f)

        processed_images = []
        processed_annotations = []
        skipped_images = 0

        # Process each image in the dataset
        for image in tqdm(data['images'], desc="Processing images"):
            try:
                image_path = os.path.join(images_dir, image['file_name'])
                if not os.path.exists(image_path):
                    skipped_images += 1
                    print(f"Image not found: {image_path}. Skipping.")
                    continue

                # Open and resize the image
                with Image.open(image_path) as img:
                    original_width, original_height = img.size
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    resized_file_name = f"resized_{image['file_name']}"
                    img_resized.save(os.path.join(output_images_dir, resized_file_name))

                # Update image metadata
                image['file_name'] = resized_file_name
                image['width'], image['height'] = target_size
                processed_images.append(image)

                # Update annotations for the resized image
                for ann in data['annotations']:
                    if ann['image_id'] == image['id']:
                        bbox = ann['bbox']
                        normalized_bbox = [
                            bbox[0] / original_width * target_size[0],
                            bbox[1] / original_height * target_size[1],
                            bbox[2] / original_width * target_size[0],
                            bbox[3] / original_height * target_size[1]
                        ]
                        ann['bbox'] = normalized_bbox
                        processed_annotations.append(ann)

            except Exception as e:
                print(f"Error processing image {image['file_name']}: {e}")
                skipped_images += 1

        # Update dataset with processed images and annotations
        data['images'] = processed_images
        data['annotations'] = processed_annotations

        # Save the normalized dataset to JSON
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Normalized dataset saved to {output_json_path}")
        print(f"Skipped images: {skipped_images}")

    except Exception as e:
        print(f"Error during normalization: {e}")


if __name__ == "__main__":
    merged_json_path = r"<path_to_merged_coco_json>"
    images_dir = r"<path_to_images_directory>"
    output_json_path = r"<path_to_normalized_json>"
    normalize_and_standardize_dataset(merged_json_path, images_dir, output_json_path)
