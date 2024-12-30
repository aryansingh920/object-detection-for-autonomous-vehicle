"""
Created on 30/12/2024

@author: Aryan

Filename: main.py

Relative Path: src/preprocessing/main.py
"""


import os
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import magic
from datetime import datetime
import re
from tqdm import tqdm
import yaml

regex = re.compile(r'^(train|val|test)_(image|calib|velodyne)$')


class KITTIToCOCOConverter:
    def __init__(self, kitti_root, coco_output):
        self.kitti_root = Path(kitti_root)
        self.coco_output = Path(coco_output)
        self.images = []
        self.test_images = []
        self.annotations = []
        self.categories = []
        self.calib_data = []
        self.test_calib_data = []
        self.velodyne_data = []
        self.test_velodyne_data = []
        self.annotation_id = 0

    def parse_kitti_data(self):
        """Parse KITTI dataset into calibration, image, and Velodyne data."""
        calib_path = self.kitti_root / "data_object_calib/training/calib"
        image_path = self.kitti_root / "data_object_image_2/training/image_2"
        velodyne_path = self.kitti_root / "data_object_velodyne/training/velodyne"

        test_calib_path = self.kitti_root / "data_object_calib/testing/calib"
        test_image_path = self.kitti_root / "data_object_image_2/testing/image_2"
        test_velodyne_path = self.kitti_root / "data_object_velodyne/testing/velodyne"

        self.calib_files = list(calib_path.glob("*.txt"))
        self.image_files = list(image_path.glob("*.png"))
        self.velodyne_files = list(velodyne_path.glob("*.bin"))

        self.test_calib_files = list(test_calib_path.glob("*.txt"))
        self.test_image_files = list(test_image_path.glob("*.png"))
        self.test_velodyne_files = list(test_velodyne_path.glob("*.bin"))

        print(f"Parsed {len(self.calib_files)} calibration files, {len(self.image_files)} images, and {len(self.velodyne_files)} Velodyne files.")
        print(f"Parsed {len(self.test_calib_files)} calibration files, {len(self.test_image_files)} images, and {len(self.test_velodyne_files)} Velodyne files.")

    def get_coco_data(self, split):
        """Retrieve COCO data for the specified split."""
        if split == 'train':
            return {"images": self.train_images, "annotations": []}
        elif split == 'val':
            return {"images": self.val_images, "annotations": []}
        elif split == 'test':
            return {"images": self.test_images, "annotations": []}
        else:
            raise ValueError(
                f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")


    def organize_data(self):
        """Organize data into COCO format: categories, images, annotations."""
        # Define categories (update based on your dataset)
        self.categories = [
            {"id": 1, "name": "Car"},
            {"id": 2, "name": "Pedestrian"},
            {"id": 3, "name": "Cyclist"},
            {"id": 4, "name": "Truck"},
            {"id": 5, "name": "Van"},
            {"id": 6, "name": "Person_sitting"},
            {"id": 7, "name": "Tram"},
            {"id": 8, "name": "Misc"}
        ]

        # Process images
        for img_id, image_file in enumerate(self.image_files):
            image = Image.open(image_file)
            width, height = image.size

            # Get detailed metadata
            metadata = self.get_detailed_file_metadata(
                self.kitti_root / image_file.relative_to(self.kitti_root))

            # Add image info
            self.images.append({
                "id": img_id,
                "file_name": f"data/kitti/{image_file.relative_to(self.kitti_root)}",
                "width": width,
                "height": height,
                "date_captured": metadata["created"].isoformat(),
                "license": 1
            })

        # Process calibration files
        for calib_id, calib_file in enumerate(self.calib_files):
            self.calib_data.append({
                "id": calib_id,
                "file_name": f"data/kitti/{calib_file.relative_to(self.kitti_root)}",
                "type": "calibration"
            })

        # Process Velodyne files
        for velodyne_id, velodyne_file in enumerate(self.velodyne_files):
            self.velodyne_data.append({
                "id": velodyne_id,
                "file_name": f"data/kitti/{velodyne_file.relative_to(self.kitti_root)}",
                "type": "point_cloud"
            })

        # Process test images
        for img_id, image_file in enumerate(self.test_image_files):
            image = Image.open(image_file)
            width, height = image.size

            # Get detailed metadata
            metadata = self.get_detailed_file_metadata(
                self.kitti_root / image_file.relative_to(self.kitti_root))

            self.test_images.append({
                "id": img_id,
                "file_name": f"data/kitti/{image_file.relative_to(self.kitti_root)}",
                "width": width,
                "height": height,
                "date_captured": metadata["created"].isoformat(),
                "license": 1
            })

        # Process test calibration files
        for calib_id, calib_file in enumerate(self.test_calib_files):
            self.test_calib_data.append({
                "id": calib_id,
                "file_name": f"data/kitti/{calib_file.relative_to(self.kitti_root)}",
                "type": "calibration"
            })

        # Process test Velodyne files
        for velodyne_id, velodyne_file in enumerate(self.test_velodyne_files):
            self.test_velodyne_data.append({
                "id": velodyne_id,
                "file_name": f"data/kitti/{velodyne_file.relative_to(self.kitti_root)}",
                "type": "point_cloud"
            })

    def get_detailed_file_metadata(self, file_path):
        """Get detailed metadata of a file."""
        # Get basic file statistics
        stat_info = os.stat(file_path)
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_size": stat_info.st_size,  # Size in bytes
            "created": datetime.fromtimestamp(stat_info.st_ctime),
            "modified": datetime.fromtimestamp(stat_info.st_mtime),
            "accessed": datetime.fromtimestamp(stat_info.st_atime),
        }

        # Use magic to get detailed MIME type
        mime = magic.Magic(mime=True)
        metadata["mime_type"] = mime.from_file(file_path)

        return metadata

    def split_data(self, train_ratio=0.8):
        """Split data into training and validation sets."""
        total_images = len(self.images)
        train_count = int(total_images * train_ratio)

        random.shuffle(self.images)
        self.train_images = self.images[:train_count]
        self.val_images = self.images[train_count:]

        calib_train_count = int(len(self.calib_data) * train_ratio)
        self.train_calib = self.calib_data[:calib_train_count]
        self.val_calib = self.calib_data[calib_train_count:]

        velodyne_train_count = int(len(self.velodyne_data) * train_ratio)
        self.train_velodyne = self.velodyne_data[:velodyne_train_count]
        self.val_velodyne = self.velodyne_data[velodyne_train_count:]

        print(
            f"Split data into {len(self.train_images)} training and {len(self.val_images)} validation images.")
        print(
            f"Split calibration into {len(self.train_calib)} training and {len(self.val_calib)} validation files.")
        print(
            f"Split Velodyne into {len(self.train_velodyne)} training and {len(self.val_velodyne)} validation files.")

    def create_coco_json(self):
        """Create COCO JSON files for train and val sets."""
        base_info = {
            "info": {
                "description": "KITTI Dataset converted to COCO format",
                "url": "http://www.cvlibs.net/datasets/kitti/",
                "version": "1.0",
                "year": f"{datetime.now().year}",
                "contributor": "Aryan",
                "date_created": f"{datetime.now().isoformat()}"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }
            ]
        }

        return {
            "train_image": {
                **base_info,
                "images": self.train_images,
                "annotations": [],
                "categories": self.categories,
            },
            "val_image": {
                **base_info,
                "images": self.val_images,
                "annotations": [],
                "categories": self.categories,
            },
            "train_calib": {
                **base_info,
                "calibration": self.train_calib,
                "annotations": [],
                "categories": [],
                "type": "calibration"
            },
            "val_calib": {
                **base_info,
                "calibration": self.val_calib,
                "annotations": [],
                "categories": [],
                "type": "calibration"
            },
            "train_velodyne": {
                **base_info,
                "velodyne": self.train_velodyne,
                "annotations": [],
                "categories": [],
                "type": "point_cloud"
            },
            "val_velodyne": {
                **base_info,
                "velodyne": self.val_velodyne,
                "annotations": [],
                "categories": [],
                "type": "point_cloud"
            },
            "test_image": {
                **base_info,
                "images": self.test_images,
                "annotations": [],
                "categories": self.categories,
            },
            "test_calib": {
                **base_info,
                "calibration": self.test_calib_data,
                "annotations": [],
                "categories": [],
                "type": "calibration"
            },
            "test_velodyne": {
                **base_info,
                "velodyne": self.test_velodyne_data,
                "annotations": [],
                "categories": [],
                "type": "point_cloud"
            },
        }

    def normalize_and_standardize_dataset(self, target_size=(640, 640)):
        """
        Normalizes and standardizes the dataset by resizing images and normalizing bounding box coordinates.

        Parameters:
        - target_size: tuple, target size (width, height) for resizing images.
        """
        try:
            # Define directories for normalized data
            normalized_images_dir = self.coco_output / "normalized_images"
            normalized_images_dir.mkdir(parents=True, exist_ok=True)

            # Iterate over train and validation sets
            for split in ['train', 'val']:
                images = getattr(self, f"{split}_images")
                annotations = self.get_coco_data(split)['annotations']
                skipped_images = 0

                # Create a mapping from image_id to image data for quick access
                image_id_map = {image['id']: image for image in images}

                for image in tqdm(images, desc=f"Normalizing {split} images"):
                    try:
                        # Convert to Path object
                        image_path = Path(image['file_name'])
                        if not image_path.exists():
                            skipped_images += 1
                            print(f"Image not found: {image_path}. Skipping.")
                            continue

                        # Open and resize the image
                        with Image.open(image_path) as img:
                            original_width, original_height = img.size
                            img_resized = img.resize(
                                target_size, Image.Resampling.LANCZOS)
                            resized_file_name = f"resized_{image_path.name}"
                            resized_image_path = normalized_images_dir / split
                            resized_image_path.mkdir(parents=True, exist_ok=True)
                            img_resized.save(
                                resized_image_path / resized_file_name)

                        # Update image metadata
                        image['file_name'] = str(
                            resized_image_path / resized_file_name)
                        image['width'], image['height'] = target_size

                        # Update annotations for the resized image
                        for ann in self.annotations:
                            if ann['image_id'] == image['id']:
                                bbox = ann['bbox']
                                normalized_bbox = [
                                    bbox[0] / original_width * target_size[0],
                                    bbox[1] / original_height * target_size[1],
                                    bbox[2] / original_width * target_size[0],
                                    bbox[3] / original_height * target_size[1]
                                ]
                                ann['bbox'] = normalized_bbox
                                ann['area'] = normalized_bbox[2] * \
                                    normalized_bbox[3]

                    except Exception as e:
                        print(f"Error processing image {image['file_name']}: {e}")
                        skipped_images += 1

                print(
                    f"Normalized {split} images. Skipped images: {skipped_images}")

        except Exception as e:
            print(f"Error during normalization: {e}")

    def save_data(self):
        """Save COCO data to output folder."""
        self.coco_output.mkdir(parents=True, exist_ok=True)

        # Normalize the dataset before saving
        self.normalize_and_standardize_dataset(target_size=(640, 640))

        coco_jsons = self.create_coco_json()

        for key, coco_data in coco_jsons.items():
            match = regex.match(key)
            if not match:
                print(f"Skipping key '{key}' as it does not match the regex.")
                continue
            prefix, suffix = match.groups()
            # mkdir data/coco/prefix
            (self.coco_output / prefix).mkdir(parents=True, exist_ok=True)
            with open(self.coco_output / prefix / f"{key}.json", "w") as json_file:
                json.dump(coco_data, json_file, indent=4)

            # convert the json to yaml
            with open(self.coco_output / prefix / f"{key}.yaml", "w") as yaml_file:
                yaml.dump(coco_data, yaml_file, default_flow_style=False)

        print(f"COCO data saved to {self.coco_output}.")


# # Example usage
# if __name__ == "__main__":
#     converter = KITTIToCOCOConverter(
#         kitti_root="data/kitti", coco_output="data/coco")
#     converter.parse_kitti_data()
#     converter.organize_data()
#     converter.split_data(train_ratio=0.8)
#     converter.normalize_and_standardize_dataset(target_size=(1242, 375))
#     converter.save_data()
