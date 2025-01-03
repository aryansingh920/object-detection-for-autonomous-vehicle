"""
Created on 31/12/2024

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

# Updated regex to include annotations as well
regex = re.compile(r'^(train|val|test)_(image|calib|velodyne|annotations)$')


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

        # New attributes for annotations
        self.train_annotations = []
        self.val_annotations = []

        # Define category mapping from KITTI to COCO
        self.category_mapping = {
            "Car": 1,
            "Pedestrian": 2,
            "Cyclist": 3,
            "Truck": 4,
            "Van": 5,
            "Person_sitting": 6,
            "Tram": 7,
            "Misc": 8
        }

    def parse_kitti_data(self):
        """Parse KITTI dataset into calibration, image, Velodyne, and label data."""
        calib_path = self.kitti_root / "data_object_calib/training/calib"
        image_path = self.kitti_root / "data_object_image_2/training/image_2"
        velodyne_path = self.kitti_root / "data_object_velodyne/training/velodyne"

        # **Update this path to the actual location of your label files**
        # Assuming labels are in data/kitti/training/
        label_path = self.kitti_root / "training" / "label_2"

        test_calib_path = self.kitti_root / "data_object_calib/testing/calib"
        test_image_path = self.kitti_root / "data_object_image_2/testing/image_2"
        test_velodyne_path = self.kitti_root / "data_object_velodyne/testing/velodyne"

        self.calib_files = list(calib_path.glob("*.txt"))
        self.image_files = list(image_path.glob("*.png"))
        self.velodyne_files = list(velodyne_path.glob("*.bin"))

        # **Ensure label_path exists and contains .txt files**
        if not label_path.exists():
            print(
                f"Label path {label_path} does not exist. Please check the path.")
            self.label_files = []
        else:
            self.label_files = list(label_path.glob("*.txt"))
            print(
                f"Found {len(self.label_files)} label files in {label_path}.")

        self.test_calib_files = list(test_calib_path.glob("*.txt"))
        self.test_image_files = list(test_image_path.glob("*.png"))
        self.test_velodyne_files = list(test_velodyne_path.glob("*.bin"))

        print(f"Parsed {len(self.calib_files)} calibration files, {len(self.image_files)} images, {len(self.velodyne_files)} Velodyne files, and {len(self.label_files)} label files.")
        print(f"Parsed {len(self.test_calib_files)} calibration files, {len(self.test_image_files)} images, and {len(self.test_velodyne_files)} Velodyne files.")


    def parse_labels(self):
        """Parse label files and create COCO annotations."""
        for label_file in tqdm(self.label_files, desc="Parsing label files"):
            file_stem = label_file.stem
            image_id = self.get_image_id(file_stem)
            if image_id is None:
                print(
                    f"No matching image found for label file {label_file.name}. Skipping.")
                continue

            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 15:
                    print(
                        f"Invalid label format in file {label_file.name}: {line}")
                    continue

                class_name = parts[0]
                if class_name == "DontCare":
                    continue

                bbox = [
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]) - float(parts[4]),
                    float(parts[7]) - float(parts[5])
                ]

                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                category_id = self.category_mapping.get(class_name)
                if category_id is None:
                    print(
                        f"Unknown class '{class_name}' in file {label_file.name}. Skipping.")
                    continue

                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                }
                self.annotations.append(annotation)
                self.annotation_id += 1

    # def parse_labels(self):
    #     """Parse label files and create COCO annotations."""
    #     # **Update this path if necessary**
    #     # Assuming labels are in data/kitti/training/
    #     label_path = self.kitti_root / "training"

    #     if not label_path.exists():
    #         print(
    #             f"Label path {label_path} does not exist. Skipping label parsing.")
    #         return

    #     for label_file in tqdm(self.label_files, desc="Parsing label files"):
    #         image_stem = label_file.stem  # e.g. '000000'
    #         image_id = self.get_image_id(image_stem)
    #         if image_id is None:
    #             print(
    #                 f"No matching image found for label file {label_file.name}. Skipping.")
    #             continue  # Skip if image not found

    #         with open(label_file, 'r') as f:
    #             lines = f.readlines()

    #         for line in lines:
    #             parts = line.strip().split()
    #             if len(parts) < 15:
    #                 print(
    #                     f"Invalid label format in file {label_file.name}: {line}")
    #                 continue  # Invalid label format

    #             class_name = parts[0]
    #             if class_name == "DontCare":
    #                 continue  # Skip DontCare objects

    #             # Extract bounding box coordinates
    #             try:
    #                 bbox = [
    #                     float(parts[4]),  # x1
    #                     float(parts[5]),  # y1
    #                     float(parts[6]),  # x2
    #                     float(parts[7])   # y2
    #                 ]
    #             except ValueError:
    #                 print(
    #                     f"Invalid bounding box values in file {label_file.name}: {line}")
    #                 continue  # Skip if coordinates are invalid

    #             # Convert bbox to [x, y, width, height]
    #             x, y, x2, y2 = bbox
    #             width = x2 - x
    #             height = y2 - y

    #             if width <= 0 or height <= 0:
    #                 # Negative or zero dimension -> skip
    #                 continue

    #             # Map category
    #             category_id = self.category_mapping.get(class_name)
    #             if category_id is None:
    #                 print(
    #                     f"Unknown class '{class_name}' in file {label_file.name}. Skipping.")
    #                 continue  # Skip unknown classes

    #             annotation = {
    #                 "id": self.annotation_id,
    #                 "image_id": image_id,
    #                 "category_id": category_id,
    #                 "bbox": [x, y, width, height],
    #                 "area": width * height,
    #                 "iscrowd": 0
    #             }
    #             self.annotations.append(annotation)
    #             self.annotation_id += 1

    def get_image_id(self, image_stem):
        """Retrieve image ID based on image filename stem."""
        for img in self.images:
            if Path(img['file_name']).stem == image_stem:
                return img['id']
        for img in self.test_images:
            if Path(img['file_name']).stem == image_stem:
                return img['id']
        return None

    def organize_data(self):
        """Organize data into COCO format: categories, images, annotations."""
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

        image_id_map = {}

        # Process images
        for img_id, image_file in enumerate(self.image_files, start=1):
            try:
                image = Image.open(image_file)
                width, height = image.size

                metadata = self.get_detailed_file_metadata(
                    self.kitti_root / image_file.relative_to(self.kitti_root))

                image_data = {
                    "id": img_id,
                    "file_name": f"data/kitti/{image_file.relative_to(self.kitti_root)}",
                    "width": width,
                    "height": height,
                    "date_captured": metadata["created"].isoformat(),
                    "license": 1
                }
                self.images.append(image_data)
                image_id_map[image_file.stem] = img_id
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")

        # Process calibration files
        for calib_file in self.calib_files:
            file_stem = calib_file.stem
            if file_stem in image_id_map:
                self.calib_data.append({
                    "id": image_id_map[file_stem],
                    "file_name": f"data/kitti/{calib_file.relative_to(self.kitti_root)}",
                    "type": "calibration"
                })

        # Process Velodyne files
        for velodyne_file in self.velodyne_files:
            file_stem = velodyne_file.stem
            if file_stem in image_id_map:
                self.velodyne_data.append({
                    "id": image_id_map[file_stem],
                    "file_name": f"data/kitti/{velodyne_file.relative_to(self.kitti_root)}",
                    "type": "point_cloud"
                })

        # Process test images
        for img_id, image_file in enumerate(self.test_image_files, start=1):
            try:
                image = Image.open(image_file)
                width, height = image.size

                metadata = self.get_detailed_file_metadata(
                    self.kitti_root / image_file.relative_to(self.kitti_root))

                image_data = {
                    "id": img_id,
                    "file_name": f"data/kitti/{image_file.relative_to(self.kitti_root)}",
                    "width": width,
                    "height": height,
                    "date_captured": metadata["created"].isoformat(),
                    "license": 1
                }
                self.test_images.append(image_data)
                image_id_map[image_file.stem] = img_id
            except Exception as e:
                print(f"Error processing test image {image_file}: {e}")

        # Process test calibration files
        for calib_file in self.test_calib_files:
            file_stem = calib_file.stem
            if file_stem in image_id_map:
                self.test_calib_data.append({
                    "id": image_id_map[file_stem],
                    "file_name": f"data/kitti/{calib_file.relative_to(self.kitti_root)}",
                    "type": "calibration"
                })

        # Process test Velodyne files
        for velodyne_file in self.test_velodyne_files:
            file_stem = velodyne_file.stem
            if file_stem in image_id_map:
                self.test_velodyne_data.append({
                    "id": image_id_map[file_stem],
                    "file_name": f"data/kitti/{velodyne_file.relative_to(self.kitti_root)}",
                    "type": "point_cloud"
                })

        # Parse labels and create annotations
        self.parse_labels()

        # Sort data by image_id
        self.images.sort(key=lambda x: x['id'])
        self.calib_data.sort(key=lambda x: x['id'])
        self.velodyne_data.sort(key=lambda x: x['id'])
        self.test_images.sort(key=lambda x: x['id'])
        self.test_calib_data.sort(key=lambda x: x['id'])
        self.test_velodyne_data.sort(key=lambda x: x['id'])
        self.annotations.sort(key=lambda x: x['image_id'])

    # def organize_data(self):
    #     """Organize data into COCO format: categories, images, annotations."""
    #     # Define categories (update based on your dataset)
    #     self.categories = [
    #         {"id": 1, "name": "Car"},
    #         {"id": 2, "name": "Pedestrian"},
    #         {"id": 3, "name": "Cyclist"},
    #         {"id": 4, "name": "Truck"},
    #         {"id": 5, "name": "Van"},
    #         {"id": 6, "name": "Person_sitting"},
    #         {"id": 7, "name": "Tram"},
    #         {"id": 8, "name": "Misc"}
    #     ]

    #     # Process images
    #     for img_id, image_file in enumerate(self.image_files, start=1):
    #         try:
    #             image = Image.open(image_file)
    #             width, height = image.size

    #             # Metadata
    #             metadata = self.get_detailed_file_metadata(
    #                 self.kitti_root / image_file.relative_to(self.kitti_root))

    #             self.images.append({
    #                 "id": img_id,
    #                 "file_name": f"data/kitti/{image_file.relative_to(self.kitti_root)}",
    #                 "width": width,
    #                 "height": height,
    #                 "date_captured": metadata["created"].isoformat(),
    #                 "license": 1
    #             })
    #         except Exception as e:
    #             print(f"Error processing image {image_file}: {e}")

    #     # Process calibration files
    #     for calib_id, calib_file in enumerate(self.calib_files, start=1):
    #         self.calib_data.append({
    #             "id": calib_id,
    #             "file_name": f"data/kitti/{calib_file.relative_to(self.kitti_root)}",
    #             "type": "calibration"
    #         })

    #     # Process Velodyne files
    #     for velodyne_id, velodyne_file in enumerate(self.velodyne_files, start=1):
    #         self.velodyne_data.append({
    #             "id": velodyne_id,
    #             "file_name": f"data/kitti/{velodyne_file.relative_to(self.kitti_root)}",
    #             "type": "point_cloud"
    #         })

    #     # Process test images
    #     for img_id, image_file in enumerate(self.test_image_files, start=1):
    #         try:
    #             image = Image.open(image_file)
    #             width, height = image.size

    #             # Metadata
    #             metadata = self.get_detailed_file_metadata(
    #                 self.kitti_root / image_file.relative_to(self.kitti_root))

    #             self.test_images.append({
    #                 "id": img_id,
    #                 "file_name": f"data/kitti/{image_file.relative_to(self.kitti_root)}",
    #                 "width": width,
    #                 "height": height,
    #                 "date_captured": metadata["created"].isoformat(),
    #                 "license": 1
    #             })
    #         except Exception as e:
    #             print(f"Error processing test image {image_file}: {e}")

    #     # Process test calibration files
    #     for calib_id, calib_file in enumerate(self.test_calib_files, start=1):
    #         self.test_calib_data.append({
    #             "id": calib_id,
    #             "file_name": f"data/kitti/{calib_file.relative_to(self.kitti_root)}",
    #             "type": "calibration"
    #         })

    #     # Process test Velodyne files
    #     for velodyne_id, velodyne_file in enumerate(self.test_velodyne_files, start=1):
    #         self.test_velodyne_data.append({
    #             "id": velodyne_id,
    #             "file_name": f"data/kitti/{velodyne_file.relative_to(self.kitti_root)}",
    #             "type": "point_cloud"
    #         })

    #     # Parse labels and create annotations
    #     self.parse_labels()


    def split_data(self, train_ratio=0.8):
        """Split data into training and validation sets with consistent `image_id`."""
        # Validate consistency across datasets
        image_ids = {img['id'] for img in self.images}
        calib_ids = {cal['id'] for cal in self.calib_data}
        velodyne_ids = {vel['id'] for vel in self.velodyne_data}
        annotation_ids = {ann['image_id'] for ann in self.annotations}

        consistent_ids = image_ids & calib_ids & velodyne_ids & annotation_ids

        if len(consistent_ids) != len(image_ids):
            missing_in_calib = image_ids - calib_ids
            missing_in_velodyne = image_ids - velodyne_ids
            missing_in_annotations = image_ids - annotation_ids

            if missing_in_calib:
                print(f"Missing in calib: {missing_in_calib}")
            if missing_in_velodyne:
                print(f"Missing in velodyne: {missing_in_velodyne}")
            if missing_in_annotations:
                print(f"Missing in annotations: {missing_in_annotations}")

            raise ValueError(
                "Inconsistent `image_id` across datasets. Ensure all data files are aligned.")

        # Filter datasets to include only consistent `image_id`
        self.images = [
            img for img in self.images if img['id'] in consistent_ids]
        self.calib_data = [
            cal for cal in self.calib_data if cal['id'] in consistent_ids]
        self.velodyne_data = [
            vel for vel in self.velodyne_data if vel['id'] in consistent_ids]
        self.annotations = [
            ann for ann in self.annotations if ann['image_id'] in consistent_ids]

        # Split data
        total_images = len(self.images)
        train_count = int(total_images * train_ratio)

        random.shuffle(self.images)  # Shuffle for random splitting
        train_ids = {img['id'] for img in self.images[:train_count]}
        val_ids = {img['id'] for img in self.images[train_count:]}

        # Assign splits
        self.train_images = [
            img for img in self.images if img['id'] in train_ids]
        self.val_images = [img for img in self.images if img['id'] in val_ids]

        self.train_calib = [
            cal for cal in self.calib_data if cal['id'] in train_ids]
        self.val_calib = [
            cal for cal in self.calib_data if cal['id'] in val_ids]

        self.train_velodyne = [
            vel for vel in self.velodyne_data if vel['id'] in train_ids]
        self.val_velodyne = [
            vel for vel in self.velodyne_data if vel['id'] in val_ids]

        self.train_annotations = [
            ann for ann in self.annotations if ann['image_id'] in train_ids]
        self.val_annotations = [
            ann for ann in self.annotations if ann['image_id'] in val_ids]

        print(
            f"Split data into {len(self.train_images)} training and {len(self.val_images)} validation images.")
        print(
            f"Split calibration into {len(self.train_calib)} training and {len(self.val_calib)} validation files.")
        print(
            f"Split Velodyne into {len(self.train_velodyne)} training and {len(self.val_velodyne)} validation files.")
        print(
            f"Assigned {len(self.train_annotations)} annotations to training and {len(self.val_annotations)} annotations to validation.")


    # def split_data(self, train_ratio=0.8):
    #     """Split data into training and validation sets."""
    #     total_images = len(self.images)
    #     train_count = int(total_images * train_ratio)

    #     random.shuffle(self.images)
    #     self.train_images = self.images[:train_count]
    #     self.val_images = self.images[train_count:]

    #     calib_train_count = int(len(self.calib_data) * train_ratio)
    #     self.train_calib = self.calib_data[:calib_train_count]
    #     self.val_calib = self.calib_data[calib_train_count:]

    #     velodyne_train_count = int(len(self.velodyne_data) * train_ratio)
    #     self.train_velodyne = self.velodyne_data[:velodyne_train_count]
    #     self.val_velodyne = self.velodyne_data[velodyne_train_count:]

    #     print(
    #         f"Split data into {len(self.train_images)} training and {len(self.val_images)} validation images.")
    #     print(
    #         f"Split calibration into {len(self.train_calib)} training and {len(self.val_calib)} validation files.")
    #     print(
    #         f"Split Velodyne into {len(self.train_velodyne)} training and {len(self.val_velodyne)} validation files.")

    #     # Assign annotations to train and val based on image splits
    #     train_image_ids = set(img['id'] for img in self.train_images)
    #     val_image_ids = set(img['id'] for img in self.val_images)

    #     for ann in self.annotations:
    #         if ann['image_id'] in train_image_ids:
    #             self.train_annotations.append(ann)
    #         elif ann['image_id'] in val_image_ids:
    #             self.val_annotations.append(ann)

    #     print(
    #         f"Assigned {len(self.train_annotations)} annotations to training and {len(self.val_annotations)} annotations to validation.")


    def create_coco_json(self):
        """Create COCO JSON files for train, val, and test sets."""
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
            ],
            "categories": self.categories
        }

        def filter_consistent_ids(images, calib, velodyne, annotations):
            image_ids = {img['id'] for img in images}
            calib_ids = {cal['id'] for cal in calib}
            velodyne_ids = {vel['id'] for vel in velodyne}
            annotation_ids = {ann['image_id'] for ann in annotations}
            consistent_ids = image_ids & calib_ids & velodyne_ids & annotation_ids
            return consistent_ids

        splits = ['train', 'val', 'test']
        coco_data = {}

        for split in splits:
            images = getattr(self, f"{split}_images", [])
            calib = getattr(self, f"{split}_calib", [])
            velodyne = getattr(self, f"{split}_velodyne", [])
            annotations = getattr(self, f"{split}_annotations", [])

            consistent_ids = filter_consistent_ids(
                images, calib, velodyne, annotations)
            images = [img for img in images if img['id'] in consistent_ids]
            calib = [cal for cal in calib if cal['id'] in consistent_ids]
            velodyne = [vel for vel in velodyne if vel['id'] in consistent_ids]
            annotations = [
                ann for ann in annotations if ann['image_id'] in consistent_ids]

            coco_data[f"{split}_image"] = {**base_info, "images": images}
            coco_data[f"{split}_calib"] = {**base_info,
                                           "calibration": calib, "type": "calibration"}
            coco_data[f"{split}_velodyne"] = {**base_info,
                                              "velodyne": velodyne, "type": "point_cloud"}
            coco_data[f"{split}_annotations"] = {**base_info,
                                                 "annotations": annotations, "categories": self.categories}

        return coco_data

    # def create_coco_json(self):
    #     """Create COCO JSON files for train, val, and test sets."""
    #     base_info = {
    #         "info": {
    #             "description": "KITTI Dataset converted to COCO format",
    #             "url": "http://www.cvlibs.net/datasets/kitti/",
    #             "version": "1.0",
    #             "year": f"{datetime.now().year}",
    #             "contributor": "Aryan",
    #             "date_created": f"{datetime.now().isoformat()}"
    #         },
    #         "licenses": [
    #             {
    #                 "id": 1,
    #                 "name": "Attribution-NonCommercial-ShareAlike License",
    #                 "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    #             }
    #         ],
    #         "categories": self.categories
    #     }

    #     return {
    #         "train_image": {
    #             **base_info,
    #             "images": self.train_images,
    #             # "annotations": self.train_annotations
    #         },
    #         "val_image": {
    #             **base_info,
    #             "images": self.val_images,
    #             # "annotations": self.val_annotations
    #         },
    #         "train_calib": {
    #             **base_info,
    #             "calibration": self.train_calib,
    #             "annotations": [],
    #             "categories": [],
    #             "type": "calibration"
    #         },
    #         "val_calib": {
    #             **base_info,
    #             "calibration": self.val_calib,
    #             "annotations": [],
    #             "categories": [],
    #             "type": "calibration"
    #         },
    #         "train_velodyne": {
    #             **base_info,
    #             "velodyne": self.train_velodyne,
    #             "annotations": [],
    #             "categories": [],
    #             "type": "point_cloud"
    #         },
    #         "val_velodyne": {
    #             **base_info,
    #             "velodyne": self.val_velodyne,
    #             "annotations": [],
    #             "categories": [],
    #             "type": "point_cloud"
    #         },
    #         "test_image": {
    #             **base_info,
    #             "images": self.test_images,
    #             "annotations": [],
    #             "categories": self.categories
    #         },
    #         "test_calib": {
    #             **base_info,
    #             "calibration": self.test_calib_data,
    #             "annotations": [],
    #             "categories": [],
    #             "type": "calibration"
    #         },
    #         "test_velodyne": {
    #             **base_info,
    #             "velodyne": self.test_velodyne_data,
    #             "annotations": [],
    #             "categories": [],
    #             "type": "point_cloud"
    #         },
    #         # Below are the new separate annotation entries:
    #         "train_annotations": {
    #             **base_info,
    #             "annotations": self.train_annotations,
    #             "categories": self.categories
    #         },
    #         "val_annotations": {
    #             **base_info,
    #             "annotations": self.val_annotations,
    #             "categories": self.categories
    #         },
    #         "test_annotations": {
    #             **base_info,
    #             "annotations": [],
    #             "categories": self.categories
    #         },
    #     }

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

            for split in ['train', 'val']:
                images = getattr(self, f"{split}_images")
                annotations = self.train_annotations if split == 'train' else self.val_annotations
                skipped_images = 0

                # Create a mapping from image_id to image data for quick access
                image_id_map = {image['id']: image for image in images}

                for image in tqdm(images, desc=f"Normalizing {split} images"):
                    try:
                        image_path = Path(image['file_name'])
                        if not image_path.exists():
                            skipped_images += 1
                            print(f"Image not found: {image_path}. Skipping.")
                            continue

                        with Image.open(image_path) as img:
                            original_width, original_height = img.size
                            img_resized = img.resize(
                                target_size, Image.Resampling.LANCZOS)
                            resized_file_name = f"resized_{image_path.name}"
                            resized_image_path = normalized_images_dir / split
                            resized_image_path.mkdir(
                                parents=True, exist_ok=True)
                            img_resized.save(
                                resized_image_path / resized_file_name)

                        # Update image metadata
                        image['file_name'] = str(
                            resized_image_path / resized_file_name)
                        image['width'], image['height'] = target_size

                        # Update annotations
                        for ann in annotations:
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
                        print(
                            f"Error processing image {image['file_name']}: {e}")
                        skipped_images += 1

                print(
                    f"Normalized {split} images. Skipped images: {skipped_images}")

        except Exception as e:
            print(f"Error during normalization: {e}")


    def save_data(self):
        """Save COCO data to output folder."""
        self.coco_output.mkdir(parents=True, exist_ok=True)

        coco_jsons = self.create_coco_json()

        for key, coco_data in coco_jsons.items():
            match = regex.match(key)
            if not match:
                print(f"Skipping key '{key}' as it does not match the regex.")
                continue
            prefix, suffix = match.groups()
            (self.coco_output / prefix).mkdir(parents=True, exist_ok=True)
            with open(self.coco_output / prefix / f"{key}.json", "w") as json_file:
                json.dump(coco_data, json_file, indent=4)

            # Also save as YAML
            with open(self.coco_output / prefix / f"{key}.yaml", "w") as yaml_file:
                yaml.dump(coco_data, yaml_file, default_flow_style=False)

        print(f"COCO data saved to {self.coco_output}.")

    # def save_data(self):
    #     """Save COCO data to output folder."""
    #     self.coco_output.mkdir(parents=True, exist_ok=True)

    #     coco_jsons = self.create_coco_json()

    #     for key, coco_data in coco_jsons.items():
    #         match = regex.match(key)
    #         if not match:
    #             print(f"Skipping key '{key}' as it does not match the regex.")
    #             continue
    #         prefix, suffix = match.groups()
    #         # e.g. prefix='train' suffix='image' (or 'annotations', etc.)
    #         (self.coco_output / prefix).mkdir(parents=True, exist_ok=True)
    #         with open(self.coco_output / prefix / f"{key}.json", "w") as json_file:
    #             json.dump(coco_data, json_file, indent=4)

    #         # Also save as YAML
    #         with open(self.coco_output / prefix / f"{key}.yaml", "w") as yaml_file:
    #             yaml.dump(coco_data, yaml_file, default_flow_style=False)

    #     print(f"COCO data saved to {self.coco_output}.")

    def get_detailed_file_metadata(self, file_path):
        """Get detailed metadata of a file."""
        stat_info = os.stat(file_path)
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_size": stat_info.st_size,
            "created": datetime.fromtimestamp(stat_info.st_ctime),
            "modified": datetime.fromtimestamp(stat_info.st_mtime),
            "accessed": datetime.fromtimestamp(stat_info.st_atime),
        }

        mime = magic.Magic(mime=True)
        metadata["mime_type"] = mime.from_file(file_path)

        return metadata
