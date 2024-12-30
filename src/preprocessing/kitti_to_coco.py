"""
Created on 29/12/2024

@author: Aryan

Filename: kitti_to_coco.py

Relative Path: src/preprocessing/kitti_to_coco.py
"""

import os
import json
import datetime
from PIL import Image


def kitti_to_coco(kitti_base_path, output_json_path):
    """
    Converts the KITTI dataset to COCO format and saves it to a JSON file.

    Parameters:
    - kitti_base_path: str, path to the KITTI dataset directory.
    - output_json_path: str, path to save the output COCO JSON file.
    """
    # Initialize COCO format dictionary
    coco_format = {
        "info": {
            "description": "KITTI Dataset converted to COCO format",
            "url": "http://www.cvlibs.net/datasets/kitti/",
            "version": "1.0",
            "year": 2024,
            "contributor": "Aryan",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Define KITTI categories
    categories = ["Car", "Van", "Truck", "Pedestrian",
                  "Person_sitting", "Cyclist", "Tram", "Misc"]
    for i, category in enumerate(categories):
        coco_format["categories"].append({
            "id": i + 1,
            "name": category,
            "supercategory": "none"
        })

    # Paths for KITTI images and labels
    image_dir = os.path.join(kitti_base_path, "data_object_image_2", "training", "image_2")
    label_dir = os.path.join(kitti_base_path, "training", "label_2")

    image_id, annotation_id = 1, 1

    # Iterate through KITTI images
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            img = Image.open(image_path)
            width, height = img.size

            # Add image information to COCO format
            coco_format["images"].append({
                "id": image_id,
                "license": 1,
                "file_name": filename,
                "height": height,
                "width": width,
                "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Corresponding label file
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 15 and parts[0] in categories:
                            # Bounding box coordinates: x_min, y_min, width, height
                            bbox = [float(parts[4]), float(parts[5]),
                                    float(parts[6]) - float(parts[4]),
                                    float(parts[7]) - float(parts[5])]
                            area = bbox[2] * bbox[3]

                            # Add annotation to COCO format
                            coco_format["annotations"].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": categories.index(parts[0]) + 1,
                                "bbox": bbox,
                                "area": area,
                                "iscrowd": 0
                            })
                            annotation_id += 1

            image_id += 1

    # Save COCO format JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f, indent=2)
    print(f"COCO JSON saved at {output_json_path}")


if __name__ == "__main__":
    kitti_base_path = r"data/kitti"
    output_json_path = r"output/kitti_coco.json"
    kitti_to_coco(kitti_base_path, output_json_path)
