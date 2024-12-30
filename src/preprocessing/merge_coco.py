"""
Created on 29/12/2024

@author: Aryan

Filename: merge_coco.py

Relative Path: src/preprocessing/merge_coco.py
"""

import json


def merge_coco_datasets(kitti_json_path, output_json_path):
    """
    Merges COCO annotations from the KITTI dataset into a unified COCO JSON.

    Parameters:
    - kitti_json_path: str, path to the KITTI COCO JSON file.
    - output_json_path: str, path to save the output merged COCO JSON file.
    """
    with open(kitti_json_path, 'r') as f:
        kitti_data = json.load(f)

    # Initialize the merged data
    merged_data = {
        "info": {
            "description": "Merged KITTI COCO dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "Aryan",
            "date_created": "2024-12-29"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Map licenses
    license_map = {license['name']: idx + 1 for idx, license in enumerate(kitti_data['licenses'])}
    merged_data['licenses'] = list(license_map.values())

    # Map categories and preserve their uniqueness
    category_map = {}
    for category in kitti_data['categories']:
        if category['name'] not in category_map:
            category['id'] = len(category_map) + 1
            category_map[category['name']] = category['id']
            merged_data['categories'].append(category)

    # Map images and ensure unique IDs
    image_id_map = {}
    for image in kitti_data['images']:
        image_id_map[image['id']] = len(merged_data['images']) + 1
        image['id'] = image_id_map[image['id']]
        merged_data['images'].append(image)

    # Map annotations to updated image IDs and category IDs
    for ann in kitti_data['annotations']:
        if ann['category_id'] not in category_map:
            print(f"Warning: category_id {ann['category_id']} is missing. Skipping this annotation.")
            continue  # Skip this annotation if the category_id is missing
        if ann['image_id'] not in image_id_map:
            print(f"Warning: image_id {ann['image_id']} is missing. Skipping this annotation.")
            continue
        ann['image_id'] = image_id_map[ann['image_id']]
        ann['category_id'] = category_map[ann['category_id']]
        merged_data['annotations'].append(ann)

    # Save merged data to JSON
    with open(output_json_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    print(f"Merged dataset saved to {output_json_path}")


if __name__ == "__main__":
    kitti_json_path = r"<path_to_kitti_coco_json>"
    output_json_path = r"<output_merged_coco_json>"
    merge_coco_datasets(kitti_json_path, output_json_path)
