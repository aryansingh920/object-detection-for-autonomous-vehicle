"""
Created on 29/12/2024

@author: Aryan

Filename: validate_dataset.py

Relative Path: src/preprocessing/validate_dataset.py
"""

import json


def validate_coco_dataset(file_path):
    """
    Validates a COCO-format dataset JSON file for completeness and consistency.

    Parameters:
    - file_path: str, path to the COCO-format JSON file.

    Returns:
    - bool: True if validation passes, False otherwise.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check required sections
        required_sections = ['info', 'licenses', 'images', 'annotations', 'categories']
        for section in required_sections:
            if section not in data:
                print(f"Error: Missing required section: {section}")
                return False

        # Validate images
        image_ids = set()
        for img in data['images']:
            if 'id' not in img or 'file_name' not in img or 'width' not in img or 'height' not in img:
                print(f"Error: Image entry missing required fields: {img}")
                return False
            image_ids.add(img['id'])

        # Validate categories
        category_ids = set()
        for cat in data['categories']:
            if 'id' not in cat or 'name' not in cat:
                print(f"Error: Category entry missing required fields: {cat}")
                return False
            category_ids.add(cat['id'])

        # Validate annotations
        annotation_ids = set()
        for ann in data['annotations']:
            if 'id' not in ann or 'image_id' not in ann or 'category_id' not in ann or 'bbox' not in ann:
                print(f"Error: Annotation entry missing required fields: {ann}")
                return False
            if ann['image_id'] not in image_ids:
                print(f"Error: Annotation refers to invalid image ID {ann['image_id']}")
                return False
            if ann['category_id'] not in category_ids:
                print(f"Error: Annotation refers to invalid category ID {ann['category_id']}")
                return False
            if ann['id'] in annotation_ids:
                print(f"Error: Duplicate annotation ID found: {ann['id']}")
                return False
            annotation_ids.add(ann['id'])

        print("Dataset validation passed!")
        return True

    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return False
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return False
    except Exception as e:
        print(f"Unexpected error during validation: {e}")
        return False


if __name__ == "__main__":
    file_path = r"<path_to_coco_json>"
    validate_coco_dataset(file_path)
