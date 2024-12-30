"""
Created on 29/12/2024

@author: Aryan

Filename: coco_validator.py

Relative Path: src/utils/coco_validator.py
"""

import json


def validate_coco_format(data):
    required_sections = ['info', 'licenses',
                         'images', 'annotations', 'categories']
    for section in required_sections:
        if section not in data:
            print(f"Missing required section: {section}")
            return False
    if not data['categories']:
        print("Categories section is empty")
        return False
    if not data['annotations']:
        print("Annotations section is empty")
        return False
    return True


def validate_annotation_references(data):
    image_ids = {img['id'] for img in data['images']}
    category_ids = {cat['id'] for cat in data['categories']}

    for ann in data['annotations']:
        if ann['image_id'] not in image_ids:
            print(f"Invalid image ID {ann['image_id']} in annotation")
            return False
        if ann['category_id'] not in category_ids:
            print(f"Invalid category ID {ann['category_id']} in annotation")
            return False
    return True
