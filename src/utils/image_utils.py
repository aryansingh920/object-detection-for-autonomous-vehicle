"""
Created on 29/12/2024

@author: Aryan

Filename: image_utils.py

Relative Path: src/utils/image_utils.py
"""

from PIL import Image


def resize_image(image_path, target_size):
    try:
        with Image.open(image_path) as img:
            return img.resize(target_size, Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None


def save_image(img, output_path):
    try:
        img.save(output_path)
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
