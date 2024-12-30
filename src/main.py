"""
Created on 29/12/2024

@author: Aryan

Filename: main.py

Relative Path: src/main.py
"""

from preprocessing.main import KITTIToCOCOConverter
from config.config import Config

def main():
    # Define paths (Update these paths as per your directory structure)


    # Step 1: Preprocess datasets (convert only KITTI)
    print("\n=== Step 1: Preprocessing Datasets ===")
    converter = KITTIToCOCOConverter(
        kitti_root=Config.kitti_base_path, coco_output=Config.coco_base_path)
    converter.parse_kitti_data()
    converter.organize_data()
    converter.split_data(train_ratio=Config.train_val_ratio)
    converter.normalize_and_standardize_dataset(
        target_size=Config.target_size) if Config.normalize else None
    converter.save_data()





if __name__ == "__main__":
    main()
