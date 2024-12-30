"""
Created on 29/12/2024

@author: Aryan

Filename: main.py

Relative Path: src/main.py
"""

from preprocessing.main import KITTIToCOCOConverter

def main():
    # Define paths (Update these paths as per your directory structure)
    kitti_base_path = r"data/kitti"
    coco_base_path = r"data/coco"
    train_val_ratio = 0.8
    normalize = True
    target_size = (1242, 375)

    # Step 1: Preprocess datasets (convert only KITTI)
    print("\n=== Step 1: Preprocessing Datasets ===")
    converter = KITTIToCOCOConverter(
        kitti_root=kitti_base_path, coco_output=coco_base_path)
    converter.parse_kitti_data()
    converter.organize_data()
    converter.split_data(train_ratio=train_val_ratio)
    converter.normalize_and_standardize_dataset(
        target_size=target_size) if normalize else None
    converter.save_data()





if __name__ == "__main__":
    main()
