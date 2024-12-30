"""
Created on 29/12/2024

@author: Aryan

Filename: main.py

Relative Path: src/main.py
"""

from pipeline.main import preprocess_datasets, process_data, validate_data, train_model, evaluate_model


def main():
    # Define paths (Update these paths as per your directory structure)
    kitti_base_path = r"data/kitti"
    # Remove or comment out any NuScenes related paths
    # nuscenes_base_path = r"<path_to_nuscenes_dataset>"
    output_dir = r"input/coco"
    images_dir = r"input/images"

    # Step 1: Preprocess datasets (convert only KITTI)
    print("\n=== Step 1: Preprocessing Datasets ===")
    # Now only convert KITTI dataset to COCO format
    merged_coco_path = preprocess_datasets(kitti_base_path, output_dir)

    # Step 2: Process data (copy images and normalize)
    print("\n=== Step 2: Processing Data ===")
    normalized_coco_path = process_data(
        merged_coco_path, images_dir, output_dir)

    # Step 3: Validate processed data
    print("\n=== Step 3: Validating Data ===")
    validate_data(normalized_coco_path)

    # Placeholder for training (future work)
    print("\n=== Step 4: Training Model ===")
    train_model()

    # Placeholder for evaluation (future work)
    print("\n=== Step 5: Evaluating Model ===")
    evaluate_model()


if __name__ == "__main__":
    main()
