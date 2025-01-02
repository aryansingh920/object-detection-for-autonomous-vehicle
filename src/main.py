"""
Created on 30/12/2024

@author: Aryan

Filename: main.py
Relative Path: src/main.py
"""
from preprocessing.main import KITTIToCOCOConverter
from preprocessing.validate_dataset import DataValidator
from train.YOLO_trainer import TorchVisionTrainer
from config.config import Config





def main():
    # Step 1: Preprocess datasets (convert only KITTI) if needed
    if Config.preprocess:
        preprocess_datasets()

    # Step 1.1: Validate dataset
    if Config.validate:
        validate_dataset()

    # Step 3: Train model if needed
    if Config.train:
        train_model()


def preprocess_datasets():
    print("\n=== Step 1: Preprocessing Datasets ===")
    converter = KITTIToCOCOConverter(
        kitti_root=Config.kitti_base_path,
        coco_output=Config.coco_base_path
    )
    converter.parse_kitti_data()
    converter.organize_data()
    converter.split_data(train_ratio=Config.train_val_ratio)

    if Config.normalize:
        converter.normalize_and_standardize_dataset(
            target_size=Config.target_size_for_normalization
        )
    converter.save_data()


def validate_dataset():
    dataset_path = Config.coco_base_path
    for dir in ["train", "val"]:
        # if not (dataset_path / dir).exists():
        #     raise FileNotFoundError(
        #         f"Directory {dir} not found in {dataset_path}")
        # else:
        validator = DataValidator(dataset_path=dataset_path, split=dir)
        validator.validate_dataset()

def train_model():
    print("\n=== Step 2: Training Model ===")
    # Create data_root=Path("data/coco") or use Config.coco_base_path
    config = Config()
    # config.data_root = Path(config.coco_base_path)

    # Build trainer from config
    trainer = TorchVisionTrainer(config)
    trainer.overfit_on_batch(num_steps=100)
    trainer.train()


if __name__ == "__main__":
    main()
