"""
Created on 30/12/2024

@author: Aryan

Filename: main.py
Relative Path: src/main.py
"""

from pipeline import preprocess_datasets, validate_dataset, train
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
        train()



if __name__ == "__main__":
    main()
