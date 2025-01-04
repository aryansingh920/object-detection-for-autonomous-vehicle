#!/bin/zsh

# rm -rf .venv
# # Create a new virtual environment
# python3.10 -m venv .venv

# # Activate the environment
source .venv/bin/activate

# # Install PyTorch with MPS support
# pip3.10 install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# # Install other requirements
# pip3.10 install -r requirements.txt

#  python3 src/preprocessing/update_json_paths.py
# # python3.10 src/test_setup.py
# python3.10 src/main.py


# clear
# python3.10 src/main.py



python3.10 src/realtime/main.py \
    --weights models/yolo_training/train/weights/best.pt \
    --images_folder data/kitti/raw_Data/2011_09_26/2011_09_26_drive_0032_sync/image_03/data \
    --timestamps data/kitti/raw_Data/2011_09_26/2011_09_26_drive_0032_sync/image_03/timestamps.txt \
    --fps 30

python3.10 src/realtime/main.py \
    --weights models/yolo_training/train/weights/best.pt \
    --images_folder data/kitti/raw_Data/2011_09_26/2011_09_26_drive_0023_sync/image_03/data \
    --timestamps data/kitti/raw_Data/2011_09_26/2011_09_26_drive_0023_sync/image_03/timestamps.txt \
    --fps 30
