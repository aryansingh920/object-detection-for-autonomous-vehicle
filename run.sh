#!/bin/zsh

rm -rf .venv
# Create a new virtual environment
python3.10 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install PyTorch with MPS support
pip3.10 install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Install other requirements
pip3.10 install -r requirements.txt

 python3 src/preprocessing/update_json_paths.py
# python3.10 src/test_setup.py
python3.10 src/main.py

# python3.10 -m src.pipeline.train_model \
#     --dataset_type kitti \
#     --data_root /path/to/kitti/dataset \
#     --output_json kitti_annotations.json \
#     --dataset_yaml dataset.yaml \
#     --epochs 50 \
#     --batch_size 4 \
#     --img_size 640 \
#     --model_arch yolov8n.pt

