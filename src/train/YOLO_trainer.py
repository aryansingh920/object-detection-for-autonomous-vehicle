import os
import json
import yaml
import torch
import random
import shutil

# For YOLOv5
# from pathlib import Path
# import sys
# YOLOV5_REPO_PATH = "/path/to/yolov5"  # If you cloned YOLOv5, set this
# sys.path.append(str(YOLOV5_REPO_PATH))
# from models.yolo import Model
# from train import run as yolov5_train

# For YOLOv8 (Ultralytics)
try:
    from ultralytics import YOLO  # YOLOv8
except ImportError:
    YOLO = None


################################################################################
# 1. Hyperparameter Configuration
################################################################################
class HyperParameters:
    """
    Holds hyperparameters for the training process.
    You can expand this class as needed.
    """

    def __init__(self,
                 model_name: str = "models/yolov8n.pt",  # or "yolov5s.pt"
                 epochs: int = 1,
                 batch_size: int = 4,
                 # single int or (width, height)
                 image_size: int = (1242, 375),
                 lr: float = 0.01,
                 ):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.lr = lr  # We'll feed this into lr0 for YOLOv8


################################################################################
# 2. Device Selection (CPU, CUDA, MPS)
################################################################################
def auto_select_device():
    """
    Automatically select the best available device.
    Checks for:
      1. MPS (Mac M1/M2) if available and built,
         but if MPS performance is slow or not stable,
         we can fallback to CPU for certain operations.
      2. CUDA if available.
      3. Otherwise CPU.
    Returns a string usable by PyTorch (e.g., 'mps', 'cuda:0', or 'cpu').
    """
    # 1. Try MPS
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"

    # 2. Try CUDA
    if torch.cuda.is_available():
        return "cuda"

    # 3. Fallback CPU
    return "cpu"


################################################################################
# 3. Build YOLO Data YAML
################################################################################
def build_yolo_data_yaml(
        train_images_dir: str,
        val_images_dir: str,
        categories: list,
        yaml_path: str = "data/kitti_coco.yaml"
):
    """
    Creates a YOLO-style .yaml file that points to your train/val image directories
    and lists your class names. Saves it to `yaml_path`.

    YOLO expects something like:

    names:
      - class0
      - class1
      ...
    path:
      train: /path/to/train/images
      val:   /path/to/val/images
    """
    # Extract category names sorted by ID to maintain consistent indexing
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    names = [c["name"] for c in sorted_cats]

    data_yaml = {
        "path": os.getcwd(),
        "train": train_images_dir,
        "val": val_images_dir,
        "names": names
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    return yaml_path


################################################################################
# 4. Dataset Helper (to parse COCO-like JSONs, if needed by YOLO training)
################################################################################
def extract_image_paths_from_json(json_file):
    """
    Given *_image.json in COCO format:
    {
      "images": [
        {
          "id": 1,
          "file_name": "...",
          "width": 1280,
          "height": 720,
          ...
        },
        ...
      ]
    }
    We return a list of file paths.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    images = data.get("images", [])
    return [img["file_name"] for img in images]


################################################################################
# 5. Training Function (using YOLOv8 or YOLOv5)
################################################################################
def train_yolo_model(
        config: HyperParameters,
        train_json_image: str = "data/coco/train/train_image.json",
        val_json_image: str = "data/coco/val/val_image.json",
        train_json_annot: str = "data/coco/train/train_annotations.json",
        val_json_annot: str = "data/coco/val/val_annotations.json",
        categories: list = None,
):
    """
    Main function to train YOLO (v5 or v8).
    For YOLOv8, we use Ultralytics' API.
    For YOLOv5, we'd typically call the train.py script or import its functions.
    """

    # 1. Auto-select device
    device = auto_select_device()
    print(f"Using device: {device}")

    # 2. If categories are not passed, load them from JSON:
    if categories is None:
        with open(train_json_annot, "r") as f:
            train_data = json.load(f)
        categories = train_data["categories"]
        print("Loaded categories from train_annotations.json.")

    # 3. Build the data.yaml for YOLO
    train_paths = extract_image_paths_from_json(train_json_image)
    val_paths = extract_image_paths_from_json(val_json_image)

    train_folder = os.path.join("data", "coco", "train_images")
    val_folder = os.path.join("data", "coco", "val_images")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    def symlink_or_copy_images(img_paths, out_folder):
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            src = os.path.abspath(img_path)
            dst = os.path.join(out_folder, filename)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except:
                    shutil.copy2(src, dst)

    print("Preparing training images...")
    symlink_or_copy_images(train_paths, train_folder)

    print("Preparing validation images...")
    symlink_or_copy_images(val_paths, val_folder)

    yaml_file_path = build_yolo_data_yaml(
        train_images_dir=train_folder,
        val_images_dir=val_folder,
        categories=categories,
        yaml_path="data/kitti_coco.yaml"
    )

    # 4. Print out the data.yaml content (Optional)
    with open(yaml_file_path, "r") as f:
        data_yaml_text = f.read()
    print("Data YAML:\n", data_yaml_text)

    # 5. Train with YOLOv8 or YOLOv5
    if YOLO is not None:
        print(f"Starting training with YOLOv8 model: {config.model_name}")
        model = YOLO(config.model_name)
        # For YOLOv8, the initial learning rate is 'lr0'
        results = model.train(
            data=yaml_file_path,
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.image_size,     # single int or list of int
            lr0=config.lr,               # <--- Use lr0 instead of lr
            device=device,
            name="kitti_coco_yolov8_experiment"
        )
        print("Training complete. Results:", results)
        # The best model is saved in 'runs/detect/kitti_coco_yolov8_experiment/weights/best.pt' by default.
    else:
        # YOLOv5 approach (commented out by default)
        raise RuntimeError(
            "Ultralytics YOLO (yolov8) is not installed, and YOLOv5 code is commented out.")


################################################################################
# 6. Example Main
################################################################################
if __name__ == "__main__":
    """
    Example usage. You can adapt this to your own CLI or script invocation.
    """
    # 1. Define your hyperparameters
    config = HyperParameters(
        model_name="model/yolov8n.pt",  # or "yolov5s.pt"
        epochs=10,
        batch_size=4,
        image_size=640,  # single int
        lr=0.001,        # initial LR -> becomes lr0 in YOLOv8
    )

    # 2. Set your JSON paths
    train_json_image = "data/coco/train/train_image.json"
    val_json_image = "data/coco/val/val_image.json"
    train_json_annot = "data/coco/train/train_annotations.json"
    val_json_annot = "data/coco/val/val_annotations.json"

    # 3. You can provide categories directly or let the script load them
    categories = [
        {"id": 1, "name": "Car"},
        {"id": 2, "name": "Pedestrian"},
        {"id": 3, "name": "Cyclist"},
        {"id": 4, "name": "Truck"},
        {"id": 5, "name": "Van"},
        {"id": 6, "name": "Person_sitting"},
        {"id": 7, "name": "Tram"},
        {"id": 8, "name": "Misc"}
    ]

    # 4. Train the model
    train_yolo_model(
        config=config,
        train_json_image=train_json_image,
        val_json_image=val_json_image,
        train_json_annot=train_json_annot,
        val_json_annot=val_json_annot,
        categories=categories
    )

    print("All done!")
