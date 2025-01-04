"""
Created on 03/01/2025

@author: Aryan

Filename: eval_yolov8.py
Relative Path: src/eval/eval_yolov8.py
"""

import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os


def main(weights_path, data_yaml, save_dir):
    """
    Evaluate a YOLOv8 model on a dataset specified in data_yaml.
    data_yaml is a file containing:
        train: path/to/train
        val: path/to/val
        test: path/to/test
        names: ...
    """

    # 1. Load model
    model = YOLO(weights_path)
    print(f"Loaded model from: {weights_path}")

    # 2. Run validation using YOLOv8's built-in evaluation
    #    This returns a dictionary of metrics including mAP, precision, recall, F1, etc.
    print("Evaluating on dataset:", data_yaml)
    metrics = model.val(
        data=data_yaml,
        conf=0.25,      # Confidence threshold (can adjust)
        iou=0.45,       # IoU threshold (can adjust)
        # Saves plots (PR curve, confusion matrix) in `model.val` default directory
        plots=True,
        save=True,      # Save output results (annotated images, etc.)
        project=save_dir,  # Save results under this directory
        exist_ok=True
    )

    # 3. Print the metrics
    # The returned metrics can include:
    # 'metrics/mAP50(B)' or 'metrics/precision(B)', etc. depending on YOLO's version
    print("\n*** Evaluation Metrics ***")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 4. (Optional) You can also visualize or further process the metrics
    # For example, let's assume we want to plot the precision-recall curve
    # or any other custom chart. YOLO automatically saves them under `save_dir`.
    # Below is just an example of manual plotting if you want to do more.

    # Example: If you have custom arrays of data for PR, you can plot them.
    # YOLOv8 also automatically saves some plots for you:
    #   - labels.jpg
    #   - conf_curve.png
    #   - pr_curve.png
    #   - F1_curve.png
    #   - etc.

    # Example custom plot for demonstration (not necessarily needed):
    # pr_values = [0.9, 0.92, 0.94, ...] # This is hypothetical
    # recall_values = [0.85, 0.87, 0.88, ...]
    # ...
    # plt.plot(recall_values, pr_values)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("Precision-Recall Curve")
    # plt.savefig(os.path.join(save_dir, "custom_pr_curve.png"))
    # plt.close()

    print(f"\nEvaluation results and plots saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 model on a labeled dataset.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the trained YOLOv8 weights file (e.g., models/custom/model_final.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the .yaml file that defines the dataset (train, val, etc.).")
    parser.add_argument("--save_dir", type=str, default="evaluation_results",
                        help="Directory where evaluation results (metrics, plots, etc.) will be saved.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args.weights, args.data, args.save_dir)
