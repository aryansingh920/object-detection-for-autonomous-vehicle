"""
Created on 29/12/2024

@author: Aryan

Filename: test_model.py

Relative Path: src/test/test_model.py
"""

import os
from ultralytics import YOLO
import wandb
from src.config import Config


def init_wandb(config: Config):
    """
    Initialize Weights & Biases for evaluation tracking.
    """
    wandb.init(
        project=f"yolov8-{config.dataset_type.lower()}-eval",
        name=f"{config.dataset_type}_evaluation",
        config={
            "model_weights": config.trained_weights_path,
            "dataset": config.dataset_type,
            "image_size": config.img_size,
            "data_yaml": config.dataset_yaml
        },
        job_type="evaluation"
    )


def evaluate_model(config: Config):
    """
    Evaluate a trained YOLOv8 model on the validation/test set
    specified in the YOLO dataset YAML. Logs results to W&B.
    """
    # Initialize W&B
    init_wandb(config)

    try:
        # ---------------------------
        # 1. Load the trained model
        # ---------------------------
        trained_weights_path = os.path.join(
            "runs",
            "detect",
            f"{config.dataset_type}_experiment",
            "weights",
            "best.pt"
        )

        if not os.path.exists(trained_weights_path):
            raise FileNotFoundError(
                f"Could not find trained weights at {trained_weights_path}.\n"
                "Please update the path in evaluate_model() or your config."
            )

        print(f"Loading model from: {trained_weights_path}")
        model = YOLO(trained_weights_path)

        # ---------------------------
        # 2. Evaluate the model
        # ---------------------------
        print(f"Evaluating model on dataset: {config.dataset_yaml}")
        results = model.val(
            data=config.dataset_yaml,
            imgsz=config.img_size,
            conf=0.25,  # Confidence threshold
            iou=0.6,    # NMS IoU threshold
            max_det=300,  # Maximum detections per image
            device='cuda',  # Use GPU if available
            verbose=True,  # Print verbose output
            save_txt=True,  # Save results to *.txt
            save_conf=True,  # Save confidences in --save-txt labels
            save_json=True,  # Save results to *.json
            project=wandb.run.project,  # Save to project/name
            name=wandb.run.name,  # Save to project/name
            plots=True,  # Generate plots
        )

        # Log metrics to W&B
        metrics = {
            "mAP50": results.maps[0],  # mAP at IoU=0.5
            "mAP50-95": results.maps[1],  # mAP at IoU=0.5:0.95
            "precision": results.results_dict['metrics/precision(B)'],
            "recall": results.results_dict['metrics/recall(B)'],
            "f1": results.results_dict['metrics/F1(B)'],
        }

        # Log confusion matrix and other plots if available
        if hasattr(results, 'confusion_matrix'):
            wandb.log({"confusion_matrix": wandb.Image(
                results.confusion_matrix.plot())})

        # Log PR curve if available
        if os.path.exists(os.path.join(wandb.run.name, 'PR_curve.png')):
            wandb.log({"PR_curve": wandb.Image(
                os.path.join(wandb.run.name, 'PR_curve.png'))})

        # Log all metrics
        wandb.log(metrics)

        print("Evaluation complete!")
        print("Evaluation metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return results

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        # Log error to W&B
        wandb.log({"error": str(e)})
        raise e

    finally:
        # Always close W&B run
        wandb.finish()


def main():
    try:
        # Use the configuration to evaluate the model
        evaluate_model(Config)
    except Exception as e:
        print(f"Error in main: {str(e)}")
        if wandb.run is not None:
            wandb.finish()
        raise e


if __name__ == "__main__":
    main()
