"""
Created on 29/12/2024

@author: Aryan

Filename: YOLO_trainer.py
Relative Path: src/train/YOLO_trainer.py
"""

import json
import logging
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import wandb
from train.YOLO_model import KITTIMultiModalDataset
from config.config import Config

def get_fasterrcnn_model(num_classes: int):
    """
    Returns a Faster R-CNN model, pre-trained on COCO,
    with a new classification head for num_classes.
    TorchVision expects num_classes = (# real classes) + 1 for background.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT")
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class TorchVisionTrainer:
    def __init__(self, config: Config):
        """
        config should contain:
          - data_root (Path)
          - batch_size (int)
          - num_classes (int)
          - lr (float)
          - num_epochs (int)
          - device (str or 'auto')
          - use_wandb (bool)
          - wandb_project_name (str)
          - patience (int) -> for early stopping
          - min_delta (float) -> for early stopping
          - any other relevant hyperparameters
        """
        self.config = config

        # Early Stopping params
        self.patience = getattr(config, "patience", 5)
        self.min_delta = getattr(config, "min_delta", 1e-4)

        # Device logic
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device

        print(f"Using device: {self.device}")

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("train.TorchVisionTrainer")

        # Initialize W&B if configured
        self.use_wandb = getattr(config, "use_wandb", False)
        if self.use_wandb:
            wandb.init(project=config.wandb_project_name)
            wandb.config.update({
                "batch_size": config.batch_size,
                "lr": config.lr,
                "num_epochs": config.num_epochs,
                "num_classes": config.num_classes,
                "device": self.device,
                "patience": self.patience,
                "min_delta": self.min_delta,
            })

        # Build datasets
        self.train_dataset = KITTIMultiModalDataset(
            coco_dir=config.data_root,
            split="train",
            image_size=config.target,
            num_classes=config.num_classes
        )
        self.val_dataset = KITTIMultiModalDataset(
            coco_dir=config.data_root,
            split="val",
            image_size=config.target,
            num_classes=config.num_classes
        )

        # Build dataloaders
        # IMPORTANT: set num_workers=0 if you're on macOS MPS to avoid certain issues
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn
        )

        # Initialize model & optimizer
        self.model = get_fasterrcnn_model(config.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    @staticmethod
    def collate_fn(batch):
        """
        TorchVision detection models expect [images, targets] as lists of length B.
        So we separate them out from the dataset's (image, target) tuples.
        """
        images = []
        targets = []
        for img, tgt in batch:
            images.append(img)
            targets.append(tgt)
        return images, targets

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        loader = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch}/{self.config.num_epochs}",
            leave=False
        )

        for i, (images, targets) in enumerate(loader):
            # Move data to the chosen device
            images = [img.to(self.device).contiguous() for img in images]

            new_targets = []
            for t in targets:
                new_t = {
                    "boxes": t["boxes"].to(self.device).contiguous(),
                    "labels": t["labels"].to(self.device).contiguous(),
                    "image_id": t["image_id"],
                }
                new_targets.append(new_t)

            # Forward pass => returns a dict of losses
            loss_dict = self.model(images, new_targets)
            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Log to W&B
            if self.use_wandb:
                wandb.log({"train_batch_loss": loss.item()})

            # Update the tqdm progress bar
            loader.set_postfix({"batch_loss": loss.item()})

            # Optional: print logs every N steps
            if i % 10 == 0:
                self.logger.info(
                    f"[Epoch {epoch}][Step {i}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"** Epoch {epoch} Training Loss: {avg_loss:.4f}")

        if self.use_wandb:
            wandb.log({"train_epoch_loss": avg_loss, "epoch": epoch})

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        loader = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch}/{self.config.num_epochs}",
            leave=False
        )

        for i, (images, targets) in enumerate(loader):
            images = [img.to(self.device).contiguous() for img in images]

            new_targets = []
            for t in targets:
                new_t = {
                    "boxes": t["boxes"].to(self.device),
                    "labels": t["labels"].to(self.device),
                    "image_id": t["image_id"],
                }
                new_targets.append(new_t)

            # In eval mode, passing targets returns the detection loss
            loss_dict = self.model(images, new_targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss.item()

            loader.set_postfix({"batch_loss": loss.item()})

        avg_val_loss = total_loss / len(self.val_loader)
        self.logger.info(
            f"** Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

        if self.use_wandb:
            wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        return avg_val_loss

    def train(self):
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, self.config.num_epochs + 1):
            self.logger.info(
                f"===== EPOCH {epoch} / {self.config.num_epochs} =====")
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            # Early Stopping Check
            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_checkpoint(epoch, best=True)
            else:
                epochs_no_improve += 1

            # If patience is exceeded, stop training
            if epochs_no_improve >= self.patience:
                self.logger.info(
                    f"No improvement for {self.patience} consecutive epochs. Early stopping.")
                break

    def overfit_on_batch(self, num_steps=100):
        """
        Debug function to see if the model can overfit on a single batch
        (or a small subset) from the DataLoader.
        """
        self.model.train()

        # Grab just 1 batch from the loader
        single_batch = next(iter(self.train_loader))
        images, targets = single_batch

        # Move to device
        images = [img.to(self.device).contiguous() for img in images]
        new_targets = []
        for t in targets:
            new_t = {
                "boxes": t["boxes"].to(self.device).contiguous(),
                "labels": t["labels"].to(self.device).contiguous(),
                "image_id": t["image_id"],
            }
            new_targets.append(new_t)

        for step in range(num_steps):
            loss_dict = self.model(images, new_targets)
            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Optionally log
            if step % 10 == 0:
                self.logger.info(
                    f"[Overfit step {step}/{num_steps}] Loss: {loss.item():.4f}")

    def save_checkpoint(self, epoch, best=False):
        model_path = Path("output")
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        fname = f"{self.config.model_name}.pt" if best else f"{self.config.model_name}_checkpoint_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), model_path / fname)
        self.logger.info(f"Saved model weights to {model_path / fname}")

        # Save model configuration as JSON
        json_fname = f"{self.config.model_name}.json" if best else f"{self.config.model_name}_checkpoint_epoch_{epoch}.json"
        model_config = {
            "epoch": epoch,
            "best": best,
            "num_classes": self.config.num_classes,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.lr,
            "num_epochs": self.config.num_epochs,
            "device": self.device,
        }
        with open(model_path / json_fname, "w") as json_file:
            json.dump(model_config, json_file, indent=4)
        self.logger.info(
            f"Saved model configuration to {model_path / json_fname}")
