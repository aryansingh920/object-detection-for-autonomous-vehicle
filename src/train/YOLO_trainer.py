"""
Created on 29/12/2024

@author: Aryan

Filename: train_model.py
Relative Path: src/train/train_model.py
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

# TorchVision detection
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# TQDM for progress bars
from tqdm import tqdm

# Our dataset
from train.YOLO_model import KITTIMultiModalDataset


def get_fasterrcnn_model(num_classes: int):
    """
    Returns a Faster R-CNN model, pre-trained on COCO,
    with a new classification head for `num_classes`.
    TorchVision expects num_classes = (# real classes) + 1 for background.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class TorchVisionTrainer:
    def __init__(
        self,
        data_root: Path,
        batch_size: int = 2,
        num_classes: int = 9,  # e.g. 8 KITTI classes + 1 background
        lr: float = 1e-3,
        num_epochs: int = 5,
        device: str = "cpu",
    ):
        """
        data_root: e.g. Path("data/coco")
        We'll load from data_root/'train' and data_root/'val'.
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("train.TorchVisionTrainer")

        # Build datasets
        self.train_dataset = KITTIMultiModalDataset(
            coco_dir=data_root,
            split="train",
            image_size=(640, 640),
            num_classes=num_classes
        )
        self.val_dataset = KITTIMultiModalDataset(
            coco_dir=data_root,
            split="val",
            image_size=(640, 640),
            num_classes=num_classes
        )

        # Build dataloaders
        # IMPORTANT: set num_workers=0 on Apple MPS to avoid _share_filename_ error
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,            # <--- to avoid Mac MPS issues
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,            # <--- to avoid Mac MPS issues
            collate_fn=self.collate_fn
        )

        # Initialize model & optimizer
        self.model = get_fasterrcnn_model(num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

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

        # Wrap self.train_loader in tqdm to show the progress bar
        loader = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch}/{self.num_epochs}",
            leave=False
        )

        for i, (images, targets) in enumerate(loader):
            # Move data to device
            images = [img.to(self.device) for img in images]
            images = [img.contiguous() for img in images]
            new_targets = []
            for t in targets:
                new_t = {
                    "boxes": t["boxes"].to(self.device).contiguous(),
                    "labels": t["labels"].to(self.device).contiguous(),
                    "image_id": t["image_id"],  # can stay on CPU
                }
                new_targets.append(new_t)

            # Forward pass => returns a dict of losses
            loss_dict = self.model(images, new_targets)
            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                self.logger.info(
                    f"[Epoch {epoch}][Step {i}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

            # Update the tqdm progress bar
            loader.set_postfix({"batch_loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"** Epoch {epoch} Training Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        loader = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch}/{self.num_epochs}",
            leave=False
        )
        for i, (images, targets) in enumerate(loader):
            images = [img.to(self.device) for img in images]
            new_targets = []
            for t in targets:
                new_t = {
                    "boxes": t["boxes"].to(self.device),
                    "labels": t["labels"].to(self.device),
                    "image_id": t["image_id"],  # can stay CPU
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
        return avg_val_loss

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"===== EPOCH {epoch} / {self.num_epochs} =====")
            self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            # If improved, save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, best=True)

    def save_checkpoint(self, epoch, best=False):
        model_path = Path("output")
        model_path.mkdir(parents=True, exist_ok=True)
        fname = "best_model.pt" if best else f"checkpoint_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), model_path / fname)
        self.logger.info(f"Saved model to {model_path / fname}")
