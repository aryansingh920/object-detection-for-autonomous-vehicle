"""
Created on 29/12/2024

@author: Aryan

Filename: train_model.py

Relative Path: src/train/train_model.py
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from config.config import YOLOConfig
from PIL import Image
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm
import os
import KITTIMultiModal_Dataset as KITTIMultiModalDataset
import YOLO_model as YOLOModel


class YOLOTrainer:
    """YOLO trainer for multi-modal object detection."""

    def __init__(
        self,
        config: YOLOConfig,
        data_dir: Path,
        output_dir: Path
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        self.setup_wandb()

        # Initialize model, optimizer, and loss function
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss()

        # Setup data loaders
        self.setup_data_loaders()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        wandb.init(
            project=self.config.defaults.get('wandb_project', 'YOLO-Training'),
            config=self.config.get_config(),
            save_code=True
        )
        self.logger.info("WandB initialized.")

    def setup_model(self):
        """Initialize YOLO model with multi-modal support."""
        # Placeholder for model initialization
        # Replace YOLOModel with your actual model class
        self.model = YOLOModel(
            num_classes=self.config.defaults['num_classes'],
            modalities=['image', 'calib', 'velodyne']
        ).to(self.config.defaults['device'])
        self.logger.info("Model initialized.")

    def setup_optimizer(self):
        """Set up optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.defaults['learning_rate']
        )
        if self.config.defaults.get('scheduler') == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.defaults.get('step_size', 10),
                gamma=self.config.defaults.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        self.logger.info("Optimizer and scheduler set up.")

    def setup_loss(self):
        """Set up loss functions for all modalities."""
        # Placeholder: Use appropriate loss functions for your task
        self.criterion = nn.CrossEntropyLoss()
        self.logger.info("Loss function set up.")

    def setup_data_loaders(self):
        """Set up data loaders for train and validation sets."""
        # Create datasets
        self.train_dataset = KITTIMultiModalDataset(
            self.data_dir, 'train', self.config
        )
        self.val_dataset = KITTIMultiModalDataset(
            self.data_dir, 'val', self.config
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.defaults['batch_size'],
            shuffle=True,
            num_workers=self.config.defaults.get('num_workers', 4),
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.defaults['batch_size'],
            shuffle=False,
            num_workers=self.config.defaults.get('num_workers', 4),
            pin_memory=True
        )
        self.logger.info("Data loaders set up.")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch in progress_bar:
            images = batch['image'].to(self.config.defaults['device'])
            calib = batch['calib'].to(self.config.defaults['device'])
            velodyne = batch['velodyne'].to(self.config.defaults['device'])
            targets = batch['targets'].to(self.config.defaults['device'])

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, calib, velodyne)

            # Compute loss
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_loss = epoch_loss / len(self.train_loader)
        wandb.log({'Train Loss': avg_loss})
        return avg_loss

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader, desc='Validation', leave=False)
            for batch in progress_bar:
                images = batch['image'].to(self.config.defaults['device'])
                calib = batch['calib'].to(self.config.defaults['device'])
                velodyne = batch['velodyne'].to(self.config.defaults['device'])
                targets = batch['targets'].to(self.config.defaults['device'])

                # Forward pass
                outputs = self.model(images, calib, velodyne)

                # Compute loss
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                progress_bar.set_postfix({'Val Loss': loss.item()})

        avg_val_loss = val_loss / len(self.val_loader)
        wandb.log({'Val Loss': avg_val_loss})
        return avg_val_loss

    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.config.defaults['num_epochs'] + 1):
            self.logger.info(
                f"Epoch {epoch}/{self.config.defaults['num_epochs']}"
            )
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Log metrics
            self.logger.info(
                f"Epoch {epoch}/{self.config.defaults['num_epochs']}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            wandb.log({
                'Epoch': epoch,
                'Train Loss': train_loss,
                'Val Loss': val_loss
            })

            # Step the scheduler if available
            if self.scheduler:
                self.scheduler.step()

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss - self.config.defaults.get('min_delta', 0.0):
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss, best=True)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.defaults.get('patience', 10):
                self.logger.info("Early stopping triggered")
                break

        # Save the final model
        self.save_model()

    def save_checkpoint(self, epoch: int, val_loss: float, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.get_config()
        }

        if best:
            checkpoint_path = self.output_dir / 'best_checkpoint.pt'
        else:
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        wandb.save(str(checkpoint_path))

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.config.defaults['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['val_loss']

    def save_model(self):
        """Save the final model."""
        model_path = self.output_dir / 'final_model.pt'
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Saved final model to {model_path}")
        wandb.save(str(model_path))


# def main():
#     # Example usage
#     config_path = Path('config/config.yaml')  # Update with your config path
#     config = YOLOConfig(config_path)

#     data_dir = Path('data/coco')
#     output_dir = Path('outputs')

#     trainer = YOLOTrainer(config, data_dir, output_dir)
#     trainer.train()


# if __name__ == "__main__":
#     main()
