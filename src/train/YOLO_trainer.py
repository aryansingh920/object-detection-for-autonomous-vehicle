import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.models as models
from torchvision.transforms import Compose, ToTensor, Resize
from pathlib import Path
import wandb
from tqdm import tqdm
import logging
import numpy as np
from train.YOLO_model import KITTIMultiModalDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_boxes(boxes, image_size):
    """
    Validate and fix bounding boxes.
    Returns fixed boxes and a boolean indicating if boxes were valid.
    """
    if len(boxes) == 0:
        return boxes, False

    # Convert to numpy for easier manipulation
    boxes = boxes.numpy()

    # Ensure boxes are within image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_size[0])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_size[1])

    # Convert [x, y, width, height] to [x1, y1, x2, y2]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0]
    boxes_xyxy[:, 1] = boxes[:, 1]
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]

    # Validate boxes
    valid_boxes = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (
        boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
    boxes = boxes[valid_boxes]

    return torch.from_numpy(boxes).float(), bool(len(boxes))


def custom_collate_fn(batch):
    """
    Custom collate function with additional validation.
    """
    images = []
    targets = []

    for image, target in batch:
        # Validate boxes
        boxes, is_valid = validate_boxes(
            target['boxes'], (image.shape[2], image.shape[1]))
        if is_valid:
            target['boxes'] = boxes
            images.append(image)
            targets.append(target)

    if not images:
        # Return empty batch if no valid examples
        return None, None

    images = default_collate(images)
    return images, targets


class YOLOTrainer:
    def __init__(self, config, data_dir, output_dir):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device selection with error handling
        self.device = self._setup_device()

        self.setup_model()
        self.setup_optimizer()
        self.setup_data_loaders()
        self.setup_wandb()

    def _setup_device(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        return device

    def setup_model(self):
        # Use the latest weights
        self.model = models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT')
        self.model.to(self.device)
        logger.info("Model initialized with Faster R-CNN.")

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            epochs=self.config['num_epochs'],
            steps_per_epoch=len(self.train_loader) if hasattr(
                self, 'train_loader') else 100
        )

    def setup_data_loaders(self):
        # Initialize datasets
        self.train_dataset = KITTIMultiModalDataset(
            self.data_dir, 'train', self.config
        )
        self.val_dataset = KITTIMultiModalDataset(
            self.data_dir, 'val', self.config
        )

        # Create data loaders with validation
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

    def setup_wandb(self):
        wandb.init(
            project=self.config['wandb_project'],
            config=self.config,
            save_code=True
        )

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        valid_batches = 0

        for images, targets in tqdm(self.train_loader, desc="Training"):
            if images is None:
                continue

            try:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device)
                            for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += losses.item()
                valid_batches += 1

            except Exception as e:
                logger.error(f"Error in training batch: {str(e)}")
                continue

        return epoch_loss / max(1, valid_batches)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                if images is None:
                    continue

                try:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device)
                                for k, v in t.items()} for t in targets]

                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    val_loss += losses.item()
                    valid_batches += 1

                except Exception as e:
                    logger.error(f"Error in validation batch: {str(e)}")
                    continue

        return val_loss / max(1, valid_batches)

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.config['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.scheduler.get_last_lr()[0]
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

        self.save_model()

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def save_model(self):
        model_path = self.output_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, model_path)
        logger.info(f"Saved final model: {model_path}")
