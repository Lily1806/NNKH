import torch
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from core.dataset import SignLanguageDataset
from models.model import SignLanguageModel
from core.trainer import Trainer


def main():
    print("🚀 Load Dataset...")

    try:
        train_dataset = SignLanguageDataset(Config.DATA_PROCESSED_TRAIN)
        val_dataset = SignLanguageDataset(Config.DATA_PROCESSED_PUBLIC_TEST)
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = SignLanguageModel(
        input_size=Config.KEYPOINT_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        num_classes=train_dataset.num_classes,
        model_type=Config.MODEL_TYPE
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = Trainer(model, train_loader, val_loader, device=device)
    trainer.train()


if __name__ == "__main__":
    main()