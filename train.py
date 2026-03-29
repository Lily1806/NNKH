import os
import torch
import torch.nn as nn
import argparse

from configs.config import Config
from core.dataset import get_dataloaders
from models.model import BiLSTMAttention
from core.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Nhận diện Ngôn ngữ Ký hiệu (Sign Language)")
    # Defaults path to data raw or processed that the user mapped
    parser.add_argument("--data_dir", type=str, default=Config.DATA_RAW_DIR, help="Đường dẫn đến thư mục chứa các thư mục con tương ứng với class")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Số epochs")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Learning rate (Tốc độ học)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Tỉ lệ chia dữ liệu train/val")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # 1. Khởi tạo folder
    Config.setup_directories()
    
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # 3. Load Dataloader
    print(f"⏳ Tải dữ liệu từ {args.data_dir}...")
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_frames=Config.MAX_FRAMES,
        keypoint_dim=Config.KEYPOINT_DIM,
        val_split=args.val_split,
        seed=args.seed
    )
    
    # 4. Khởi tạo Model
    print(f"🏗️  Khởi tạo mô hình định dạng Attention + BiLSTM...")
    model = BiLSTMAttention(
        input_size=Config.KEYPOINT_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        num_classes=len(classes),
        dropout=Config.DROPOUT
    ).to(device)
    
    # 5. Cấu hình Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    
    # AdamW kết hợp Weight Decay giúp tránh overfit mạnh hơn Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    
    # Cosine Annealing giảm LR cong đều theo hàm cos sau khi hội tụ dần
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 6. Trainer Initialize
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=Config.EARLY_STOPPING,  # số epochs ko cải thiện thì dừng
        save_dir=Config.MODEL_DIR,
        classes=classes
    )
    
    # 7. Start Training !
    trainer.fit(epochs=args.epochs)
    
    print("\n🎉 Hoàn thành quá trình đào tạo mô hình.")

if __name__ == "__main__":
    main()