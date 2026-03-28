import torch
import torch.nn as nn
import torch.optim as optim
from configs.config import Config
from core.evaluator import Evaluator


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )

        # 🔥 giảm LR khi plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.5
        )

        Config.setup_directories()

    def train(self):
        best_loss = float('inf')
        patience_counter = 0

        print(f"🚀 Training on {self.device}")

        for epoch in range(Config.EPOCHS):

            # =========================
            # TRAIN
            # =========================
            self.model.train()
            running_loss = 0
            correct = 0
            total = 0

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                loss.backward()

                # 🔥 tránh exploding gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                running_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct / total

            # =========================
            # VALIDATION
            # =========================
            if self.val_loader and len(self.val_loader) > 0:
                evaluator = Evaluator(self.model, self.val_loader, [], self.device)
                val_acc, val_loss = evaluator.evaluate_loss_acc(self.criterion)

                print(f"[{epoch+1}/{Config.EPOCHS}] "
                      f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

                self.scheduler.step(val_loss)

                # 🔥 SAVE BEST
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0

                    torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                    print("✅ Saved best model")
                else:
                    patience_counter += 1

                # 🔥 EARLY STOP
                if patience_counter >= Config.EARLY_STOPPING:
                    print("🛑 Early stopping triggered")
                    break

            else:
                print(f"[{epoch+1}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

        print("🎯 Training Done")