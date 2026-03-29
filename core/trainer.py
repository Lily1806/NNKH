import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, top_k_accuracy_score
import numpy as np
import seaborn as sns

class Trainer:
    """
    Lớp Trainer đóng gói vòng lặp training, evaluate, visualization.
    Đảm bảo Code sạch, dễ bảo trì, và không bị lỗi dependency.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience=5, save_dir="models/saved_models", classes=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.save_dir = save_dir
        self.classes = classes or []
        
        os.makedirs(save_dir, exist_ok=True)
        self.best_model_path = os.path.join(save_dir, "best_model.pth")
        
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            loss.backward()
            
            # Gradient clipping (phòng exploding gradients cho LSTM)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        acc = accuracy_score(all_targets, all_preds)
        return total_loss / len(self.train_loader), acc
        
    def evaluate(self, return_metrics=False):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_targets, all_preds)
        
        if return_metrics:
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
            
            # Top K accuracy
            try:
                top_k = min(3, len(self.classes))
                # Phải mapping index labels phù hợp với all_probs cho top_k metric
                top_k_acc = top_k_accuracy_score(all_targets, all_probs, k=top_k, labels=np.arange(len(self.classes)))
            except Exception as e:
                top_k_acc = None
                
            cm = confusion_matrix(all_targets, all_preds)
            
            metrics = {
                "macro": {"precision": precision_macro, "recall": recall_macro, "f1": f1_macro},
                "weighted": {"precision": precision_weighted, "recall": recall_weighted, "f1": f1_weighted},
                "top_k_acc": top_k_acc,
                "cm": cm
            }
            return avg_loss, acc, metrics
            
        return avg_loss, acc
        
    def fit(self, epochs):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        print("\n" + "="*50)
        print(f"🚀 Bắt đầu Training {epochs} epochs trên thiết bị {self.device}")
        print("="*50 + "\n")
        
        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            # Cập nhật learning rate (nếu dùng scheduler plateau)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
                
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            print(f"📉 Train Loss: {train_loss:.4f} | 🎯 Train Acc: {train_acc:.4f}")
            print(f"📉 Val Loss:   {val_loss:.4f} | 🎯 Val Acc:   {val_acc:.4f}")
            
            # Early stopping & save best model (ưu tiên val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'classes': self.classes
                }, self.best_model_path)
                print(f"🔥 Lưu best model vòng epoch {epoch+1}!")
            else:
                epochs_no_improve += 1
                print(f"⚠️ Val loss không cải thiện ({epochs_no_improve}/{self.patience})")
                
                if epochs_no_improve >= self.patience:
                    print(f"🛑 EARLY STOPPING KÍCH HOẠT TẠI EPOCH {epoch+1}!")
                    break
            print("-" * 50)
            
        # 📌 CUỐI CÙNG LÀ FINAL EVALUATION VÀ VẼ BẢN ĐỒ
        # Load best model đã lưu để tính
        self.model.load_state_dict(torch.load(self.best_model_path)['model_state_dict'])
        print("\n🏆 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALIDATION TỪ BEST MODEL:")
        _, eval_acc, metrics_dict = self.evaluate(return_metrics=True)
        
        print(f"Accuracy: {eval_acc:.4f}")
        print(f"Macro F1-score: {metrics_dict['macro']['f1']:.4f}")
        print(f"Weighted F1-score: {metrics_dict['weighted']['f1']:.4f}")
        if metrics_dict['top_k_acc'] is not None:
             print(f"Top-K Accuracy (K={min(3, len(self.classes))}): {metrics_dict['top_k_acc']:.4f}")
             
        # Visualize
        print(f"\n🎨 Đang kết xuất biểu đồ kết quả vào: {self.save_dir}")
        self.plot_history()
        self.plot_confusion_matrix(metrics_dict['cm'])
        
    def plot_history(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label='Train Loss')
        plt.plot(self.history["val_loss"], label='Val Loss')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label='Train Acc')
        plt.plot(self.history["val_acc"], label='Val Acc')
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()
        
    def plot_confusion_matrix(self, cm):
        # Tránh lỗi trục tọa độ dài
        plt.figure(figsize=(max(8, len(self.classes)*0.5), max(6, len(self.classes)*0.5)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()