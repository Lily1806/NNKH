import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Evaluator:
    """
    Evaluates the model and computes metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
    """
    def __init__(self, model, data_loader, classes, device="cpu"):
        self.model = model
        self.data_loader = data_loader
        self.classes = classes
        self.device = torch.device(device)
        self.model.to(self.device)
        
    def evaluate_loss_acc(self, criterion):
        """Simple evaluation for loss and accuracy (used during validation)"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
        return (100 * correct / total), (running_loss / len(self.data_loader))

    def evaluate(self):
        """Full evaluation metrics for testing"""
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(y_batch.numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("==========================\n")
        
        if len(self.classes) > 0:
            self.plot_confusion_matrix(cm)
            
        return acc, precision, recall, f1, cm
        
    def plot_confusion_matrix(self, cm):
        """Saves a plot of the confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "confusion_matrix.png")
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
