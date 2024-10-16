import torch
from torch import optim
from pathlib import Path
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import torch.cuda.amp as amp
import optuna
from HPC_integration import parallel_model_training, hyperparameter_tuning_hpc
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import time


def save_checkpoint(model, optimizer, epoch, filepath): # checkpoint)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, filepath)
    print(f"Model checkpoint saved successfully at {filepath}")


class Trainer:
    def __init__(self, model, train_dir, num_epochs, output_dir, world_size=1, patience=10):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_dir = Path(train_dir)
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.world_size = world_size
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.accuracy = Accuracy(num_classes=6, average='macro')
        self.precision = Precision(num_classes=6, average='macro')
        self.recall = Recall(num_classes=6, average='macro')
        self.f1_score = F1Score(num_classes=6, average='macro')
        self.confusion_matrix = ConfusionMatrix(num_classes=6)
        self.mAP = MeanAveragePrecision()
        self.scaler = amp.GradScaler()
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        print(f"Trainer initialized with device: {self.device}, num_epochs: {self.num_epochs}, output_dir: {self.output_dir}")

    def train(self):
        train_loader = self.get_data_loader()
        val_loader = self.get_data_loader(validation=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001) # 0.001 = convergent
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion) # wasn't used!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            val_loss = self.validate(val_loader, criterion)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoints(epoch, optimizer, val_loss)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        return self.model

    def validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def get_data_loader(self, validation=False):
        # implies that all the data has already uploaded!!!
        from data_loading import get_dataloader
        if validation:
            return get_dataloader(self.train_dir.parent / 'val', batch_size=32, num_workers=4)
        return get_dataloader(self.train_dir, batch_size=32, num_workers=4)

    def train_epoch(self, epoch, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, (images, labels) in enumerate(self.get_data_loader()):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            with amp.autocast():
                outputs = self.model(images)
                loss = criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        self.log_metrics(epoch, running_loss / len(self.get_data_loader()), all_preds, all_labels)

    def log_metrics(self, epoch, loss, preds, labels):
        accuracy = self.accuracy(preds, labels)
        precision = self.precision(preds, labels)
        recall = self.recall(preds, labels)
        f1 = self.f1_score(preds, labels)
        cm = self.confusion_matrix(preds, labels)
        # -----------------------------------add more complex metrics!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        
        #  confusion matrix...

    def save_checkpoints(self, epoch, optimizer, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            checkpoint_path = self.output_dir / f'best_checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(self.model, optimizer, epoch, checkpoint_path)
        elif epoch - self.best_epoch > 3:
            print("Divergence detected. Saving model checkpoint.")
            checkpoint_path = self.output_dir / f'divergence_checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(self.model, optimizer, epoch, checkpoint_path)

    def optimize(self, n_trials=10, num_nodes=1, node_rank=0):
        print(f"Starting hyperparameter optimization with {n_trials} trials on {num_nodes} nodes")
        study = optuna.create_study(direction='minimize')
        hyperparameter_tuning_hpc(study, n_trials, num_nodes, node_rank)
        study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", study.best_params)
        print("Best loss:", study.best_value)

    def objective(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch, optimizer, criterion)
        
        return self.best_loss

    def train_with_gradient_accumulation(self, accumulation_steps=4):
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001) # adamW is optimal for the task (confirmed)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(self.get_data_loader()):
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                
                running_loss += loss.item() * accumulation_steps
            
            epoch_loss = running_loss / len(self.get_data_loader())
            self.log_metrics(epoch, epoch_loss, [], [])  
