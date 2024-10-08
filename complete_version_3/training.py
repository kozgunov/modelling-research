from torch import torch, optim
from ensemble import EnsembleModel
from pathlib import Path
import torch
import optuna
from optuna.trial import TrialState
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from model import YOLOModel
from modelling_YOLOv8.deploy import device
from HPC_integration import parallel_model_training, data_parallelism, hyperparameter_tuning_hpc
import time
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


def save_checkpoint(model, optimizer, epoch, filepath):  # save the model and optimizer state to the checkpoint file.
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, filepath)
    print("model saved successfully")


class Trainer:
    def __init__(self, model, train_loader, criterion, num_epochs, output_dir, world_size=1):
        self.model = model.get_model()
        self.device = model.get_device()
        self.train_loader = train_loader
        self.criterion = criterion
        self.world_size = world_size
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.confusion_matrices = []
        self.mAP_metric = MeanAveragePrecision()
        print(f"[DEBUG] Trainer initialized with device: {self.device}, num_epochs: {self.num_epochs}, output_dir: {self.output_dir}")

    def train(self, optimizer):
        if self.world_size > 1:
            print("[DEBUG] Using parallel model training for HPC")
            parallel_model_training(self.model, self.train_loader, self.criterion, optimizer, self.num_epochs, self.world_size) # parallel model training for HPC
        for epoch in range(self.num_epochs):
            print(f"[DEBUG] Starting epoch {epoch + 1}/{self.num_epochs}")
            self.model.train()  # model for training mode
            running_loss = 0.0
            all_preds = []
            all_labels = []
            all_boxes_pred = []
            all_boxes_true = []
            total_tp, total_fp, total_fn = 0, 0, 0
            start_time = time.time()

            for batch_idx, (images, labels, targets) in enumerate(self.train_loader):
                print(f"[DEBUG] Processing batch {batch_idx + 1}/{len(self.train_loader)}")
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                print(f"[DEBUG] Batch {batch_idx + 1} loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)  # metrics Calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                tp = ((preds == labels) & (labels == 1)).sum().item()
                fp = ((preds != labels) & (preds == 1)).sum().item()
                fn = ((preds != labels) & (labels == 1)).sum().item()
                total_tp += tp
                total_fp += fp
                total_fn += fn
                print(f"[DEBUG] Batch {batch_idx + 1} metrics - TP: {tp}, FP: {fp}, FN: {fn}")

                all_boxes_pred.append(outputs.cpu())
                all_boxes_true.append(targets.cpu())

            self.mAP_metric.update(all_boxes_pred, all_boxes_true)
            mAP_score = self.mAP_metric.compute()
            print(f'[DEBUG] mAP for epoch {epoch+1}: {mAP_score:.4f}')

            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}')
            end_time = time.time()
            detection_speed = (end_time - start_time) / len(self.train_loader)
            print(f"[DEBUG] Epoch {epoch + 1} completed with loss: {epoch_loss:.4f}")
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}')
            print(f'[DEBUG] Detection Speed: {detection_speed:.4f} seconds per batch')
            print(f'Detection Speed: {detection_speed:.4f} seconds per batch')

            precision = total_tp / (total_tp + total_fp + 1e-6)  # calculate metrics
            recall = total_tp / (total_tp + total_fn + 1e-6)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
            iou = box_iou(all_boxes_pred[0], all_boxes_true[0]).mean().item()
            print(f'[DEBUG] Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, IoU: {iou:.4f}')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, IoU: {iou:.4f}')

            cm = confusion_matrix(all_labels, all_preds)  # confusion Matrix
            self.confusion_matrices.append(cm)
            print(f'[DEBUG] Confusion Matrix for epoch {epoch + 1}: {cm}')

            # plt miracles
            plt.figure(figsize=(6, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix for Epoch {epoch + 1}')
            plt.colorbar()
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(self.output_dir / f'confusion_matrix_epoch_{epoch + 1}.png')
            plt.close()
            print(f'[DEBUG] Confusion matrix plot saved for epoch {epoch + 1}')

            # checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = epoch
                checkpoint_path = self.output_dir / f'best_checkpoint_epoch_{epoch + 1}.pth'
                save_checkpoint(self.model, optimizer, epoch, checkpoint_path)
                print(f'[DEBUG] Best checkpoint saved at epoch {epoch + 1} with loss: {epoch_loss:.4f}')
            elif epoch - self.best_epoch > 3:  # save checkpoint if divergence detected
                print("Divergence detected. Saving model checkpoint.")
                checkpoint_path = self.output_dir / f'divergence_checkpoint_epoch_{epoch + 1}.pth'
                save_checkpoint(self.model, optimizer, epoch, checkpoint_path)
                print(f'[DEBUG] Divergence checkpoint saved at epoch {epoch + 1}')

    def objective(self, trial):  # Optuna optimization
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
        print(f"[DEBUG] Suggested learning rate: {lr}")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train(optimizer)
        return self.best_loss

    def optimize(self, n_trials=10, num_nodes=1, node_rank=0): # utilize HPC for hyperparameter tuning, if  possible
        print(f"[DEBUG] Starting hyperparameter optimization with {n_trials} trials on {num_nodes} nodes")
        study = optuna.create_study(direction='minimize')
        hyperparameter_tuning_hpc(study, n_trials, num_nodes, node_rank)
        study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", study.best_params)
        print("Best loss:", study.best_value)


