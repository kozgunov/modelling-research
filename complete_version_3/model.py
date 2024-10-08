import torch
import torch.nn.utils.prune as prune
import torch.quantization

from modelling_YOLOv8.deploy import device
from modelling_YOLOv8.ensemble import EnsembleModel


class YOLOModel:
    def __init__(self, model_name='yolov8n', device=None): # installation of YOLOv8n indeed
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov8', model_name, pretrained=True, trust_repo=True)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device

    def apply_pruning(self, amount=0.4): # pruning function
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=0)  # prune 40% of the weakest ones
        print("Pruning applied to model.")

    def apply_quantization(self): # quantization function
        self.model.eval()  # model for evaluation mode
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)
        print("Quantization applied to model.")

    def apply_distillation(self, train_loader, criterion, optimizer, teacher_model=EnsembleModel(model_paths=['yolov8n']).to(device), alpha=0.5, temperature=3.0): # distillation function
        self.model.train()
        teacher_model.eval()
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = self.model(images) # forward pass through student
            soft_labels = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1) # compute distillation loss
            soft_student_outputs = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)
            # loss function that combines student loss with teacher-student similarity
            distillation_loss = torch.nn.functional.kl_div(soft_student_outputs, soft_labels, reduction='batchmean') * (temperature ** 2)
            hard_loss = criterion(student_outputs, labels)
            loss = alpha * distillation_loss + (1 - alpha) * hard_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Distillation training completed.")


