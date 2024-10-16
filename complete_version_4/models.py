import torch
import torch.nn.utils.prune as prune
import torch.quantization
from torchvision.ops import nms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class YOLOModel:
    def __init__(self, model_name='yolov8n', num_classes=6):
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov8', model_name, pretrained=True, trust_repo=True) # initialize the YOLO
        self.model.model[-1].nc = num_classes  # Set the number of classes
        self.model = torch.jit.script(self.model)  # Optimize the model using TorchScript

    def forward(self, x):
        return self.model(x)

    def get_device(self):
        return self.model.device # has to be GPU

    def apply_pruning(self, amount=0.4):
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=0)
        print("Pruning applied to model.")

    def apply_quantization(self):
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)
        print("Quantization applied to model.")


    def apply_distillation(self, train_loader, criterion, optimizer, teacher_model, alpha=0.5, temperature=3.0): # may be extra for unsupervise learning
        self.model.train()
        teacher_model.eval()
        for images, labels in train_loader:
            images, labels = images.to(self.model.device), labels.to(self.model.device)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = self.model(images)
            soft_labels = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
            soft_student_outputs = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)
            distillation_loss = torch.nn.functional.kl_div(soft_student_outputs, soft_labels, reduction='batchmean') * (temperature ** 2)
            hard_loss = criterion(student_outputs, labels)
            loss = alpha * distillation_loss + (1 - alpha) * hard_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Distillation training completed.")

class EnsembleModel:
    def __init__(self, models):
        self.models = models # ensemble models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, x):
        results = [model(x) for model in self.models]
        # -----------------apply ensemble logic like averaging, voting -------------------------------------------------
        return results[0]  

    def predict(self, image_path, iou_threshold=0.5, conf_threshold=0.3): # predicts on an image using the ensemble model.
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        all_boxes, all_scores, all_classes = [], [], []

        for model in self.models:
            with torch.no_grad():
                results = model(image_tensor)

            for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].cpu().numpy(): # the box
                if conf > conf_threshold:
                    all_boxes.append([xmin, ymin, xmax, ymax])
                    all_scores.append(conf)
                    all_classes.append(cls)

        all_boxes = torch.tensor(all_boxes, dtype=torch.float32).to(self.device)
        all_scores = torch.tensor(all_scores, dtype=torch.float32).to(self.device)
        all_classes = torch.tensor(all_classes, dtype=torch.int64).to(self.device)

        keep_indices = nms(all_boxes, all_scores, iou_threshold)
        final_boxes = all_boxes[keep_indices].cpu().numpy()
        final_scores = all_scores[keep_indices].cpu().numpy()
        final_classes = all_classes[keep_indices].cpu().numpy()

        self.visualize_results(image, final_boxes, final_scores, final_classes)

    def visualize_results(self, image, boxes, scores, classes): 
        # vsualize detection results on an image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        for (xmin, ymin, xmax, ymax), score, cls in zip(boxes, scores, classes):
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin - 10, f'Class {int(cls)} {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')
        plt.show()
