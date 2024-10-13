import torch
import torch.nn.utils.prune as prune
import torch.quantization
from torchvision.ops import nms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class YOLOModel:
    def __init__(self, model_name='yolov8n', num_classes=6):
        """
        Initialize the YOLO model.
        
        Args:
        model_name (str): Name of the YOLO model to use.
        num_classes (int): Number of classes to detect.
        """
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov8', model_name, pretrained=True, trust_repo=True)
        self.model.model[-1].nc = num_classes  # Set the number of classes
        self.model = torch.jit.script(self.model)  # Optimize the model using TorchScript

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
        x (torch.Tensor): Input tensor.
        
        Returns:
        torch.Tensor: Model output.
        """
        return self.model(x)

    def get_device(self):
        """
        Get the device the model is on.
        
        Returns:
        torch.device: The device (CPU or GPU) the model is on.
        """
        return self.model.device

    def apply_pruning(self, amount=0.4):
        """
        Apply pruning to the model to reduce its size.
        
        Args:
        amount (float): The amount of pruning to apply (0.0 to 1.0).
        """
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=0)
        print("Pruning applied to model.")

    def apply_quantization(self):
        """
        Apply quantization to the model to reduce its size and improve inference speed.
        """
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)
        print("Quantization applied to model.")

    def apply_distillation(self, train_loader, criterion, optimizer, teacher_model, alpha=0.5, temperature=3.0):
        """
        Apply knowledge distillation to transfer knowledge from a teacher model to this model.
        
        Args:
        train_loader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for model parameters.
        teacher_model: The teacher model to distill knowledge from.
        alpha (float): Weight for balancing student and teacher loss.
        temperature (float): Temperature for softening probability distributions.
        """
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
        """
        Initialize an ensemble of models.
        
        Args:
        models (list): List of models to ensemble.
        """
        self.models = models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, x):
        """
        Forward pass of the ensemble model.
        
        Args:
        x (torch.Tensor): Input tensor.
        
        Returns:
        torch.Tensor: Ensemble model output.
        """
        results = [model(x) for model in self.models]
        # Implement ensemble logic here (e.g., averaging, voting)
        return results[0]  # Placeholder, replace with actual ensemble logic

    def predict(self, image_path, iou_threshold=0.5, conf_threshold=0.3):
        """
        Make predictions on an image using the ensemble model.
        
        Args:
        image_path (str): Path to the input image.
        iou_threshold (float): IoU threshold for NMS.
        conf_threshold (float): Confidence threshold for detections.
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        all_boxes, all_scores, all_classes = [], [], []

        for model in self.models:
            with torch.no_grad():
                results = model(image_tensor)

            for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].cpu().numpy():
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
        """
        Visualize detection results on an image.
        
        Args:
        image (PIL.Image): Input image.
        boxes (np.array): Bounding boxes.
        scores (np.array): Confidence scores.
        classes (np.array): Class labels.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        for (xmin, ymin, xmax, ymax), score, cls in zip(boxes, scores, classes):
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin - 10, f'Class {int(cls)} {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')
        plt.show()
