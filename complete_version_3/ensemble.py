import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import nms


class EnsembleModel:
    def __init__(self, model_names, device='cuda'):
        self.models = [torch.hub.load('ultralytics/yolov8', model_name, pretrained=True).to(device) for model_name in model_names] # ensemble approach
        self.device = device

    def predict(self, image_path, iou_threshold=0.5, conf_threshold=0.3):
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        all_boxes = []
        all_scores = []
        all_classes = []

        # Get predictions from each model
        for model in self.models:
            with torch.no_grad():
                results = model(image_tensor)

            for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > conf_threshold:
                    all_boxes.append([xmin, ymin, xmax, ymax])
                    all_scores.append(conf)
                    all_classes.append(cls)

        # Convert lists to tensors
        all_boxes = torch.tensor(all_boxes, dtype=torch.float32).to(self.device)
        all_scores = torch.tensor(all_scores, dtype=torch.float32).to(self.device)
        all_classes = torch.tensor(all_classes, dtype=torch.int64).to(self.device)

        # Apply Non-Maximum Suppression to remove overlapping boxes
        keep_indices = nms(all_boxes, all_scores, iou_threshold)
        final_boxes = all_boxes[keep_indices].cpu().numpy()
        final_scores = all_scores[keep_indices].cpu().numpy()
        final_classes = all_classes[keep_indices].cpu().numpy()

        # Draw bounding boxes on the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        for (xmin, ymin, xmax, ymax), score, cls in zip(final_boxes, final_scores, final_classes):
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin - 10, f'Class {int(cls)} {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

        plt.show()
