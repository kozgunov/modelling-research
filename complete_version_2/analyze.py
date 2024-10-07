import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, model):
        self.model = model
        self.device = model.get_device()

    def analyze_image(self, image_path):
        """
        Analyze a single image using the model and output the detections with bounding boxes.
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Get predictions from model
        with torch.no_grad():
            results = self.model.get_model()(image_tensor)

        # Visualize results
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')

        # Draw bounding boxes and labels
        for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].cpu().numpy():
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin - 10, f'{self.model.get_model().names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

        plt.show()
