import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self, model):
        self.model = model
        self.device = model.get_device()

    def analyze_image(self, image_path): # analyze single images using the model & output detections with bounding boxes

        image = Image.open(image_path).convert('RGB') # load and preprocess the image
        image_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            results = self.model.get_model()(image_tensor) # get predictions from model

        plt.figure(figsize=(10, 10)) # visualize results
        plt.imshow(image)
        plt.axis('off')

        for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].cpu().numpy(): # draw bounding boxes and labels
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin - 10, f'{self.model.get_model().names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')
        plt.show()
