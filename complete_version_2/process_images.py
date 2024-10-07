from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

class ImageProcessor:
    def __init__(self, model, input_dir, output_dir):
        self.model = model
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_and_save_images(self):
        class_counts = {}

        # Iterate through all PNG images in the directory
        images = list(self.input_dir.glob('*.png'))
        for image_path in images:
            image = Image.open(image_path).convert('RGB')
            results = self.model.get_model()(image)

            # Count detected classes
            for _, _, _, _, _, cls in results.xyxy[0].numpy():
                cls_name = self.model.get_model().names[int(cls)]
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                else:
                    class_counts[cls_name] = 1

            # Visualize results
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')

            # Draw bounding boxes and labels
            for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].numpy():
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
                plt.gca().add_patch(rect)
                plt.gca().text(xmin, ymin, f'{self.model.get_model().names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

            # Save results
            plt.savefig(self.output_dir / image_path.name)
            plt.close()

        # Visualize class counts
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values(), color='blue')
        plt.xlabel('Classes')
        plt.ylabel('Number of Detections')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.savefig(self.output_dir / "class_counts.png")
        plt.close()
