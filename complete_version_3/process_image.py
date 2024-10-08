from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T
from model import YOLOModel


class ImageProcessor:
    def __init__(self, model= YOLOModel.get_model(), input_dir='D:/2nd_database_png', output_dir='C:/Users/user/pythonProject/modelling_YOLOv8/result_1'):
        self.model = model
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transforms = T.Compose([
            T.Resize((640, 640)),  # Resize to standard input size for YOLO
            T.CenterCrop(500),  # Crop to focus on the central object
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color jitter
            T.RandomRotation(15),  # Random rotation up to 15 degrees
            T.RandomHorizontalFlip(),  # Horizontal flip with 50% probability
            T.RandomVerticalFlip(),  # Vertical flip with 50% probability
            T.ToTensor(),  # Convert to Tensor (standard PyTorch format)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std for RGB
        ])

    def process_and_save_images(self):
        class_counts = {}

        images = list(self.input_dir.glob('*.png')) # iterate through all PNG images in the directory
        for image_path in images:
            image = Image.open(image_path).convert('RGB')
            results = self.model.get_model()(image)

            for _, _, _, _, _, cls in results.xyxy[0].numpy(): # count detected classes
                cls_name = self.model.get_model().names[int(cls)]
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                else:
                    class_counts[cls_name] = 1

            plt.figure(figsize=(10, 10)) # visualize results
            plt.imshow(image)
            plt.axis('off')

            for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].numpy():
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2) # draw bounding boxes and labels
                plt.gca().add_patch(rect)
                plt.gca().text(xmin, ymin, f'{self.model.get_model().names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

            plt.savefig(self.output_dir / image_path.name) # save results
            plt.close()

        # visualize class counting
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values(), color='blue')  # -------  broken here -------------
        plt.xlabel('Classes')
        plt.ylabel('Number of Detections')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.savefig(self.output_dir / "class_counts.png")
        plt.close()



    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_transformed = self.transforms(image) # Apply transformations

        return image_transformed


preprocessor = ImageProcessor()
image_tensor = preprocessor.preprocess('D:/2nd_database_png')
for i in image_tensor:
    # Load the tensor to device (e.g., CUDA) for model inference or training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension











