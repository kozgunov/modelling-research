import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as T
import cv2
from data_loading import ShipDataset

class ImageProcessor:
    def __init__(self, model, input_dir='input', output_dir='output'):
        self.model = model
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_and_save_images(self):
        class_counts = {}
        images = list(self.input_dir.glob('*.png'))
        for image_path in images:
            image = Image.open(image_path).convert('RGB')
            results = self.model.get_model()(image)

            for _, _, _, _, _, cls in results.xyxy[0].numpy():
                cls_name = self.model.get_model().names[int(cls)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            self.visualize_results(image, results, image_path.name)

        self.plot_class_distribution(class_counts)

    def visualize_results(self, image, results, filename):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')

        for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].numpy():
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin, f'{self.model.get_model().names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_class_distribution(self, class_counts):
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values(), color='blue')
        plt.xlabel('Classes')
        plt.ylabel('Number of Detections')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.savefig(self.output_dir / "class_counts.png")
        plt.close()

    def preprocess(self, image_path):
        dataset = ShipDataset(image_path, transform=self.transforms)
        return dataset[0][0]  # Return the first item's image

    def analyze_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.model.get_device())

        with torch.no_grad():
            results = self.model.get_model()(image_tensor)

        self.visualize_results(image, results, f"analyzed_{Path(image_path).name}")

        plt.figure(figsize=(10, 10)) # visualize results
        plt.imshow(image)
        plt.axis('off')

        for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].cpu().numpy(): # draw bounding boxes and labels
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.gca().text(xmin, ymin - 10, f'{self.model.get_model().names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')
        plt.show()

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transforms(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.model.get_device())
        
        with torch.no_grad():
            results = self.model(input_tensor)
        
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            detections.append({
                'bbox': xyxy,
                'class': int(cls),
                'confidence': float(conf)
            })
        
        return detections

    def detect_colors(self, frame, tracked_objects):
        colors = []
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            roi = frame[y1:y2, x1:x2]
            avg_color = np.mean(roi, axis=(0, 1))
            colors.append(avg_color)
        return colors

    def annotate_frame(self, frame, tracked_objects):
        annotated_frame = frame.copy()
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            color = obj.get('color', (0, 255, 0))  # Default to green if color not set
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Ship {obj['id']}: {obj['class']}"
            if obj.get('occluded', False):
                label += " (Occluded)"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return annotated_frame
