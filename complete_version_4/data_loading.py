import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np

class ShipDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.glob('*.png'))
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labels = self._load_labels()

    def _load_labels(self):
        label_file = self.root_dir.parent / 'labels.txt'
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split(',')
                labels[img_name] = int(label)
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[img_path.name]
        return image, label

def get_dataloader(data_dir, batch_size=32, num_workers=4):
    dataset = ShipDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
