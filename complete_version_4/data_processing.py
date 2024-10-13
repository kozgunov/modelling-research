import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

def organize_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    (output_path / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'test').mkdir(parents=True, exist_ok=True)

    # Get all PNG files
    all_files = list(input_path.glob('*.png'))

    # Split the dataset
    train_files, test_files = train_test_split(all_files, test_size=(1 - train_ratio), random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(file, output_path / 'train' / file.name)
    for file in val_files:
        shutil.copy(file, output_path / 'val' / file.name)
    for file in test_files:
        shutil.copy(file, output_path / 'test' / file.name)

    print(f"Dataset organized: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test images.")

def preprocess_images(input_dir, output_dir, target_size=(640, 640)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_path in input_path.glob('*.png'):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Save preprocessed image
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), cv2.cvtColor((img_normalized * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print(f"Preprocessing complete. All images resized to {target_size} and normalized.")
