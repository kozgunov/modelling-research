import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import skimage
import imageio
import tensorflow as tf
import h5py
from keras import preprocessing
from tensorflow.keras.preprocessing import image
import os


class ImagePreprocessor:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.processed_images = []

    def load_images(self): # загрузка фото
        images = []
        for file_name in os.listdir(self.image_dir):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(self.image_dir, file_name)
                image = cv2.imread(file_path)
                images.append(image)
        return images

    def preprocess_image(self, image, cv2=None): # предварительная обработка
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование в RGB
        processed_image = cv2.resize(processed_image, (224, 224))  # Изменение размера для модели
        processed_image = processed_image / 255.0  # Нормализация
        # Масштабирование и изменение размера
        # Коррекция цвета
        # Аугментация данных
        # Устранение шума и артефактов
        # Геометрические трансформации
        # Обрезка и заполнение (Padding)
        # Кластеризация и выбор областей интереса
        # Преобразования интенсивности
        return processed_image

    def process_all_images(self): # управление процессом предварительной обработки всех фото
        raw_images = self.load_images()
        self.processed_images = [self.preprocess_image(img) for img in raw_images]
        return self.processed_images

    def visualize_processed_images(self): # пример обработанной фотографии (рандомно 10 фоток)
        if not self.processed_images:
            print("Нет обработанных изображений для отображения.")
            return
        plt.figure(figsize=(10, 10))
        for i, img in enumerate(self.processed_images):
            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()



image_dir = 'C:/Users/user/pythonProject/modelling_research'
preprocessor = ImagePreprocessor(image_dir)
preprocessor.process_all_images()
preprocessor.visualize_processed_images()

images = []
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    img = image.load_img(img_path, target_size=(224, 224))  # предполагая, что вы используете этот размер
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images.append(x)

# Конвертировать список изображений в один numpy массив
data_images = np.vstack(images)

# Загрузка предварительно обработанных данных
data_np = np.load('path/to/processed/data.npy')

# Загрузка данных из HDF5 файла
with h5py.File('path/to/processed/data.h5', 'r') as hf:
    data = hf['dataset_name'][:]


