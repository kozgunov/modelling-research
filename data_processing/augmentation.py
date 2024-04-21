from PIL import Image, ImageEnhance, ImageOps
import os
import random
import numpy as np


def rotate_image(image, max_angle, probability):
    if random.random() < probability:
        angle = random.uniform(-max_angle, max_angle)
        return image.rotate(angle, expand=True)
    return image


def shift_image(image, max_shift, probability):
    if random.random() < probability:
        x_shift = random.randint(-max_shift, max_shift)
        y_shift = random.randint(-max_shift, max_shift)
        return image.transform(image.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))
    return image


def change_saturation(image, min_factor, max_factor, probability):
    if random.random() < probability:
        factor = random.uniform(min_factor, max_factor)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    return image


def augment_image(image_path, output_path, augmentation_functions):
    image = Image.open(image_path)
    for augment in augmentation_functions:
        image = augment(image)
    image.save(output_path)


def change_hue(image, delta, probability):
    if random.random() < probability:
        image = image.convert('HSV')
        np_image = np.array(image)
        np_image[:,:,0] = (np_image[:,:,0] + delta) % 180
        image = Image.fromarray(np_image, 'HSV')
        return image.convert('RGB')
    return image


def change_brightness(image, min_factor, max_factor, probability):
    if random.random() < probability:
        factor = random.uniform(min_factor, max_factor)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    return image


input_folder = 'C:/Users/user/pythonProject/modelling_research/***'
output_folder = 'C:/Users/user/pythonProject/modelling_research/***'
max_angle = 25  # MAX угол поворота
max_shift = 20  # MAX смещение в пикселях
min_saturation = 0.5  # MIN коэффициент насыщенности
max_saturation = 1.5  # MAX коэффициент насыщенности
hue_delta = 30  # сдвиг оттенка
min_brightness = 0.7  # MIN коэффициент яркости
max_brightness = 1.3  # MAX коэффициент яркости



augmentation_settings = [
    # методы аугментации непоследовательно (с вероятностями применения к каждому фото)
    (lambda img, prob: rotate_image(img, max_angle=25, probability=prob), 0.5),
    (lambda img, prob: shift_image(img, max_shift=20, probability=prob), 0.5),
    (lambda img, prob: change_saturation(img, min_factor=0.5, max_factor=1.5, probability=prob), 0.7),
    (lambda img, prob: change_hue(img, delta=30, probability=prob), 0.3),
    (lambda img, prob: change_brightness(img, min_factor=0.7, max_factor=1.3, probability=prob), 0.6)
]

if not os.path.exists(output_folder): # создать папку для вывода, если ее нет
    os.makedirs(output_folder)


for filename in os.listdir(input_folder): # main по каждому изображению (из ... - в ...)
    file_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    augment_image(file_path, output_path, augmentation_settings)
