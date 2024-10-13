from PIL import Image
import os
import random


def resize_images(input_folder, output_folder, target_size=(640, 640), zoom_range=(1.0, 1.2), scale_factor=0.5):
    #if not os.path.exists(output_folder): # делаем папку для вывода, если она не указана
    #    os.makedirs(output_folder)

    for filename in os.listdir(input_folder): # проходимся по  всем фоткам в папке
        img_path = os.path.join(input_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(img_path)

            img_resized = img.resize(target_size, Image.LANCZOS) # размер меняем в стандартный

            zoom_factor = random.uniform(zoom_range[0], zoom_range[1]) # случайное увеличение
            width, height = img_resized.size
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            img_zoomed = img_resized.resize((new_width, new_height), Image.LANCZOS)

            x_center = new_width // 2
            y_center = new_height // 2
            left = max(0, x_center - width // 2)
            top = max(0, y_center - height // 2)
            img_cropped = img_zoomed.crop((left, top, left + width, top + height)) # обрезаем увеличенное изображение до целевого размера

            output_filename = os.path.splitext(filename)[0] + ".png"  # + расширение
            output_path = os.path.join(output_folder, output_filename)
            img_cropped.save(output_path, 'PNG')  # сохраняем в png


input_folder = 'C:/Users/user/pythonProject/modelling_research/database'
output_folder = 'C:/Users/user\pythonProject/modelling_research/2nd_database_jpeg_cropped'
resize_images(input_folder, output_folder)
