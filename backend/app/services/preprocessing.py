import os
import shutil
import numpy as np
from PIL import Image, ImageOps
import io
import zipfile
import cv2
import numpy as np

TEMP_DIR = "temp"


def preprocess_image(file: bytes):
    try:
        image = Image.open(io.BytesIO(file))
        print(f"File format: {image.format}")
        if image.format not in ["JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"]:
            raise ValueError("Файл не является поддерживаемым растровым изображением.")
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Ошибка обработки файла: {e}")


def preprocess_archive(file: bytes):
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    try:
        with zipfile.ZipFile(io.BytesIO(file), 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
    except Exception as e:
        raise ValueError(f"Ошибка разархивирования: {e}")


def set_image_size(img_path: str, save_path: str, size: tuple[int, int]):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    ratio = img.width / img.height
    # Широкое изображение
    if ratio > 1:
        new_width = size[0]
        new_height = int(size[0] / ratio)
    # Высокое изображение
    else:
        new_height = size[1]
        new_width = int(size[1] * ratio)
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    img_padded = ImageOps.pad(img_resized, size, color="white", centering=(0.5, 0.5))
    img_padded.save(save_path)


def preprocess_dataset(size: tuple[int, int]):
    # Если папка уже была, то удалить из нее прошлое содержимое
    classes = os.listdir(TEMP_DIR)
    for cl in classes:
        temp_cl_path = os.path.join(TEMP_DIR, cl)
        if os.path.exists(temp_cl_path) == False:
            os.mkdir(temp_cl_path)
        folder_path = os.path.join(TEMP_DIR, cl)
        image_names = os.listdir(folder_path)
        for img_name in image_names:
            img_path = os.path.join(TEMP_DIR, cl, img_name)
            # датасет будет грузиться 1 раз для обучения, ради экономии места перезаписываем старую картинку
            set_image_size(img_path, img_path, size)


def load_colored_images_and_labels():
    images = []
    labels = []
    classes = os.listdir(TEMP_DIR)
    for class_label in classes:
        class_folder = os.path.join(TEMP_DIR, class_label)
        for file in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB
            images.append(img)
            labels.append(class_label)
    images_arr = np.array(images)
    labels_arr = np.array(labels)
    return images_arr, labels_arr
