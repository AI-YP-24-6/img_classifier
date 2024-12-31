import io
import os
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image

from Backend.app.services.pipeline import resize_image

TEMP_DIR = "data/temp"
DATASET_DIR = "data/raw"


def preprocess_image(file: bytes) -> np.ndarray:
    """
    Обработка изображения для подачи на предсказание
    """
    try:
        image = Image.open(io.BytesIO(file))
        print(f"File format: {image.format}")
        if image.format not in ["JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"]:
            raise ValueError("Файл не является поддерживаемым растровым изображением.")
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Ошибка обработки файла: {e}") from e


def preprocess_archive(file: bytes) -> None:
    """
    Загрузка датасета на хранение
    """
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    # Если был обработанный датасет, то тоже удалить его, т.к. будет новый
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    try:
        with zipfile.ZipFile(io.BytesIO(file), "r") as zip_ref:
            zip_ref.extractall(DATASET_DIR)
    except Exception as e:
        raise ValueError(f"Ошибка разархивирования: {e}") from e


def preprocess_dataset(size: tuple[int, int]) -> None:
    """
    Обработка датасета с приведением изображений к единому размеру
    """
    if os.path.exists(TEMP_DIR) is False:
        os.mkdir(TEMP_DIR)
    if len(os.listdir(TEMP_DIR)) == 0:
        classes = os.listdir(DATASET_DIR)
        for cl in classes:
            temp_cl_path = os.path.join(TEMP_DIR, cl)
            if os.path.exists(temp_cl_path) is False:
                os.mkdir(temp_cl_path)
            folder_path = os.path.join(DATASET_DIR, cl)
            image_names = os.listdir(folder_path)
            for img_name in image_names:
                img_path = os.path.join(DATASET_DIR, cl, img_name)
                save_path = os.path.join(TEMP_DIR, cl, img_name)
                # датасет будет грузиться 1 раз для обучения, ради экономии места перезаписываем старую картинку
                img = Image.open(img_path)
                img_padded = resize_image(img, size)
                img_padded.save(save_path)


def load_colored_images_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразование обработанного датасета для подачи в модель
    """
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
