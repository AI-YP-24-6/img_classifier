import hashlib
import os
from typing import Any

import numpy as np
from PIL import Image

DATASET_DIR = "data/raw"


def check_dataset_uploaded() -> bool:
    """
    Проверка, что датасет загружен
    """
    return os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0


def classes_info() -> dict[str, int]:
    """
    Возвращает информацию о количестве изображений в каждом классе
    """
    classes = os.listdir(DATASET_DIR)
    class_counts = {cl: len(os.listdir(os.path.join(DATASET_DIR, cl))) for cl in classes}
    return class_counts


def get_hash(file_path: str) -> str:
    """
    Возвращает хеш изображения
    """
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_duplicates(folder_path: str) -> list[tuple[str, Any]]:
    """
    Проверяет дубликаты изображений по хешу
    """
    hashes = {}
    duplicates = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image_hash = get_hash(file_path)

        if image_hash in hashes:
            duplicates.append((file_path, hashes[image_hash]))
        else:
            hashes[image_hash] = file_path
    return duplicates


def duplicates_info() -> dict[str, int]:
    """
    Возвращает информацию о дубликатах в датасете
    """
    duplicates = {}
    classes = os.listdir(DATASET_DIR)
    for cls in classes:
        path = os.path.join(DATASET_DIR, cls)
        dup = find_duplicates(path)
        duplicates[cls] = len(dup)
    return duplicates


def sizes_info() -> list[list[str | Any]]:
    """
    Возвращает строки для таблицы размеров изображений
    """
    row_list = []
    for cl in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, cl)
        sizes = []
        for img in os.listdir(folder_path):
            image = Image.open(os.path.join(folder_path, img))
            sizes.append([cl, img, *image.size])
        row_list.extend(sizes)
    return row_list


def check_image_color(folder_path: str, cl: str) -> list[list[str | Any]]:
    """
    Формирование строк с информацией о цветах изображений
    """
    colors = []
    for img in os.listdir(folder_path):
        image = Image.open(os.path.join(folder_path, img))
        img_rgb = image.convert("RGB")
        np_img = np.array(img_rgb)

        r = np_img[:, :, 0].flatten()
        g = np_img[:, :, 1].flatten()
        b = np_img[:, :, 2].flatten()
        colors.append(
            [
                cl,
                img,
                round(np.mean(r), 2),
                round(np.mean(g), 2),
                round(np.mean(b), 2),
                round(np.std(r), 2),
                round(np.std(g), 2),
                round(np.std(b), 2),
            ]
        )
    return colors


def colors_info() -> list[list[str | Any]]:
    """
    Возвращает строки для таблицы цветов изображений
    """
    row_list = []
    for cl in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, cl)
        row_list.extend(check_image_color(folder_path, cl))
    return row_list
