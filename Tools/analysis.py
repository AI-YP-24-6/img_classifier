import json
import os
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


<<<<<<< HEAD
def find_image_files(directory: Path) -> list[str]:
=======
<<<<<<< HEAD
def find_image_files(path: Path) -> list[str]:
>>>>>>> feature-fastapi
    """
    Получение списка путей к jpeg-файлам в нужной директории
    :param directory: Путь к папке поиска
    :return: Список путей к jpeg-файлам
    """
    jpeg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
<<<<<<< HEAD
            if file.endswith(("jpg", "JPG", "jpeg", "tif")):
=======
            if file.endswith((".jpg", "JPG", "jpeg", "tif")):
=======
def find_image_files(directory: Path) -> list[str]:
    """
    Получение списка путей к jpeg-файлам в нужной директории
    :param directory: Путь к папке поиска
    :return: Список путей к jpeg-файлам
    """
    jpeg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(("jpg", "JPG", "jpeg", "tif")):
>>>>>>> 3d25aabb8347edce5a9e182f4e9b2083f9459fc4
>>>>>>> feature-fastapi
                full_path = os.path.join(root, file)
                jpeg_files.append(full_path)
    return jpeg_files


<<<<<<< HEAD
def find_needed_jpeg_files(directory: Path, target_names: list[str | int]) -> list[str]:
=======
<<<<<<< HEAD
def find_needed_jpeg_files(path: Path, names: list[str | int]) -> list[str]:
>>>>>>> feature-fastapi
    """
    Поиск нужных jpeg-файлов в нужной директории по их id имени
    :param directory: Путь к папке поиска
    :param target_names: Список имен для поиска
    :return: Список путей к jpeg-файлам
    """
    if not isinstance(target_names[0], str):
        target_names = [str(name) for name in target_names]
    jpeg_files = find_image_files(directory)
    needed_jpeg_files = []
    for jpeg_file in jpeg_files:
        for name in target_names:
            if name in jpeg_file:
                needed_jpeg_files.append(jpeg_file)
<<<<<<< HEAD

=======
=======
def find_needed_jpeg_files(directory: Path, target_names: list[str | int]) -> list[str]:
    """
    Поиск нужных jpeg-файлов в нужной директории по их id имени
    :param directory: Путь к папке поиска
    :param target_names: Список имен для поиска
    :return: Список путей к jpeg-файлам
    """
    if not isinstance(target_names[0], str):
        target_names = [str(name) for name in target_names]
    jpeg_files = find_image_files(directory)
    needed_jpeg_files = []
    for jpeg_file in jpeg_files:
        for name in target_names:
            if name in jpeg_file:
                needed_jpeg_files.append(jpeg_file)

>>>>>>> 3d25aabb8347edce5a9e182f4e9b2083f9459fc4
>>>>>>> feature-fastapi
    return needed_jpeg_files


def display_images(image_paths: list[str], subtitle: str = "", images_per_row: int = 5) -> None:
    """
    Показывает изображения в графическом окне
    :param images_per_row: Сколько изображений на одной строчке
    :param image_paths: Список путей к изображениям
    :param subtitle: Подзаголовок
    :return: None
    """
    image_count = len(image_paths)

    for start in range(0, image_count, images_per_row):
        end = min(start + images_per_row, image_count)
        fig, axs = plt.subplots(1, end - start, figsize=(15, 5))
        fig.suptitle(subtitle)

        axs = axs if isinstance(axs, Iterable) else [axs]
        for i, ax in enumerate(axs):
            img_path = image_paths[start + i]
            img_name = get_image_json_info(img_path.split(".")[0] + ".json").get("name", "")
            img_name = (img_name[:27] + "...") if img_name and len(img_name) > 30 else img_name

            ax.set_title(img_name)
            ax.imshow(Image.open(img_path))
            ax.axis("off")
        plt.show()


def get_image_json_info(path: str) -> dict:
    """
    Извлечение информации об изображении из JSON-файла
    :param path: Путь к изображению
    :return: Словарь с информацией о JSON-файле
    """
    try:
        with open(path, encoding="ascii") as file:
            return json.load(file)
    except (UnicodeDecodeError, FileNotFoundError):
        return {}
