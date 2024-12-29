import hashlib
import os
from typing import Any

import numpy as np
from PIL import Image

DATASET_DIR = "data/raw"


def classes_info() -> dict[str, int]:
    classes = os.listdir(DATASET_DIR)
    class_counts = {cl: len(os.listdir(os.path.join(DATASET_DIR, cl))) for cl in classes}
    return class_counts


def get_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_duplicates(folder_path: str) -> list[tuple[str, Any]]:
    hashes = {}
    duplicates = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        hash = get_hash(file_path)

        if hash in hashes:
            duplicates.append((file_path, hashes[hash]))
        else:
            hashes[hash] = file_path
    return duplicates


def duplicates_info() -> dict[str, int]:
    duplicates = dict()
    classes = os.listdir(DATASET_DIR)
    for cls in classes:
        path = os.path.join(DATASET_DIR, cls)
        dup = find_duplicates(path)
        duplicates[cls] = len(dup)
    return duplicates


def check_image_sizes(folder_path, cl):
    sizes = []
    for img in os.listdir(folder_path):
        image = Image.open(os.path.join(folder_path, img))
        sizes.append([cl, img, *image.size])
    return sizes


def sizes_info():
    row_list = []
    for cl in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, cl)
        row_list.extend(check_image_sizes(folder_path, cl))
    return row_list


def check_image_color(folder_path, cl):
    colors = []
    for img in os.listdir(folder_path):
        image = Image.open(os.path.join(folder_path, img))
        img_rgb = image.convert("RGB")
        np_img = np.array(img_rgb)
        np_img = np.array(img_rgb)

        r = np_img[:, :, 0].flatten()
        g = np_img[:, :, 1].flatten()
        b = np_img[:, :, 2].flatten()
        r_mean = round(np.mean(r), 2)
        g_mean = round(np.mean(g), 2)
        b_mean = round(np.mean(b), 2)
        r_std = round(np.std(r), 2)
        g_std = round(np.std(g), 2)
        b_std = round(np.std(b), 2)
        colors.append([cl, img, r_mean, g_mean, b_mean, r_std, g_std, b_std])
    return colors


def colors_info():
    row_list = []
    for cl in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, cl)
        row_list.extend(check_image_color(folder_path, cl))
    return row_list
