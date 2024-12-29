import os
import random
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = "temp"
PREVIEW_DIR = "preview"
PREVIEW_PATH = os.path.join(PREVIEW_DIR, "preview.png")


def plot_images(num_images=5):
    classes = os.listdir(DATASET_DIR)
    fig, axs = plt.subplots(len(classes), num_images, figsize=(15, 5 * len(classes)))
    if len(classes) == 1:
        axs = [axs]
    for row, cl in enumerate(classes):
        folder_path = os.path.join(DATASET_DIR, cl)
        image_files = random.sample(os.listdir(folder_path), min(num_images, len(os.listdir(folder_path))))
        for col, img_name in enumerate(image_files):
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path)
            ax = axs[row][col] if len(classes) > 1 else axs[0][col]
            ax.imshow(img)
            ax.set_title(cl)
            ax.axis("off")
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    if os.path.exists(PREVIEW_DIR) == False:
        os.mkdir(PREVIEW_DIR)
    plt.savefig(PREVIEW_PATH, format="png")
    buffer.seek(0)
    plt.close()
    return buffer


def remove_preview():
    if os.path.exists(PREVIEW_PATH):
        os.remove(PREVIEW_PATH)


def preview_dataset(num_images: int):
    if os.path.exists(PREVIEW_PATH):
        with open(PREVIEW_PATH, "rb") as f:
            buffer = BytesIO(f.read())
            return buffer
    else:
        return plot_images(num_images=num_images)
