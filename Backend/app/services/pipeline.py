from typing import Any, Optional, Self

import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Resampling
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Изменение размера изображения к заданному
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    ratio = image.width / image.height
    if ratio > 1:
        new_width = size[0]
        new_height = int(size[0] / ratio)
    else:
        new_height = size[1]
        new_width = int(size[1] * ratio)
    img_resized = image.resize((new_width, new_height), Resampling.LANCZOS)
    img_padded = ImageOps.pad(img_resized, size, color="white", centering=(0.5, 0.5))
    return img_padded


def extract_hog_color_features(
    images,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    size: tuple[int, int] = (64, 64),
) -> np.array:
    """
    Изменение размера изображений к заданному и извлечение HOG-признаков из изображений
    """
    hog_features = []
    for image in images:
        img_hog_features = []
        img = Image.fromarray(image)
        img_padded = resize_image(img, size)
        resized_image = np.array(img_padded)
        for channel in cv2.split(resized_image):
            features = hog(
                channel,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm="L2-Hys",
                visualize=False,
            )
            img_hog_features.append(features)
        hog_features.append(np.hstack(img_hog_features))
    return np.array(hog_features)


class HogTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        orientations: int = 3,
        pixels_per_cell: tuple[int, int] = (10, 10),
        cells_per_block: tuple[int, int] = (2, 2),
        size: tuple[int, int] = (64, 64),
    ):
        """
        Инициализация параметров для ресайза изображений и извлечения признаков
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.size = size

    def fit(self, *_) -> Self:
        """
        Обучение ничего не делает
        """
        return self

    def transform(self, X: list[list[float]]) -> np.array:
        """
        Получение HOG-признаков из изображений
        """
        return extract_hog_color_features(
            X,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            size=self.size,
        )

    def predict(self, X: list[list[float]]) -> np.array:
        """
        Повторение метода transform для предсказаний на изображениях
        """
        return self.transform(X)


def create_model(hyperparameters: Optional[dict[str, Any]]) -> Pipeline:
    """
    Создание pipline с HOG-признаками, PCA и SVM.
    С возможностью установить гиперпараметры для PCA и SVC, используя `pca__`, `svc__`
    """
    hog_transformer = HogTransformer(orientations=3, pixels_per_cell=(10, 10), cells_per_block=(2, 2), size=(64, 64))
    if hyperparameters is not None:
        svc_dict = {}
        pca_dict = {}
        for param in hyperparameters:
            if param.startswith("svc__"):
                svc_dict[param[5:]] = hyperparameters[param]
            if param.startswith("pca__"):
                pca_dict[param[5:]] = hyperparameters[param]
        pca = PCA(**pca_dict)
        svc = SVC(**svc_dict)
        return make_pipeline(hog_transformer, pca, svc)
    pca = PCA(n_components=0.6)
    svc = SVC()
    return make_pipeline(hog_transformer, pca, svc)
