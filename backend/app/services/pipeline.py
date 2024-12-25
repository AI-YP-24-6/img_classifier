import cv2
from PIL import Image, ImageOps
from PIL.Image import Resampling
from skimage.feature import hog

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from typing import List, Union

import numpy as np

def resize_image(image, size:tuple[int, int]) -> np.array:
    img = Image.fromarray(image)
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
    img_resized = img.resize((new_width, new_height), Resampling.LANCZOS)
    img_padded = ImageOps.pad(img_resized, size, color="white", centering=(0.5, 0.5))
    return np.array(img_padded)

def extract_hog_color_features(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), size=(64, 64)) -> np.array:
    hog_features = []
    for image in images:
        img_hog_features = []
        resized_image = resize_image(image, size)
        for channel in cv2.split(resized_image):
            features = hog(
                channel,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm='L2-Hys',
                visualize=False
            )
            img_hog_features.append(features)
        hog_features.append(np.hstack(img_hog_features))
    return np.array(hog_features)

class HogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=3, pixels_per_cell=(10, 10), cells_per_block=(2, 2), size=(64, 64)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.size = size

    def fit(self, X:List[List[float]], y:Union[List[str], None]=None):
        return self

    def transform(self, X:List[List[float]]) -> np.array:
        return extract_hog_color_features(
            X,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            size=self.size
            )

    def predict(self, X:List[List[float]]) -> np.array:
        return self.transform(X)


def create_model() -> Pipeline:
    hog_transformer = HogTransformer(orientations=3, pixels_per_cell=(10, 10), cells_per_block=(2, 2), size=(64, 64))
    pca = PCA(n_components=0.6)
    svc = SVC()
    return make_pipeline(hog_transformer, pca, svc)
