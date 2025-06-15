---
hide:
  - navigation
---

# Использование HOG для классификации изображений

## Краткое описание
- В данном ноутбуке исследуется применение дескриптора HOG (Histogram of Oriented Gradients) для задачи классификации изображений.
- Проводится сравнение нескольких моделей машинного обучения: SVC, RandomForest, LightGBM и CatBoost.
- Для уменьшения размерности признакового пространства используется метод главных компонент (PCA).
- Выполняется подбор гиперпараметров для каждой модели с целью улучшения метрик качества.

## Содержание
- [Шаг 1: Установка зависимостей](#шаг-1-установка-зависимостей)
- [Шаг 2: Импорт библиотек](#шаг-2-импорт-библиотек)
- [Шаг 3: Настройка окружения и путей](#шаг-3-настройка-окружения-и-путей)
- [Шаг 4: Загрузка и распаковка данных](#шаг-4-загрузка-и-распаковка-данных)
- [Шаг 5: Подготовка функций для извлечения HOG-признаков](#шаг-5-подготовка-функций-для-извлечения-hog-признаков)
- [Шаг 6: Извлечение и загрузка HOG-признаков](#шаг-6-извлечение-и-загрузка-hog-признаков-из-изображений)
- [Шаг 7: Обучение и оценка SVC](#шаг-7-обучение-и-оценка-svc-с-использованием-pca)
- [Шаг 8: Обучение и оценка RandomForest](#шаг-8-обучение-и-оценка-randomforest-classifier-с-pca)
- [Шаг 9: Обучение и оценка LightGBM](#шаг-9-обучение-и-оценка-lightgbm-classifier-с-pca)
- [Шаг 10: Обучение и оценка CatBoost](#шаг-10-обучение-и-оценка-catboost-classifier-с-pca)
- [Ключевые результаты](#ключевые-результаты)

### Шаг 1: Установка зависимостей {#шаг-1-установка-зависимостей}
*Цель шага*: Установка необходимых библиотек `catboost` и `lightgbm`.

```python
!pip install catboost -q
!pip install lightgbm -q
```

**Результат**:
> Библиотеки успешно установлены.

### Шаг 2: Импорт библиотек  {#шаг-2-импорт-библиотек}
*Цель шага*: Импорт всех необходимых модулей для обработки данных, извлечения признаков и машинного обучения.

```python
import zipfile
import os
import shutil
import random
import gdown

import cv2
from PIL import Image, ImageOps

from skimage.feature import hog

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import lightgbm as lgb
import scipy.stats as st

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from tqdm.auto import tqdm
```

### Шаг 3: Настройка окружения и путей {#шаг-3-настройка-окружения-и-путей}
*Цель шага*: Инициализация констант и определение путей к данным.

```python
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

try:
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_DIR = os.path.join("/content/drive", "MyDrive")
except ImportError:
    DRIVE_DIR = os.getcwd()

DATASET_DIR = os.path.join(os.getcwd(), "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

TEMP_DIR = os.path.join(os.getcwd(), "temp")
TEMP_TRAIN_DIR = os.path.join(TEMP_DIR, "train")
TEMP_TEST_DIR = os.path.join(TEMP_DIR, "test")

ZIP_PATH = os.path.join(DRIVE_DIR, "dataset_32_classes_splitted.zip")
os.makedirs(DATASET_DIR, exist_ok=True)
```

### Шаг 4: Загрузка и распаковка данных {#шаг-4-загрузка-и-распаковка-данных}
*Цель шага*: Загрузка датасета с Google Drive и его распаковка.

```python
file_id = "1-1ehpRd0TnwB1hTHQbFHzdf55SrIri4f"
if os.path.exists(ZIP_PATH):
    print("Архив уже добавлен")
else:
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", os.path.join(os.getcwd(), "dataset_32_classes.zip"), quiet=False
    )

# Распаковка архива
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall("./dataset")
```
**Результат**:
> Архив с изображениями загружен и распакован в папку `dataset`.

### Шаг 5: Подготовка функций для извлечения HOG-признаков {#шаг-5-подготовка-функций-для-извлечения-hog-признаков}
*Цель шага*: Определение функций для извлечения HOG-признаков из изображений.

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block):
    features = hog(img, orientations=orient,
                   pixels_per_cell=pix_per_cell,
                   cells_per_block=cell_per_block,
                   transform_sqrt=True,
                   feature_vector=True)
    return features

def upload_colored_hog_features(color_space_code, pix_per_cell, cell_per_block):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for class_name in tqdm(os.listdir(TRAIN_DIR)):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, color_space_code)

            hog_features = []
            for channel in range(img.shape[2]):
                features = get_hog_features(img[:,:,channel], 9, pix_per_cell, cell_per_block)
                hog_features.extend(features)

            X_train.append(hog_features)
            y_train.append(class_name)

    for class_name in tqdm(os.listdir(TEST_DIR)):
        class_dir = os.path.join(TEST_DIR, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, color_space_code)

            hog_features = []
            for channel in range(img.shape[2]):
                features = get_hog_features(img[:,:,channel], 9, pix_per_cell, cell_per_block)
                hog_features.extend(features)

            X_test.append(hog_features)
            y_test.append(class_name)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
```

### Шаг 6: Извлечение и загрузка HOG-признаков из изображений {#шаг-6-извлечение-и-загрузка-hog-признаков-из-изображений}
*Цель шага*: Вызов функции для извлечения признаков HOG из изображений. Используется цветовое пространство YCrCb.

```python
X_train_hog, X_test_hog, y_train, y_test = upload_colored_hog_features(cv2.COLOR_BGR2YCrCb, (10, 10), (2, 2))
```
**Результат**:
> Признаки HOG извлечены для обучающей и тестовой выборок. Размерность `X_train_hog`: (1961, 1944), `X_test_hog`: (497, 1944).

### Шаг 7: Обучение и оценка SVC с использованием PCA {#шаг-7-обучение-и-оценка-svc-с-использованием-pca}
*Цель шага*: Обучение и оценка модели Support Vector Classifier (SVC) с использованием PCA для уменьшения размерности.

```python
# Базовая модель
pca_svc = make_pipeline(PCA(n_components=0.6), SVC(random_state=42))
pca_svc.fit(X_train_hog, y_train)
pca_svc_pred = pca_svc.predict(X_test_hog)
print(classification_report(y_test, pca_svc_pred))

# Подбор гиперпараметров
param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__kernel': ['linear', 'rbf', 'poly']}
pca_svc_best = make_pipeline(PCA(n_components=0.6), SVC(random_state=42))
grid_search = GridSearchCV(pca_svc_best, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_hog, y_train)
pca_svc_best_pred = grid_search.predict(X_test_hog)
print(classification_report(y_test, pca_svc_best_pred))
```
**Результат**:
- **Базовая модель**: `accuracy` = 0.70, `f1-macro` = 0.70.
- **Лучшая модель (C=10, kernel='rbf')**: `accuracy` = 0.76, `f1-macro` = 0.76.

### Шаг 8: Обучение и оценка RandomForest Classifier с PCA {#шаг-8-обучение-и-оценка-randomforest-classifier-с-pca}
*Цель шага*: Обучение и оценка модели RandomForestClassifier с PCA.

```python
# Базовая модель
pca_rf = make_pipeline(PCA(n_components=0.6), RandomForestClassifier(random_state=42))
pca_rf.fit(X_train_hog, y_train)
pca_rf_pred = pca_rf.predict(X_test_hog)
print(classification_report(y_test, pca_rf_pred))

# Подбор гиперпараметров
# Лучшие параметры: criterion='entropy', max_depth=None, max_features='sqrt', n_estimators=500
pca_rf_best = make_pipeline(PCA(n_components=0.6), RandomForestClassifier(random_state=42, n_estimators=500, max_features='sqrt', criterion='entropy', max_depth=None))
pca_rf_best.fit(X_train_hog, y_train)
pca_rf_best_pred = pca_rf_best.predict(X_test_hog)
print(classification_report(y_test, pca_rf_best_pred))
```
**Результат**:
- **Базовая модель**: `accuracy` = 0.76, `f1-macro` = 0.76.
- **Лучшая модель**: `accuracy` = 0.77, `f1-macro` = 0.77.

### Шаг 9: Обучение и оценка LightGBM Classifier с PCA {#шаг-9-обучение-и-оценка-lightgbm-classifier-с-pca}
*Цель шага*: Обучение и оценка модели LGBMClassifier с PCA.

```python
# Базовая модель
pca_lgbm = make_pipeline(PCA(n_components=0.6), LGBMClassifier(random_state=42))
pca_lgbm.fit(X_train_hog, y_train)
pca_lgbm_pred = pca_lgbm.predict(X_test_hog)
print(classification_report(y_test, pca_lgbm_pred))

# Подбор гиперпараметров
# Лучшие параметры: min_child_samples=12, num_leaves=60, ...
pca_lgbm_best = make_pipeline(PCA(n_components=0.6), LGBMClassifier(random_state=42, min_child_samples=12, num_leaves=60))
pca_lgbm_best.fit(X_train_hog, y_train)
pca_lgbm_best_pred = pca_lgbm_best.predict(X_test_hog)
print(classification_report(y_test, pca_lgbm_best_pred))
```
**Результат**:
- **Базовая модель**: `accuracy` = 0.74, `f1-macro` = 0.74.
- **Лучшая модель**: `accuracy` = 0.78, `f1-macro` = 0.78.

### Шаг 10: Обучение и оценка CatBoost Classifier с PCA {#шаг-10-обучение-и-оценка-catboost-classifier-с-pca}
*Цель шага*: Обучение и оценка модели CatBoostClassifier с PCA.

```python
# Базовая модель
pca_cat = make_pipeline(PCA(n_components=0.6), CatBoostClassifier(random_state=42, verbose=0))
pca_cat.fit(X_train_hog, y_train)
pca_cat_pred = pca_cat.predict(X_test_hog)
print(classification_report(y_test, pca_cat_pred))

# Подбор гиперпараметров
# Лучшие параметры: depth=10, learning_rate=0.1, ...
pca_cat_best = make_pipeline(PCA(n_components=0.6), CatBoostClassifier(random_state=42, verbose=0, depth=10, learning_rate=0.1))
pca_cat_best.fit(X_train_hog, y_train)
pca_cat_best_pred = pca_cat_best.predict(X_test_hog)
print(classification_report(y_test, pca_cat_best_pred))
```
**Результат**:
- **Базовая модель**: `accuracy` = 0.76, `f1-macro` = 0.76.
- **Лучшая модель**: `accuracy` = 0.79, `f1-macro` = 0.79.

## Ключевые результаты {#ключевые-результаты}
В данном исследовании были протестированы различные ML-модели для классификации изображений с использованием HOG-признаков. Наилучшие результаты показала модель CatBoost после подбора гиперпараметров.

### Таблица результатов
|      Модель      |                                         Гиперпараметры                                         | accuracy | f1-macro |
|:----------------:|:----------------------------------------------------------------------------------------------:|:--------:|:--------:|
|     SVC+PCA      |                                        n_components=0.6                                        |   0.70   |   0.70   |
|     SVC+PCA      |                              n_components=0.6, C=10, kernel='rbf'                              |   0.76   |   0.76   |
| RandomForest+PCA |                                        n_components=0.6                                        |   0.76   |   0.76   |
| RandomForest+PCA |  n_components=0.6, criterion='entropy', max_depth=None, max_features='sqrt', n_estimators=500  |   0.77   |   0.77   |
|   LightGBM+PCA   |                                        n_components=0.6                                        |   0.74   |   0.74   |
|   LightGBM+PCA   | n_components=0.6, min_child_samples=12, num_leaves=60, reg_alpha=2.88e-05, reg_lambda=2.44e-08 |   0.78   |   0.78   |
|   CatBoost+PCA   |                                        n_components=0.6                                        |   0.76   |   0.76   |
|   CatBoost+PCA   |     n_components=0.6, depth=10, learning_rate=0.1, min_child_samples=44, reg_lambda=0.0517     | **0.79** | **0.79** |

- **Вывод**: CatBoost с настроенными параметрами (`depth=10`, `learning_rate=0.1` и др.) и PCA (`n_components=0.6`) достиг наивысшей точности **79%**.
- Использование PCA с сохранением 60% дисперсии позволило значительно сократить время обучения без существенной потери качества.
- Подбор гиперпараметров для всех моделей привел к улучшению метрик `accuracy` и `f1-score`.
