---
hide:
  - navigation
---

# Ансамбль моделей (Voting Classifier) на основе HOG и SIFT

## Краткое описание
- В данном ноутбуке создается ансамбль моделей с помощью `VotingClassifier` для улучшения качества классификации изображений.
- Ансамбль объединяет две различные модели, каждая из которых использует свой метод извлечения признаков: HOG и SIFT.
- Первая модель: HOG-признаки + PCA + CatBoost.
- Вторая модель: SIFT-признаки + PCA + LightGBM.
- Итоговая модель достигает точности (accuracy) **81%** за счет комбинирования сильных сторон обоих подходов.

## Содержание
- [Шаг 1: Установка и импорт библиотек](#шаг-1-установка-и-импорт-библиотек)
- [Шаг 2: Настройка окружения и загрузка данных](#шаг-2-настройка-окружения-и-загрузка-данных)
- [Шаг 3: Создание кастомных трансформеров](#шаг-3-создание-кастомных-трансформеров)
- [Шаг 4: Определение пайплайна HOG + CatBoost](#шаг-4-определение-пайплайна-hog--catboost)
- [Шаг 5: Определение пайплайна SIFT + LightGBM](#шаг-5-определение-пайплайна-sift--lightgbm)
- [Шаг 6: Создание и обучение ансамбля](#шаг-6-создание-и-обучение-ансамбля-моделей)
- [Шаг 7: Оценка модели](#шаг-7-оценка-модели-на-тестовых-данных)
- [Ключевые результаты](#ключевые-результаты)

### Шаг 1: Установка и импорт библиотек {#шаг-1-установка-и-импорт-библиотек}
*Цель шага*: Установка и импорт необходимых библиотек для проекта.

```python
!pip install catboost -q
!pip install lightgbm -q

import zipfile
import os
import shutil
import random
import gdown
import cv2
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
from tqdm.auto import tqdm
```

### Шаг 2: Настройка окружения и загрузка данных {#шаг-2-настройка-окружения-и-загрузка-данных}
*Цель шага*: Инициализация констант, определение путей и загрузка датасета.

```python
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

# Определение путей к данным
DRIVE_DIR = os.getcwd()
DATASET_DIR = os.path.join(os.getcwd(), "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
ZIP_PATH = os.path.join(DRIVE_DIR, "dataset_32_classes_splitted.zip")

# Распаковка архива
if not os.path.exists(TRAIN_DIR):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("./dataset")
```
**Результат**:
> Данные загружены и готовы к использованию.

### Шаг 3: Создание кастомных трансформеров {#шаг-3-создание-кастомных-трансформеров}
*Цель шага*: Создание классов-трансформеров для HOG и SIFT, чтобы встроить извлечение признаков в пайплайн `scikit-learn`.

```python
class HogTransformer(BaseEstimator, TransformerMixin):
    # ... (код трансформера HOG)

class SiftTransformer(BaseEstimator, TransformerMixin):
    # ... (код трансформера SIFT)
```

**Подробности**:
- `HogTransformer`: извлекает HOG-признаки из изображения. Гибко настраивается через параметры `orientations`, `pixels_per_cell` и др.
- `SiftTransformer`: извлекает SIFT-дескрипторы и агрегирует их с помощью Bag of Visual Words (BoVW) с использованием KMeans.

### Шаг 4: Определение пайплайна HOG + CatBoost {#шаг-4-определение-пайплайна-hog--catboost}
*Цель шага*: Создание пайплайна, который последовательно извлекает HOG-признаки, применяет PCA и обучает модель CatBoost.

```python
hog_catboost_pipeline = Pipeline([
    ('hog_transformer', HogTransformer(orientations=3, pixels_per_cell=(10, 10), cells_per_block=(2, 2))),
    ('pca', PCA(n_components=0.6)),
    ('catboost', CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, depth=10, learning_rate=0.1))
])
```

### Шаг 5: Определение пайплайна SIFT + LightGBM {#шаг-5-определение-пайплайна-sift--lightgbm}
*Цель шага*: Создание пайплайна, который извлекает SIFT-признаки, применяет PCA и обучает модель LightGBM.

```python
sift_lgbm_pipeline = Pipeline([
    ('sift_transformer', SiftTransformer()),
    ('pca', PCA(n_components=0.6)),
    ('lgbm', LGBMClassifier(random_state=RANDOM_STATE, min_child_samples=66, num_leaves=165))
])
```

### Шаг 6: Создание и обучение ансамбля моделей {#шаг-6-создание-и-обучение-ансамбля-моделей}
*Цель шага*: Объединение двух пайплайнов в один `VotingClassifier` для принятия совместного решения.

```python
voting_clf = VotingClassifier(
    estimators=[
        ('hog_catboost', hog_catboost_pipeline),
        ('sift_lgbm', sift_lgbm_pipeline)
    ],
    voting='soft', # Усреднение вероятностей для более точного прогноза
    n_jobs=-1
)

voting_clf.fit(X_train, y_train)
```

**Результат**:
> Ансамбль моделей успешно обучен на тренировочных данных.

### Шаг 7: Оценка модели на тестовых данных {#шаг-7-оценка-модели-на-тестовых-данных}
*Цель шага*: Оценка производительности ансамбля на тестовых данных.

```python
y_pred = voting_clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Результат**:
> ```
>               precision    recall  f1-score   support
>
>      airplane       0.88      0.88      0.88        16
>      bear       0.81      0.81      0.81        16
>      bicycle       0.93      0.88      0.90        16
>      bird       0.69      0.69      0.69        16
>      boat       0.88      0.88      0.88        16
>      ...        ...
>      truck       0.80      0.75      0.77        16
>      turtle       0.87      0.81      0.84        16
>
>    accuracy                           0.81       497
>   macro avg       0.81      0.81      0.81       497
>weighted avg       0.81      0.81      0.81       497
> ```

## Ключевые результаты {#ключевые-результаты}

- **Точность ансамбля**: Итоговая модель `VotingClassifier` достигла **accuracy 81%** и **f1-score (macro) 81%** на тестовой выборке.
- **Интерпретация модели**: Ансамбль успешно объединил модели, основанные на разных типах признаков. HOG хорошо описывает общую форму объектов, в то время как SIFT фокусируется на ключевых точках. Их комбинация позволила создать более робастный классификатор, который превосходит по качеству каждую из моделей в отдельности.
- **Преимущества подхода**: Использование `VotingClassifier` с параметром `voting='soft'` позволяет усреднять предсказанные вероятности, что часто дает лучший результат, чем простое голосование по большинству (`hard voting`).
