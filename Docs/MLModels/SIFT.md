---
hide:
  - navigation
---

# Классификация изображений с помощью SIFT

## Краткое описание
- В данном ноутбуке исследуется использование дескриптора SIFT (Scale-Invariant Feature Transform) для задачи классификации изображений.
- Применяется упрощенный подход к формированию вектора признаков: для каждого изображения вычисляется среднее значение по первым 128 SIFT-дескрипторам.
- Проводится сравнение производительности нескольких ML-моделей: `SVC`, `RandomForest`, `LightGBM` и `CatBoost`.
- Для `LightGBM` и `CatBoost` выполняется подбор гиперпараметров с помощью `RandomizedSearchCV`.

## Содержание
- [Шаг 1: Установка и импорт библиотек](#шаг-1-установка-и-импорт-библиотек)
- [Шаг 2: Загрузка данных](#шаг-2-загрузка-данных)
- [Шаг 3: Извлечение SIFT-признаков](#шаг-3-извлечение-sift-признаков)
- [Шаг 4: Обучение базовых моделей](#шаг-4-обучение-базовых-моделей)
- [Шаг 5: Подбор гиперпараметров для LightGBM](#шаг-5-подбор-гиперпараметров-для-lightgbm)
- [Шаг 6: Подбор гиперпараметров для CatBoost](#шаг-6-подбор-гиперпараметров-для-catboost)
- [Ключевые результаты](#ключевые-результаты)

### Шаг 1: Установка и импорт библиотек
*Цель шага*: Установка и импорт всех необходимых зависимостей.

```python
!pip install catboost -q
!pip install lightgbm -q
!pip install gdown -q

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
```

### Шаг 2: Загрузка данных
*Цель шага*: Загрузка и распаковка архива с изображениями.

```python
DATASET_DIR = os.path.join(os.getcwd(), "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
ZIP_PATH = os.path.join(os.getcwd(), "dataset_32_classes.zip")

if not os.path.exists(TRAIN_DIR):
    # Логика загрузки и распаковки gdown и zipfile
    pass
```

### Шаг 3: Извлечение SIFT-признаков
*Цель шага*: Определение функций для извлечения и агрегации SIFT-дескрипторов.

**Подробности**:
Вместо полноценного подхода Bag of Visual Words (BoVW), здесь используется упрощение: для каждого изображения извлекаются SIFT-дескрипторы, и если их больше 128, берутся первые 128, после чего вычисляется их среднее. Это позволяет получить вектор признаков фиксированной длины (128), но может приводить к потере информации.

```python
def get_SIFT_descriptors(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def create_feature_vector(descriptors, num_features=128):
    feature_vector = np.zeros(num_features)
    if descriptors is not None and len(descriptors) > 0:
        if descriptors.shape[0] < num_features:
            feature_vector = np.mean(descriptors, axis=0)
        else:
            feature_vector = np.mean(descriptors[:num_features], axis=0)
    return feature_vector

def analyze_dataset(image_folder, size_img):
    # ... (код для обхода датасета и применения функций выше)
    pass

# Извлечение признаков
X_train, y_train = analyze_dataset(TRAIN_DIR, 64)
X_test, y_test = analyze_dataset(TEST_DIR, 64)
```

### Шаг 4: Обучение базовых моделей
*Цель шага*: Обучение и оценка нескольких классификаторов на извлеченных SIFT-признаках.

```python
# SVC
svc = SVC(random_state=RANDOM_STATE)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print("SVC Report:\n", classification_report(y_test, svc_pred))

# RandomForest
rf = RandomForestClassifier(random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("RandomForest Report:\n", classification_report(y_test, rf_pred))

# LightGBM
lgbm = LGBMClassifier(random_state=RANDOM_STATE)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
print("LightGBM Report:\n", classification_report(y_test, lgbm_pred))

# CatBoost
cat = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_test)
print("CatBoost Report:\n", classification_report(y_test, cat_pred))
```

### Шаг 5: Подбор гиперпараметров для LightGBM
*Цель шага*: Улучшение производительности LightGBM с помощью `RandomizedSearchCV`.

```python
# Определение сетки параметров
param_dist = {
    'num_leaves': [10, 20, 30, 40, 50],
    'min_child_samples': [5, 10, 15, 20],
    # ... другие параметры
}

lgbm_search = RandomizedSearchCV(LGBMClassifier(random_state=RANDOM_STATE), param_distributions=param_dist, n_iter=10, cv=3, random_state=RANDOM_STATE)
lgbm_search.fit(X_train, y_train)
lgbm_best_pred = lgbm_search.predict(X_test)
print("Optimized LightGBM Report:\n", classification_report(y_test, lgbm_best_pred))
```

### Шаг 6: Подбор гиперпараметров для CatBoost
*Цель шага*: Улучшение производительности CatBoost с помощью `RandomizedSearchCV`.

```python
# Определение сетки параметров
param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    # ... другие параметры
}

cat_search = RandomizedSearchCV(CatBoostClassifier(random_state=RANDOM_STATE, verbose=0), param_distributions=param_dist, n_iter=10, cv=3, random_state=RANDOM_STATE)
cat_search.fit(X_train, y_train)
cat_best_pred = cat_search.predict(X_test)
print("Optimized CatBoost Report:\n", classification_report(y_test, cat_best_pred))
```

## Ключевые результаты

### Сводная таблица результатов
| Модель | Accuracy (базовая) | F1-macro (базовая) | Accuracy (оптимизированная) | F1-macro (оптимизированная) |
|:---|:---:|:---:|:---:|:---:|
| SVC | 0.65 | 0.64 | - | - |
| RandomForest | 0.67 | 0.66 | - | - |
| LightGBM | 0.70 | 0.69 | **0.72** | **0.71** |
| CatBoost | 0.70 | 0.69 | 0.71 | 0.70 |

### Интерпретация модели
- **Эффективность SIFT**: Упрощенный метод извлечения SIFT-признаков позволил достичь максимальной точности **72%** с помощью оптимизированной модели LightGBM. Этот результат несколько ниже, чем у моделей на основе HOG, что может указывать на недостатки метода агрегации признаков.
- **Производительность моделей**: Градиентные бустинги (LightGBM и CatBoost) показали себя значительно лучше, чем классические модели, такие как SVC и RandomForest.
- **Влияние оптимизации**: Подбор гиперпараметров с помощью `RandomizedSearchCV` позволил улучшить метрики для LightGBM и CatBoost, подтверждая важность этого этапа.
- **Ограничения**: Метод усреднения дескрипторов является грубым приближением. Для дальнейшего улучшения качества рекомендуется использовать более продвинутые техники, такие как Bag of Visual Words (BoVW) или Fisher Vectors, которые позволяют кодировать информацию о распределении локальных признаков более эффективно.
