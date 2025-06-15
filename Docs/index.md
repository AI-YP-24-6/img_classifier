# **Классификатор изображений фруктов и овощей**

Годовой проект магистратуры Искусственный интеллект ВШЭ, первый курс.
___

## Команда:

- [Красюков Александр Сергеевич](https://github.com/MrASK2024) — (tg: `@kas_dev`)
- [Мевший Илья Павлович](https://github.com/milia20) — (tg: `@Ilya12`)
- [Климашевский Илья Алексеевич](https://github.com/Ilya-Klim) — (tg: `@ferox_Y`)
- [Писаренко Сергей Сергеевич](https://github.com/SerejkaP) — (tg: `@SerejP`)

## Куратор:

[Тимур Ермешев](https://github.com/ermetim) - (tg: `@SofaViking`)
___

## Описание проекта:

Задача классификации изображений - одна из классических задач компьютерного зрения, которая не
теряет своей актуальности, поскольку в различных приложениях и сервисах требуется классифицировать
объекты, изображенные на фотографии.
Проект представляет собой сервис, решающий эту задачу.

## Этапы проекта:

1. Сбор данных
2. Предобработка изображений
3. Подготовка к машинному обучению
4. Машинное обучение (ML)
5. Подбор гиперпараметров
6. Глубокое обучение (DL)
7. Реализация микросервиса

## Список чекпоинтов:

Исследовательский анализ данных:

* [сбор данных](EDA/dataset.md)
* [парсинг](EDA/parser.md)

## Project layout

```bash
.
├── Backend                             # backend FastAPI app
│    ├── app
│    │    ├── __init__.py
│    │    ├── api
│    │    │    ├── __init__.py
│    │    │    ├── models.py            # файл описания моделей данных
│    │    │    └── v1
│    │    │        ├── __init__.py
│    │    │        └── api_route.py     # файл с роутами FastAPI
│    │    ├── main.py                   # основной файл приложения
│    │    └── services                  # вспомогательные файлы
│    │        ├── __init__.py
│    │        ├── analysis.py
│    │        ├── model_loader.py
│    │        ├── model_trainer.py
│    │        ├── pipeline.py
│    │        ├── preprocessing.py
│    │        └── preview.py
│    └── data
│        └── baseline.pkl               # подготовленная модель
├── Baseline                            # baseline ML моделей
│    ├── baseline.ipynb
│    ├── baseline.md
│    ├── baseline_HOG.ipynb
│    ├── baseline_ResNet18.ipynb
│    ├── baseline_SIFT.ipynb
│    └── baseline_Vgg16.ipynb
├── DLModels                            # baseline DLModels
├── MLModels                            # baseline MLModels
├── Client                              # streamlit app
│    ├── app_client.py                  # основной файл приложения
│    ├── eda_page.py                    # файл отображения статистики модели
│    ├── model_inference.py             # код для инференса
│    ├── model_training_page.py         # код для подготовки модели
│    └── run.py                         # файл для запуска streamlit
├── Docs                                # документация
│    ├── EDA                            # документация по EDA
│    └── img                            # изображения для документации
├── EDA
│    ├── EDA.md                     # Краткое описание EDA объединенного датасета
│    ├── EDA_Fruits360.md           # Краткое описание EDA датасета
│    ├── EDA_Vegetables.md          # Краткое описание EDA датасета
│    ├── EDA_tasty_fruit.md         # Краткое описание EDA датасета
│    └── Notebooks
│         ├── EDA.ipynb             # notebook с EDA объединенного датасета
│         ├── EDA_Vegetables.ipynb  # notebook с EDA датасета Vegetables
│         ├── EDA_fruits360.ipynb   # notebook с EDA датасета fruits360
│         ├── EDA_tasty_fruit.ipynb # notebook с EDA датасета tasty_fruit
│         └── dataset_merging.ipynb # notebook с объединением датасетов
├── LICENSE                         # Лицензия MIT
├── Notebooks                       # Папка с ноутбуками
│    ├── download_datasets.ipynb    # Загрузка сохраненных датасетов с Yandex disk
│    ├── goskatalog.ipynb           # notebook анализа датасета goskatalog
│    ├── kaggle.json.example        # example of kaggle.json
│    └── parser.ipynb               # notebook парсера сайта goskatalog.ru
├── README.md                       # Описание проекта
├── Tools                           # Папка с инструментами проекта
│    ├── __init__.py                # Позволяет импортировать все модули из файлов директории
│    ├── analysis.py                # Позволяет находить все jpeg-файлы в директории
│    ├── download.py                # Позволяет скачать и разархивировать датасет
│    ├── logger_config.py           # Позволяет настроить логгер для сервера и клиента
│    ├── notebook.py                # Позволяет обновлять идентификаторы ячеек в ноутбуке
│    └── parser.py                  # Парсер сайта goskatalog.ru
├── .dockerignore                   # игнорируемые файлы для Docker
├── .editorconfig                   # конфигурация редактора IDE
├── .env.example                    # пример файла .env
├── .flake8                         # конфигурационный файл для форматера flake8
├── .gitignore                      # игнорируемые файлы, которые нельзя коммитить в Git
├── .pre-commit-config.yaml         # конфигурация для pre-commit hook
├── backend.Dockerfile              # Файл конфигурации docker backend приложения
├── checkpoint.md                   # Описание этапов проекта
├── client.Dockerfile               # Файл конфигурации docker client приложения
├── compose.yaml                    # Файл конфигурации docker-compose
├── dataset.md                      # Описание датасетов
├── mkdocs.yml                      # Файл конфигурации mkdocs
├── poetry.lock                     # Генерируемый файл по зависимостям
└── pyproject.toml                  # Главный конфигурационный файл
```
