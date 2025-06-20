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

## Описание данных:
Используется [данный датасет](https://drive.google.com/file/d/1-1ehpRd0TnwB1hTHQbFHzdf55SrIri4f/view)
В датасете представлены изображения фруктов и овощей.
Датасет имеет **33 класса по 1400 изображений в каждом** (1120 трейн / 280 тест)<br/>
[Подробнее про данные...](dataset.md)

## Разведочный анализ (EDA) датасета изображений овощей и фруктов
Был проведен разведочный анализ.
В результате был выявлен дублирующийся класс: *Strawberry/Strawberries*. Стоит провести эксперименты с разными размерами изображений, чтобы понять, какой размер стоит использовать для обучения модели.<br/>
[Подробнее про разведочный анализ...](./EDA/EDA.md)

## Baseline
В качестве baseline-модели был выбран **метод опорных векторов SVC** с использованием PCA для уменьшения размерности данных. Для выделения признаков были испробованы HOG, SIFT, ResNet. Для baseline был выбран **HOG**, т.к. он выиграл в скорости по сравнению с ResNet и в качестве метрик по сравнению с SIFT.
Для обучения использовались цветные изображения размером 64*64px.
Т.к. в каждом классе содержится 1400 изображений, то для сбалансированного датасета используется метрика **accuracy**. Также вспомогательной метрикой используется **f1 macro**.<br/>
[Подробнее про baseline...](./Baseline/baseline.md)

## Нелинейные ML-модели
Лучше всего показал себя бустинг: CatBoost для признаков, извлеченных с помощью HOG, LightGBM - для SIFT.<br/>
[Подробнее про проведенные эксперименты...](./MLModels/ml_models.md)

## Применение моделей глубинного обучения
Хорошо себя показала модель EfficientNet-B0.<br/>
[Подробнее про проведенные эксперименты...](./DLModels/DLModels.md)

## Разработка микросервиса
Взаимодействие с полученной моделью реализовано с помощью веб-интерфейса FastAPI и streamlit-приложения.

На данный момент оба модуля упакованы в докер контейнер. Streamlit-приложение отвечает на запросы пользователя, используя функционал сервиса FastAPI.
Кроме того, проект развернут на арендованном VPS. Посмотреть работу FastAPI можно по адресу:
```
http://213.171.25.239:54545/api/openapi
```
Приложение Streamlit развернуто по адресу:
```
http://213.171.25.239:8501/
```
Процесс развертывания подробно описан в секции **Virtual Private Server**.

Ниже на гифках краткая демонстрация функционала веб-интерфейса Streamlit и FastAPI.

### Веб-интерфейс FastAPI
- Запуск FastAPI-приложения и проверка работы  /api/v1/dataset/load (**Load Dataset**)
  - Метод: POST
  - Описание: Позволяет загрузить датасет для дальнейшего использования. На вход подается архив, содержащий папки (с названиями классов) с изображениями для каждого класса.. После загрузки архива происходит сбор информации о датасете. Данные о датасете хранятся в моделе DatasetInfo:
количество изображений в каждом классе
количество дубликатов в каждом классе
размеры изображений
информация о цветах изображений
DatasetInfo хранится на сервере и передается в ответе на запрос клиента.
  - Ответ:
200 OK: Возвращает информацию о загруженном датасете, включая количество изображений в каждом классе, информацию о дубликатах, размеры изображений и цветовые характеристики.
  - Ошибки:
400 Bad Request: Если загруженный файл не является ZIP-архивом или произошла ошибка при обработке.

![load_dataset.gif](Media/load_dataset.gif)

- /api/v1/dataset/info (**Get Dataset Info**)
    - Метод: GET
    - Описание: возвращает информацию о датасете, полученную при загрузке архива в методе Load Dataset. Если датасет загружен на сервер, но не проинициализирован (к примеру, после перезапуска сервера), то данные о датасете собираются в методе  Get Dataset Info, иначе - отдается хранящаяся на сервере модель. Если датасет не загружен, то метод выдаст исключение.
    - Ответ: Информация о датасете (DatasetInfo).
    - Код ответа:
200 OK — успешный запрос.
400 Bad Request — датасет не загружен.

- /api/v1/dataset/samples (**Dataset Samples**)
  - Метод: GET
  - Описание: Dataset Samples создает изображение с примерами картинок в каждом из классов. В первый раз изображение создается с помощью pyplot и сохраняется на сервере, потом используется сохраненное изображение для отправки клиенту. Если датасет не загружен, то метод выдаст исключение.
  - Ответ: Стрим изображений в формате PNG.
  - Код ответа:
  200 OK — успешный запрос.
  400 Bad Request — датасет не загружен.

![dataset_info.gif](Media/dataset_info.gif)

- /api/v1/models/fit (**fit**)
    - Метод: POST
    - Описание: fit служит для создания новой модели на сервере. В данном методе создается модель с использованием гиперпараметров для PCA и SVC. После создания новой модели есть возможность получить кривые обучения (если указать with_learning_curve = True). Модель обучается 10 секунд, по истечении времени процесс обучения модели прерывается с исключением, чтобы не нагружать сервер тяжелыми моделями. Если модели удалось обучиться, то данные о ней сохраняются в models. Если датасета для обучения нет на сервере, то метод выдаст исключение.
    - Параметры:
      - config (опционально): гиперпараметры модели.
      - with_learning_curve: сохранять ли кривую обучения.
      - name: название модели.
    - Ответ: Информация о созданной модели (ModelInfo).
    - Код ответа:
201 Created — успешное обучение.
400 Bad Request — ошибка обучения.
408 Request Timeout — превышение времени обучения.

![fit.gif](Media/fit.gif)

- /api/v1/models/list_models (**List Models**)
    - Метод: GET
    - Описание: возвращает все хранящиеся на сервер ранее обученные модели. Также на сервер присутствует заранее загруженная baseline–модель. Информация о моделях возвращается в виде словаря, где ключом является идентификатор модели, а значением информация о модели ModelInfo.
    - Ответ: Словарь с информацией о моделях (dict[str, ModelInfo]).
    - Код ответа: 200 OK
- /api/v1/models/info/{model_id} (**Model Info**)
    - Метод: GET
    - Описание: возвращает пользователю информацию о конкретной модели по указанному идентификатору. Если модели с указанным идентификатором нет на сервере, то выдастся исключение.
    - Ответ: Информация о модели (ModelInfo).
    - Код ответа:
200 OK — успешный запрос.
400 Bad Request — модель не найдена.
- /api/v1/models/load (**Load**)
  - Метод: POST
  - Описание: позволяет сделать активной одну из хранящихся на сервере моделей. У модели существует уникальный идентификатор, по которому модели хранятся в models. Если модели с указанным идентификатором нет на сервере, то выдастся исключение
- /api/v1/models/predict (**Predict**)
  - Метод: POST
  - Описание: предсказывает с помощью активной модели класс по переданному в него изображению. Если активной модели нет, то метод выдаст исключение.
  - Параметры: file: файл изображения для предсказания.
  - Ответ: Предсказанный класс (PredictionResponse).
  - Код ответа:
  200 OK — успешное предсказание.
  400 Bad Request — ошибка (например, модель не выбрана).

![predict.gif](Media/predict.gif)

- /api/v1/models/predict_proba (**Predict Proba**)
    - Метод: POST
    - Описание: Возвращает предсказанный класс с вероятностью, при условии, что в загруженной модели был задан параметр svc__probability = true.
    - Параметры: file: файл изображения для предсказания.
    - Ответ: Предсказание с вероятностью (ProbabilityResponse).
    - Код ответа:
200 OK — успешное предсказание.
400 Bad Request — ошибка.

 ![predict_proba](Media/predict_proba.gif)

- /api/v1/models/unload (**Unload**)
  - Метод: POST
  - Описание: Выгружает текущую активную модель из памяти. То есть после выполнения данного метода нет активной модели и выполнить методы predict и predict_proba не получится.
  - Ответ: Успешное сообщение (ApiResponse).
  - Код ответа: 200 OK
- /api/v1/models/remove/{model_id} (**Remove**)
  - Метод: DELETE
  - Описание: удаляет модель по идентификатору из тех, которые до этого создавались методом fit. Если модель имеет тип custom (пользовательская модель) и есть на сервере, то она будет удалена и будет возвращен словарь с ModelInfo, оставшимися на сервере. Если модели нет на сервере или модель типа baseline (baseline-модель всегда хранится на сервере), то будет возвращено исключение.
  - Ответ: Список оставшихся моделей (dict[str, ModelInfo]).
  - Код ответа:
  200 OK — успешное удаление.
  404 Not Found — модель не найдена.
- /api/v1/models/remove_all (**Remove All**)
  - Метод: DELETE
  - Описание: Удаляет все пользовательские модели (custom) на сервере.
  - Ответ: возвращается словарь с имеющимися baseline моделями на сервере.
  - Код ответа: 200 OK

![unload_remove.gif](Media/unload_remove.gif)


### Приложение Streamlit

Приложение на Streamlit представляет собой сервис для обучения, анализа и применения моделей машинного обучения для
классификации изображений фруктов и овощей. Оно организовано в виде трёх разделов, представленных в боковом меню:

- **EDA:**

    Позволяет загружать датасет (в формате .zip, содержащий изображения и аннотации) на сервер.
    Отображает основную статистику датасета, включая:
    - Средний размер изображений.
    - Распределение изображений по классам.
    - Распределение дубликатов (если они имеются).

    Визуализирует средние значения и стандартные отклонения по цветовым каналам (R, G, B) для каждого класса.
    Показывает примеры изображений из загруженного датасета.
    Использует серверные API для загрузки данных, получения метрик и изображений.
- **Обучение модели:**

    Содержит два основных блока:
    - Работа с уже обученными моделями:
Список доступных моделей, обученных ранее.
Отображение параметров модели и её кривой обучения.
Возможность удаления одной или всех моделей.
    - Создание новой модели:
Выбор гиперпараметров для алгоритма SVC:
Параметр регуляризации C.
Тип ядра (например, linear, poly, rbf).
Включение оценки вероятности.
Построение кривой обучения.
Обучение новой модели с заданными параметрами и сохранение её на сервере.
Использует серверные API для обучения моделей и управления ими.
- **Инференс:**

    Позволяет загрузить изображение (форматы .jpeg, .png, .jpg) для классификации с использованием выбранной модели.
Загружает выбранную модель с сервера и выполняет предсказание.
Возвращает результат:
    - Предсказанный класс изображения.
    - Вероятность принадлежности к классу (если включена опция probability при обучении модели).

    Использует серверные API для загрузки модели и выполнения предсказания.

![streamlit.gif](Media/streamlit.gif)

## Инструкция по запуску

- Склонируйте репозиторий
```
git clone https://github.com/AI-YP-24-6/img_classifier.git
```
FastApi сервер отдельно можно запустить по в файле Backend/app/main.py

Streamlit приложение отдельно можно запускается по в файле Frontend/run.py

Для запуска FastApi и Streamlit одновременно выполните команды в 2 консолях:
```
$env:PYTHONPATH="C:<path>\img_classifier"
streamlit run .\Client\app_client.py --server.port=8081 --server.address=127.0.0.1
uvicorn Backend.app.main:app --host=0.0.0.0 --port=54545
```

- Для развертывания двух докер образов выполните команду:
```
docker compose up -d --build
```
Описание docker-compose:
В данном сервисе поднимаются 2 объединенных докер контейнера с веб-приложением FastApi и веб-приложением Streamlit.

## Virtual Private Server

Приложение развернуто на VPS от Cloud.ru.
Ниже представлена инструкция по развертыванию приложения на VPS от Cloud.ru:

Для начала зайдите на официальный сайт Cloud.ru, чтобы создать новую виртуальную машину (VM). Выберите вариант с бесплатным тарифом (free tier).

Далее нам предложат список предустановленных программных компонентов для вашей новой виртуальной машины. Можете выбрать наиболее подходящие опции в зависимости от ваших потребностей.

Рекомендуется сразу настроить SSH-соединение для более удобного доступа к серверу. После завершения настройки вы сможете подключаться к машине двумя способами: либо напрямую через веб-интерфейс сайта, используя пароль, либо через SSH.

Из списка можем выбрать что будет предустановлено у нас на сервер.

После создания машины мы можем как сразу подключиться к ней через интерфейс сайта по паролю, так и по ssh
```
ssh -i ~/.ssh/<cloudru-key> <login>@<ip>
```
По умолчанию там запускается -sh, которая не очень удобна для работы. Чтобы переключиться на стандартный Bash, выполните команду /bin/bash.

Теперь у нас настроена базовая виртуальная машина, но без доступа к интернету и открытым портам. Для открытия нужных портов следуйте инструкциям на официальном сайте Cloud.ru. Например, этой: https://cloud.ru/docs/marketplace/ug/services/mind-migrate/mind__subnet-configuration.html#id4.

Не забудем также убедиться, что внутренние брандмауэры на самой виртуальной машине разрешают использование этих портов. В системе Ubuntu это делается через утилиту UFW, а в других системах – через команды вроде firewall-cmd.

Как проверить работоспособность открытых портов? Попробуем отправить запрос на 213.171.25.239:54545. Если мы можем получить ответ, значит порт открыт.

Если вы используете Windows, попробуйте выполнить следующую команду:
```
Test-NetConnection <ip> -Port 54545
```
Она выведет результат проверки соединения, например: TcpTestSucceeded : False

Если вы работаете в Linux, воспользуйтесь командой telnet:
```
telnet <ip> 54545
```
Но почему Windows выдает что порт закрыт по протоколу TCP, хотя сам хост отвечает на запросы ICMP (ping). Это происходит потому, что на данном порте еще нет запущенного приложения, которое могло бы принимать входящие соединения.

Чтобы протестировать работу порта в Linux, вы можете использовать команду netcat (nc) для запуска простого сервера на нужном порте:
```
netcat -l 8501
```
Далее уже можно запускать готовое приложение
```
git clone -branch <feature-fastapi>> <progect http url >
docker compose up -d --build
```
И теперь на: 8501 можно будет увидеть готовое приложение
___

## Этапы проекта
1. Сбор данных
2. Предобработка изображений
3. Подготовка к машинному обучению
4. Машинное обучение (ML)
5. Подбор гиперпараметров
6. Глубокое обучение (DL)
7. Реализация микросервиса

- ✔️ Чекпоинт 1. Знакомство
- ✔️ Чекпоинт 2. Данные и EDA
- ✔️ Чекпоинт 3. Линейные модели | Простые DL модели - 1
- ✔️ Чекпоинт 4. MVP
Далее на данный момент порядок и даты не фиксированы:
- ✔️   Нелинейные ML-модели | Простые DL модели - 2
- ✔️   Модели глубинного обучения
- ✔️   Финал

## WorkFlow
В проекте придерживаемся [Github Flow](https://docs.github.com/en/get-started/using-github/github-flow)

*Будет дополняться по мере развития проекта...*
