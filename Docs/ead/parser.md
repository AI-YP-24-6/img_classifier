## Об источнике

[goskatalog.ru](https://goskatalog.ru/portal/#/) - Государственный каталог Музейного фонда
Российской Федерации

Регулярно обновляющаяся, единственная в РФ электронная база данных, содержащая основные сведения о
каждом музейном предмете и каждой музейной коллекции, включенных в состав Музейного фонда Российской
Федерации, объединяющего все государственные музеи РФ

На сайте представлены снимки картин, а также сведения о них.

Парсинг сайта осуществлялся по api через 3 запросами:

- получение списка музейных предметов
- получение информации о музейном предмете
- получение исходного снимка музейного предмета

## Сбор данных

Для создания датасета мы использовали api методы:
передаваемые параметры для каждого запроса:
```json
[{
    "fieldName": "name",
    "fieldType": "String",
    "operator": "CONTAINS",
    "fromValue": "null",
    "toValue": "null",
    "value": слово фильтра поиска
},
{
    "fieldName": "typologyId",
    "fieldType": "Number",
    "operator": "EQUALS",
    "fromValue": "null",
    "toValue": "null",
    "value": 1
}]
```
Из этого Post метода мы получаем количество картин на сайте с такими параметрами поиска, чтобы рекурсивно загружать по 100 изображений за запрос.
```python
post_url = "https://goskatalog.ru/muzfo-rest/rest/exhibits/ext"
post_count_params = {
    "statusIds": 6,
    "publicationLimit": "false",
    "cacheEnabled": "true",
    "calcCountsType": 1,
    "limit": 0,
    "offset": 0,
}
```
Получить список картин можно по следующему запросу:
```python
post_url = "https://goskatalog.ru/muzfo-rest/rest/exhibits/ext"
post_params = {
    "statusIds": 6,
    "publicationLimit": "false",
    "cacheEnabled": "true",
    "calcCountsType": 0,
    "dirFields": "desc",
    "limit": 100,
    "offset": сдвиг,
    "sortFields": "id",
}
```
Запросы для получения данных картин и изображение самого большого изображения:
```python
data_url = f"https://goskatalog.ru/muzfo-rest/rest/exhibits/{id картины}"
image_url = f"https://goskatalog.ru/muzfo-imaginator/rest/images/original/{id фотографии}?originalName={оригинальное название картины}"
```
## Запуск парсера
Запускаем парсер, вызывая функцию парсера [Tools.parser.goskatalog_parser()](../tools.md#Tools.parser.goskatalog_parser)
Сам парсер вызывает [Tools.parser.download_art()](../tools.md#Tools.parser.download_art) для каждого изображения.

После мы архивируем результат запроса [Tools.parser.zip_files()](../tools.md#Tools.parser.zip_files) в zip архив и загружаем его на яндекс диск.

Список всех классов на русском языке из других датасетов, которые парсили:

    "Яблоко",
    "Апельсин",
    "Ананас",
    "Клубник",
    "Банан",
    "Арбуз",
    "Тыква",
    "Капуста",
    "Морковь",
    "Цветная капуста",
    "Помидор",
    "Абрикос",
    "Плоды кактуса",
    "Дыня",
    "кукуруза",
    "лимоны",
    "мандарин",
    "Персики",
    "Груша",

Далее идет анализ датасета в [jupyter notebook](../goskatalog.ipynb).

Dataset: ссылка на изначально собранный [yandex-disk](https://disk.yandex.ru/d/bzb677Qx_d6aeg)
