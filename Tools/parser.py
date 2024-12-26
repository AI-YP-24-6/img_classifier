import json
import shutil
from pathlib import Path
from time import sleep

import requests
from tqdm.auto import tqdm


def goskatalog_parser(classes: tuple, path: Path) -> (Path, list):
    """Скачивает картины с сайта госкаталог и возвращает словарь с результатами парсинга
    :param classes: список скачиваемых категорий
    :param path: место куда будут сохраняться картины
    :return: (путь к папке с картинами, словарь с пропущенными картинами)
    """
    missing_art = []

    session = requests.Session()
    session.headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
        "Accept-Language": "en-GB,ru;q=0.7,en;q=0.3",
    }
    post_count_params = {
        "statusIds": 6,
        "publicationLimit": "false",
        "cacheEnabled": "true",
        "calcCountsType": 1,
        "limit": 0,
        "offset": 0,
    }
    post_url = "https://goskatalog.ru/muzfo-rest/rest/exhibits/ext"

    path /= "art_dataset"

    for class_name in classes:
        path_class = path / class_name
        path_class.mkdir(parents=True, exist_ok=True)
        json_params = [
            {
                "fieldName": "name",
                "fieldType": "String",
                "operator": "CONTAINS",
                "fromValue": "null",
                "toValue": "null",
                "value": class_name,
            },
            {
                "fieldName": "typologyId",
                "fieldType": "Number",
                "operator": "EQUALS",
                "fromValue": "null",
                "toValue": "null",
                "value": 1,
            },
        ]

        count_response = session.post(post_url, params=post_count_params, json=json_params)
        art_count = count_response.json().get("statistics")[0].get("count")

        for i in tqdm(range(0, art_count, 100), desc=f"Downloading {class_name}"):
            post_params = {
                "statusIds": 6,
                "publicationLimit": "false",
                "cacheEnabled": "true",
                "calcCountsType": 0,
                "dirFields": "desc",
                "limit": 100,
                "offset": i,
                "sortFields": "id",
            }
            response = session.post(post_url, params=post_params, json=json_params)
            for art in response.json().get("objects"):
                try:
                    download_art(art, path_class, session)
                except (requests.exceptions.HTTPError, IndexError, AttributeError):
                    missing_art.append({art.get("id"), art.get("name")})

    return path, missing_art


def download_art(art: dict, path: Path, session: requests.Session) -> None:
    """Скачивает одну картину с сайта и сохраняет ее в папку вместе с JSON-файлом
    :param art: словарь с данными картины
    :param path: путь к папке
    :param session: сессия для запросов
    """
    sleep(1)

    data_url = f"https://goskatalog.ru/muzfo-rest/rest/exhibits/{art.get('id')}"
    data_json = session.get(data_url).json()
    try:
        image_name = data_json.get("images")[0].get("fileName")
<<<<<<< HEAD
    except IndexError:
        image_name = art.get("mainImage").get("code")
        if not image_name:
            raise IndexError

    image_url = f"https://goskatalog.ru/muzfo-imaginator/rest/images/original/{art.get('mainImage')['code']}?originalName={image_name}"
=======
    except IndexError as e:
        image_name = art.get("mainImage").get("code")
        if not image_name:
            raise IndexError from e

    image_url = (
        f"https://goskatalog.ru/muzfo-imaginator/rest/images/original/{art.get('mainImage')['code']}"
        f"?originalName={image_name}"
    )
>>>>>>> 3d25aabb8347edce5a9e182f4e9b2083f9459fc4
    image = session.get(image_url)
    image.raise_for_status()

    image_id = str(data_json.get("regNumber"))
    image_name = image_id + "." + image_name.split(".")[-1]  # jpeg, png, jpg

    path_image = path / image_name
    path_json = (path / image_id).with_suffix(".json")

    path_image.write_bytes(image.content)
<<<<<<< HEAD
    with open(path_json, "w") as file:
=======
    with open(path_json, "w", encoding="utf-8") as file:
>>>>>>> 3d25aabb8347edce5a9e182f4e9b2083f9459fc4
        json.dump(data_json, file)


def zip_files(path: Path) -> str:
    """Создает zip-файл из папки с картинками
    :param path: путь к папке с картинками
    :return: путь к zip-файлу
    """
    return shutil.make_archive(path.name, "zip", path)
