import io
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import requests


def get_ya_disk_url(public_key: str) -> str:
    """Возвращает ссылку на скачивание файла с https://disk.yandex.ru по публичной общей ссылке
    :param public_key: публичная ссылка на файл
    :return: ссылка на скачивание
    """
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"

<<<<<<< HEAD
    final_url = base_url + urlencode({"public_key": public_key})
    response = requests.get(final_url, timeout=20)
=======
<<<<<<< HEAD
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
=======
    final_url = base_url + urlencode({"public_key": public_key})
    response = requests.get(final_url, timeout=20)
>>>>>>> 3d25aabb8347edce5a9e182f4e9b2083f9459fc4
>>>>>>> feature-fastapi
    download_url = response.json()["href"]
    return download_url


def download_zip(url: str, path: str | Path) -> Path:
    """Скачивает zip-файл по ссылке и возвращает путь до него
    :param url: ссылка на zip-файл с Yandex.Disk
    :param path: место, куда нужно сохранить zip-файл
    :return: путь до загруженного разархивированного zip-файла
    """
<<<<<<< HEAD
    response = requests.get(url, timeout=20)
=======
<<<<<<< HEAD
    response = requests.get(url)
=======
    response = requests.get(url, timeout=20)
>>>>>>> 3d25aabb8347edce5a9e182f4e9b2083f9459fc4
>>>>>>> feature-fastapi
    path = Path(path)
    return extract_zip(response.content, path)


def extract_zip(zip_file: bytes, destination_directory: Path) -> Path:
    """Извлекает содержимое zip-файла в папку destination_directory
    :param zip_file: файл с zip-архивом
    :param destination_directory: место разархивирования
    :return: место разархивирования
    """
    destination_directory.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
    with zipfile.ZipFile(io.BytesIO(zip_file)) as file:
        file.extractall(destination_directory)
=======
<<<<<<< HEAD
    file = zipfile.ZipFile(io.BytesIO(zip_file))
    file.extractall(destination_directory)
=======
    with zipfile.ZipFile(io.BytesIO(zip_file)) as file:
        file.extractall(destination_directory)
>>>>>>> 3d25aabb8347edce5a9e182f4e9b2083f9459fc4
>>>>>>> feature-fastapi
    return destination_directory
