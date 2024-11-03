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

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()["href"]
    return download_url


def download_zip(url: str, path: str | Path) -> Path:
    """Скачивает zip-файл по ссылке и возвращает путь до него
    :param url: ссылка на zip-файл с Yandex.Disk
    :param path: место, куда нужно сохранить zip-файл
    :return: путь до загруженного разархивированного zip-файла
    """
    response = requests.get(url)
    path = Path(path)
    return extract_zip(response.content, path)


def extract_zip(zip_file: bytes, destination_directory: Path) -> Path:
    """Извлекает содержимое zip-файла в папку destination_directory
    :param zip_file: файл с zip-архивом
    :param destination_directory: место разархивирования
    :return: место разархивирования
    """
    destination_directory.mkdir(parents=True, exist_ok=True)

    file = zipfile.ZipFile(io.BytesIO(zip_file))
    file.extractall(destination_directory)
    return destination_directory
