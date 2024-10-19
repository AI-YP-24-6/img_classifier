# Tools/__init__.py
"""
Общий пакет для готовых функций.
Модули в этом пакете:

- `download`: Позволяет скачать и разархивировать датасет.
- `parser`: Парсер сайта госкаталог.
"""
from .download import download_zip, extract_zip, get_ya_disk_url
from .parser import goskatalog_parser, zip_files
