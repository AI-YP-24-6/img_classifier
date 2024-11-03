# Tools/__init__.py
"""
Общий пакет для готовых функций.
Модули в этом пакете:

- `download`: Позволяет скачать и разархивировать датасет.
- `parser`: Парсер сайта госкаталог.
- `notebook`: Позволяет обновлять идентификаторы ячеек в ноутбуке.
- `analysis`: Позволяет находить все jpeg-файлы в директории.
"""
from .analysis import display_images, find_needed_jpeg_files
from .download import download_zip, extract_zip, get_ya_disk_url
from .notebook import set_cell_id
from .parser import goskatalog_parser, zip_files
