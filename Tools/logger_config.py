import logging
import os
import sys

from loguru import logger

LOG_FOLDER = "logs"


class InterceptHandler(logging.Handler):  # pylint: disable=too-few-public-methods
    """
    Интеграция loguru с uvicorn.
    Default handler from examples in loguru documentation.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    https://pawamoy.github.io/posts/unify-logging-for-a-gunicorn-uvicorn-app/
    """

    def emit(self, record: logging.LogRecord):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    def write(self, message):
        """
        Интеграция loguru со стандартным выходом ошибок
        """
        if message.strip():
            logger.info(message.strip())


def setup_logger(log_file=None):
    """
    Настройка loguru для записи логов.

    :param log_file: Путь к файлу для записи логов. Если None, логирование будет только в stdout.
    """
    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>|<level>{level}</level>| {message}",
    )

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(
            log_file,
            colorize=True,
            format="{time} | {level} | {message}",
            rotation="10 MB",
            retention="10 days",
            compression="zip",
        )


def configure_client_logging(log_folder=LOG_FOLDER):
    """Создание логгера для клиента"""
    setup_logger(os.path.join(log_folder, "client.log"))


def configure_server_logging(log_folder=LOG_FOLDER):
    """Создание логгера для сервера"""
    setup_logger(os.path.join(log_folder, "server.log"))
    # Удаляем все существующие хэндлеры
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    logging.basicConfig(
        handlers=[InterceptHandler()], level=logging.INFO  # Добавляем наш перехватчик в корневой логгер
    )
    sys.stderr = InterceptHandler()  # не все библиотеки используют логгеры. Sqlalchemy.exc использует sys.stderr
