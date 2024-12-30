import os
import sys
from loguru import logger

LOG_FOLDER = 'logs'

class InterceptHandler:  # pylint: disable=too-few-public-methods
    def write(self, message):
        """
        Интеграция loguru с uvicorn
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


def configure_client_logging():
    setup_logger(os.path.join(LOG_FOLDER, "client.log"))

def configure_server_logging():
    setup_logger(os.path.join(LOG_FOLDER, "server.log"))
    sys.stderr = InterceptHandler()
