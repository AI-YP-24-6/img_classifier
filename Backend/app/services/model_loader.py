import os
import pickle
from typing import Any

from Backend.app.services.pipeline import HogTransformer

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Backend", "data", "baseline.pkl")


class CustomUnpickler(pickle.Unpickler):
    """
    Класс обертка. Позволяет использовать в pickle-файле классы из других модулей
    """

    def find_class(self, module: str, name: str) -> HogTransformer | Any:
        """
        Проверка на соответствие классу и вызов метода pickle родителя
        :param module: модуль класса
        :param name: имя класса
        :return: или наш класс, или класс из pickle
        """
        if name == "HogTransformer":
            return HogTransformer

        return super().find_class(module, name)


def load_model():
    """
    Загрузка baseline-модели из pickle-файла
    """
    with open(MODEL_PATH, "rb") as file:
        model = CustomUnpickler(file).load()
    return model
