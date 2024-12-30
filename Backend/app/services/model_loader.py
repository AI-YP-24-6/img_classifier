import os
import pickle

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Backend", "data", "baseline.pkl")


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == "HogTransformer":
            from Backend.app.services.pipeline import HogTransformer

            return HogTransformer
        return super().find_class(module, name)


def load_model():
    """
    Загрузка baseline-модели из pickle-файла
    """
    with open(MODEL_PATH, "rb") as file:
        model = CustomUnpickler(file)
    return model
