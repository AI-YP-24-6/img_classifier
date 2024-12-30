import os
import pickle

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Backend", "data", "baseline.pkl")


def load_model():
    """
    Загрузка baseline-модели из pickle-файла
    """
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model
