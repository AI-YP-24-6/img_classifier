from backend.app.services.pipeline import HogTransformer
import os
import pickle

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend', 'data', 'baseline.pkl')


def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model