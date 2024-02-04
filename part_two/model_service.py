import pickle as pk

from pathlib import Path

from model import build_model


class ModelService():
    def __init__(self):
        self.model = None

    def load_model(self, model_name="rf_v1"):
        model_path = Path(f"../part_two/models/{model_name}")

        if not model_path.exists():
            build_model()

        self.model = pk.load(open(f"../part_two/models/{model_name}", "rb"))

    def predict(self, input_parameters):
        return self.model.predict([input_parameters])
