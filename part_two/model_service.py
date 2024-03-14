import pickle as pk
import pandas as pd

from pathlib import Path

from model import build_model


class ModelService:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def load_model(self, model_name="rf_v1"):
        model_path = Path(f"../part_two/models/{model_name}")

        if not model_path.exists():
            build_model()

        self.model, self.feature_names = pk.load(open(f"../part_two/models/{model_name}", "rb"))

    def predict(self, input_parameters):
        # Check if feature names are available
        if self.feature_names is None:
            raise ValueError("Feature names are not available. Please ensure model is loaded correctly.")

        # Ensure input parameters are provided in the same order as feature names
        if len(input_parameters) != len(self.feature_names):
            raise ValueError("Number of input parameters does not match number of feature names.")

        # Create a dictionary with feature names as keys and input parameters as values
        input_data = {feature_name: parameter for feature_name, parameter in zip(self.feature_names, input_parameters)}

        # Convert the dictionary to a DataFrame with a single row
        input_df = pd.DataFrame(input_data, index=[0])

        # Predict using the model
        return self.model.predict(input_df)
