import pickle
import pandas as pd

class DiabetesModel:
    def __init__(self):
        # Load trained model and feature list
        self.model = self.load_model()
        self.features = self.load_features()

    def load_model(self):
        """Load the pre-trained XGBoost model from pickle file."""
        with open("model.pkl", "rb") as f:
            return pickle.load(f)

    def load_features(self):
        """Load the saved feature order from pickle file."""
        with open("features.pkl", "rb") as f:
            return pickle.load(f)

    def predict(self, user_data):
        """
        Predict diabetes status given patient data.
        
        Parameters:
        - user_data: list of feature values in the same order as self.features

        Returns:
        - int: 1 if diabetic, 0 if non-diabetic
        """
        df = pd.DataFrame([user_data], columns=self.features)
        pred = self.model.predict(df)[0]
        return int(pred)

# Create a single instance for use in FastAPI
diabetes_model = DiabetesModel()
