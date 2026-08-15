"""
agents/prediction_agent.py
------------------------------
Loads the saved trained model (and encoders, if present) and generates
predictions on new/unseen data. Uses preprocessing/encoding.py's
apply_saved_encoders() to keep encoding IDENTICAL to training time.
"""

import os
import pandas as pd

from agents.base_agent import BaseAgent
from preprocessing.encoding import apply_saved_encoders
from utils.file_handler import load_pickle, save_csv
from utils.validator import validate_dataframe_not_empty
import config


class PredictionAgent(BaseAgent):
    """Generates predictions on new data using the saved trained model."""

    def __init__(self):
        super().__init__(name="PredictionAgent")

    def run(self, new_df: pd.DataFrame, model_path: str = None,
            save_path: str = None) -> pd.DataFrame:
        validate_dataframe_not_empty(new_df, context="PredictionAgent input")

        model_path = model_path or config.MODEL_FILE
        model = load_pickle(model_path)

        df = new_df.copy()

        # Apply the SAME encoders that were fitted during training
        encoders_path = f"{config.SAVED_MODELS_DIR}/encoders.pkl"
        if os.path.exists(encoders_path):
            encoders = load_pickle(encoders_path)
            df = apply_saved_encoders(df, encoders)
            self.logger.info(f"Applied {len(encoders)} saved encoder(s) to new data.")

        # Align columns with what the model expects (fill missing with 0)
        expected_cols = getattr(model, "feature_names_in_", None)
        if expected_cols is not None:
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[list(expected_cols)]

        predictions = model.predict(df)
        result_df = new_df.copy()
        result_df["prediction"] = predictions

        output_path = save_path or config.PREDICTIONS_FILE
        save_csv(result_df, output_path)

        self.logger.info(f"Predictions generated for {len(result_df)} rows -> {output_path}")
        return result_df
