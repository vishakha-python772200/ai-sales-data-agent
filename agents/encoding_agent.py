"""
agents/encoding_agent.py
--------------------------
Orchestrates categorical encoding by CALLING pure functions in
preprocessing/encoding.py. No encoding logic lives here directly 
only orchestration + logging + persisting the encoders.
this files used for saved encoding 
"""

import pandas as pd

from agents.base_agent import BaseAgent 
from preprocessing.encoding import encode_categorical_columns # categorical function  import karnar  ecode import kela 
from utils.file_handler import save_csv, save_pickle # process dataframe pkl and csv file madhe saved karto 
from utils.validator import validate_dataframe_not_empty # input dataframe empty ahey ki nahi he check karyasathi  creating function 
import config


class EncodingAgent(BaseAgent):
    """Encodes categorical features using preprocessing/encoding.py staring encoding from here ."""

    def __init__(self):
        super().__init__(name="EncodingAgent")

    def run(self, df: pd.DataFrame, save_path: str = None,
            encoders_path: str = None) -> pd.DataFrame:
        validate_dataframe_not_empty(df, context="EncodingAgent input")

        # Never encode the target column itself
        encoded_df, encoders = encode_categorical_columns(
            df, exclude_columns=[config.TARGET_COLUMN]
        )

        for col, info in encoders.items():
             # Logger final message print karto.
            # encoded_df.shape
            # tuple return karte.
            # Example:
            # (1000, 18)
            # 1000 = rows
            # 18 = columns
            self.logger.info(f"Encoded '{col}' using {info['type']} encoding.")

        encoders_path = encoders_path or f"{config.SAVED_MODELS_DIR}/encoders.pkl"
        save_pickle(encoders, encoders_path)

        output_path = save_path or config.PROCESSED_DATA_FILE
        save_csv(encoded_df, output_path)

        self.logger.info(f"Encoding complete. Final shape: {encoded_df.shape}")
        return encoded_df
