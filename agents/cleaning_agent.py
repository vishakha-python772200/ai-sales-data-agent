"""
agents/cleaning_agent.py
--------------------------
Orchestrates data cleaning by CALLING the pure functions in
preprocessing/data_cleaning.py. This agent itself contains NO cleaning
logic — it only decides when to run it, logs progress, and saves output.
"""

import pandas as pd

from agents.base_agent import BaseAgent
from preprocessing.data_cleaning import clean_data
from preprocessing.missing_values import get_missing_summary
from preprocessing.duplicate_handler import count_duplicates
from utils.file_handler import save_csv
from utils.validator import validate_dataframe_not_empty
import config


class CleaningAgent(BaseAgent):
    """Cleans raw data using preprocessing/data_cleaning.py pipeline."""

    def __init__(self):
        super().__init__(name="CleaningAgent")

    def run(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        validate_dataframe_not_empty(df, context="CleaningAgent input")

        # Log state BEFORE cleaning (uses preprocessing helper functions)
        dup_count = count_duplicates(df)
        missing_summary = get_missing_summary(df)
        self.logger.info(f"Before cleaning: {dup_count} duplicate rows, "
                          f"{len(missing_summary)} columns with missing values") 

        # Actual cleaning delegated to preprocessing/data_cleaning.py
        cleaned_df = clean_data(
            df,
            columns_to_drop=config.COLUMNS_TO_DROP,
            numeric_fill_strategy="median",
            categorical_fill_strategy="mode",
            treat_outliers=True
        )

        output_path = save_path or config.CLEANED_DATA_FILE
        save_csv(cleaned_df, output_path)

        self.logger.info(f"Cleaning complete. Final shape: {cleaned_df.shape}")
        return cleaned_df
