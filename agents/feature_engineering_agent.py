"""
agents/feature_engineering_agent.py
--------------------------------------
Orchestrates feature creation by CALLING pure functions in
preprocessing/feature_engineering.py. No feature-creation logic lives
here directly  only orchestration + logging + saving.

Add new domain-specific features by adding a function to
preprocessing/feature_engineering.py and calling it below.
"""

import pandas as pd

from agents.base_agent import BaseAgent
from preprocessing.feature_engineering import (
    extract_date_features, # month and day donhi sarkhe features create karto
    create_ratio_feature, # creating a numeric columns ratio features
    create_interaction_feature, # Don columns multiply karun interaction feature create karto
)
from utils.file_handler import save_csv
from utils.validator import validate_dataframe_not_empty
import config


class FeatureEngineeringAgent(BaseAgent):
    """Adds engineered features using preprocessing/feature_engineering.py."""

    def __init__(self):
        super().__init__(name="FeatureEngineeringAgent")

    def run(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        validate_dataframe_not_empty(df, context="FeatureEngineeringAgent input")
        df = df.copy()

        # 1. Date-based features (auto-detects any column with "date" in its name)
        before_cols = set(df.columns)
        df = extract_date_features(df)
        new_date_cols = set(df.columns) - before_cols
        if new_date_cols:
            self.logger.info(f"Extracted date features: {sorted(new_date_cols)}")

        # 2. Domain-specific ratio feature (only created if both columns exist)
        if "profit" in df.columns and "sales" in df.columns:
            df = create_ratio_feature(df, "profit", "sales", "profit_margin")
            self.logger.info("Created 'profit_margin' feature.")

        # 3. Domain-specific interaction feature (only created if both columns exist)
        if "quantity" in df.columns and "price" in df.columns:
            df = create_interaction_feature(df, "quantity", "price", "quantity_price_interaction")
            self.logger.info("Created 'quantity_price_interaction' feature.")

        output_path = save_path or config.PROCESSED_DATA_FILE
        save_csv(df, output_path)

        self.logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
