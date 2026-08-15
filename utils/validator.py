"""
utils/validator.py
-------------------
Common validation helpers used across agents.
"""

import os
import pandas as pd


def validate_file_exists(filepath: str) -> None:
    """Raise FileNotFoundError with a clear message if file is missing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required file not found: {filepath}")


def validate_dataframe_not_empty(df: pd.DataFrame, context: str = "") -> None:
    """Raise ValueError if dataframe is empty or None."""
    if df is None or df.empty:
        raise ValueError(f"DataFrame is empty or None. Context: {context}")


def validate_columns_exist(df: pd.DataFrame, columns: list) -> None:
    """Raise ValueError if any required column is missing from the dataframe."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in DataFrame: {missing}")


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """Raise ValueError if target column is missing or fully null."""
    validate_columns_exist(df, [target_column])
    if df[target_column].isnull().all():
        raise ValueError(f"Target column '{target_column}' is entirely null.")
