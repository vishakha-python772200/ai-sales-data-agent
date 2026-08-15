"""
preprocessing/data_cleaning.py
------------------------------------
Combines missing_values, duplicate_handler, and outlier_handler into
ONE reusable cleaning pipeline function. agents/cleaning_agent.py calls
this single function instead of repeating the logic itself.
"""

import pandas as pd

from preprocessing.missing_values import fill_missing_numeric, fill_missing_categorical
from preprocessing.duplicate_handler import remove_duplicates
from preprocessing.outlier_handler import cap_outliers_iqr


def clean_data(df: pd.DataFrame, columns_to_drop: list = None,
                numeric_fill_strategy: str = "median",
                categorical_fill_strategy: str = "mode",
                treat_outliers: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline: drop columns -> remove duplicates ->
    fill missing values -> cap outliers.

    Args:
        df: raw input DataFrame
        columns_to_drop: list of column names to remove (IDs, names, etc.)
        numeric_fill_strategy: "median", "mean", or "zero"
        categorical_fill_strategy: "mode" or "constant"
        treat_outliers: whether to cap numeric outliers using IQR

    Returns:
        Cleaned DataFrame, index reset.
    """
    df = df.copy()

    # 1. Drop unwanted columns
    if columns_to_drop:
        existing = [c for c in columns_to_drop if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

    # 2. Remove duplicate rows
    df = remove_duplicates(df)

    # 3. Fill missing values
    df = fill_missing_numeric(df, strategy=numeric_fill_strategy)
    df = fill_missing_categorical(df, strategy=categorical_fill_strategy)

    # 4. Cap outliers
    if treat_outliers:
        df = cap_outliers_iqr(df)

    df.reset_index(drop=True, inplace=True)
    return df
