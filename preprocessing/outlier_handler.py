"""
preprocessing/outlier_handler.py
------------------------------------
Pure functions to detect and treat outliers in numeric columns.
Supports IQR-capping (default, safest for production) and z-score removal.
"""

import numpy as np
import pandas as pd


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Returns a boolean Series marking which rows are outliers (IQR method)."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[column] < lower) | (df[column] > upper)


def cap_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Caps (clips) outliers to the IQR bounds instead of deleting rows.
    This is the safest default for production — no data loss, no row misalignment.

    Args:
        df: input DataFrame
        columns: list of numeric columns to treat (None = all numeric columns)

    Returns:
        DataFrame with outliers capped.
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def remove_outliers_zscore(df: pd.DataFrame, columns: list = None, threshold: float = 3.0) -> pd.DataFrame:
    """
    Removes rows where any specified column has a z-score beyond the threshold.
    Use with caution in production — this DELETES rows (row count changes).

    Args:
        df: input DataFrame
        columns: list of numeric columns to check (None = all numeric columns)
        threshold: z-score cutoff (default 3.0)

    Returns:
        DataFrame with outlier rows removed, index reset.
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()

    mask = pd.Series(True, index=df.index)
    for col in columns:
        std = df[col].std()
        if std == 0 or pd.isna(std):
            continue
        z_scores = (df[col] - df[col].mean()) / std
        mask &= z_scores.abs() <= threshold

    result = df[mask].reset_index(drop=True)
    return result
