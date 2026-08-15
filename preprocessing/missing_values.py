"""
preprocessing/missing_values.py
----------------------------------
Pure, reusable functions to handle missing values.
No agent/logger dependency here on purpose — these can be imported and
tested standalone (e.g. in a notebook or pytest) without loading the
whole agent framework.
"""

import pandas as pd


def fill_missing_numeric(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Fill missing values in numeric columns.

    Args:
        df: input DataFrame
        strategy: "median", "mean", or "zero"

    Returns:
        DataFrame with numeric NaNs filled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == "median":
                fill_value = df[col].median()
            elif strategy == "mean":
                fill_value = df[col].mean()
            elif strategy == "zero":
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy '{strategy}'. Use 'median', 'mean', or 'zero'.")
            df[col] = df[col].fillna(fill_value)

    return df


def fill_missing_categorical(df: pd.DataFrame, strategy: str = "mode") -> pd.DataFrame:
    """
    Fill missing values in categorical/object columns.

    Args:
        df: input DataFrame
        strategy: "mode" (most frequent value) or "constant"

    Returns:
        DataFrame with categorical NaNs filled.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if df[col].isnull().any():
            if strategy == "mode":
                mode_val = df[col].mode()
                fill_value = mode_val[0] if not mode_val.empty else "Unknown"
            elif strategy == "constant":
                fill_value = "Unknown"
            else:
                raise ValueError(f"Unknown strategy '{strategy}'. Use 'mode' or 'constant'.")
            df[col] = df[col].fillna(fill_value)

    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a small summary table: column, missing_count, missing_percent."""
    summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2)
    })
    return summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
