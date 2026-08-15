"""
preprocessing/feature_engineering.py
----------------------------------------
Pure functions to create new features from existing columns.
Used by agents/feature_engineering_agent.py — edit/add functions here
for your domain-specific features (e.g. sales-specific ratios).
"""

import numpy as np
import pandas as pd


def extract_date_features(df: pd.DataFrame, date_columns: list = None) -> pd.DataFrame:
    """
    Extracts year/month/day/weekday from datetime-like columns.
    If date_columns is None, auto-detects columns with "date" in their name.

    Returns:
        DataFrame with new *_year, *_month, *_day, *_weekday columns added.
    """
    df = df.copy()

    if date_columns is None:
        date_columns = [c for c in df.columns if "date" in c.lower()]

    for col in date_columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday

    return df


def create_ratio_feature(df: pd.DataFrame, numerator_col: str, denominator_col: str,
                          new_col_name: str) -> pd.DataFrame:
    """
    Safely creates a ratio feature (numerator / denominator), avoiding
    division-by-zero errors by returning 0 where denominator is 0.
    """
    df = df.copy()
    if numerator_col not in df.columns or denominator_col not in df.columns:
        return df

    df[new_col_name] = np.where(
        df[denominator_col] != 0,
        df[numerator_col] / df[denominator_col],
        0
    )
    return df


def create_interaction_feature(df: pd.DataFrame, col_a: str, col_b: str,
                                new_col_name: str) -> pd.DataFrame:
    """Creates a simple multiplicative interaction feature: col_a * col_b."""
    df = df.copy()
    if col_a not in df.columns or col_b not in df.columns:
        return df

    df[new_col_name] = df[col_a] * df[col_b]
    return df
