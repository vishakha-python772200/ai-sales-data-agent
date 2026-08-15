"""
preprocessing/skewness_handler.py
--------------------------------------
Pure functions to detect and fix skewed numeric distributions.
Highly skewed features can hurt linear models, so we offer log/sqrt/
box-cox style transforms.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew


def get_skewness(df: pd.DataFrame, columns: list = None) -> pd.Series:
    """Returns skewness score per numeric column."""
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()
    scores = {col: float(scipy_skew(df[col].dropna())) for col in columns if df[col].dropna().shape[0] > 0}
    return pd.Series(scores).sort_values(ascending=False)


def fix_skewness(df: pd.DataFrame, columns: list = None, threshold: float = 1.0) -> pd.DataFrame:
    """
    Applies log1p transform to numeric columns whose skewness exceeds the threshold.
    Only works on non-negative columns (log1p requires values > -1).

    Args:
        df: input DataFrame
        columns: columns to check (None = all numeric columns)
        threshold: absolute skewness above which a transform is applied (default 1.0)

    Returns:
        DataFrame with skewed columns log-transformed. Original column values
        are replaced (call get_skewness() first if you want to inspect before/after).
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=np.number).columns.tolist()

    for col in columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        col_skew = scipy_skew(col_data)

        if abs(col_skew) > threshold and (df[col] >= 0).all():
            df[col] = np.log1p(df[col])

    return df
