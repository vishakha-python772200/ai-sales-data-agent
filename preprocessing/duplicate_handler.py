"""
preprocessing/duplicate_handler.py
--------------------------------------
Pure functions to detect and remove duplicate rows.
"""

import pandas as pd


def count_duplicates(df: pd.DataFrame, subset: list = None) -> int:
    """Returns number of duplicate rows (based on optional subset of columns)."""
    return int(df.duplicated(subset=subset).sum())


def remove_duplicates(df: pd.DataFrame, subset: list = None, keep: str = "first") -> pd.DataFrame:
    """
    Removes duplicate rows.

    Args:
        df: input DataFrame
        subset: list of columns to consider for duplicate check (None = all columns)
        keep: "first", "last", or False (drop all duplicates)

    Returns:
        DataFrame without duplicate rows, index reset.
    """
    df = df.copy()
    df = df.drop_duplicates(subset=subset, keep=keep)
    df.reset_index(drop=True, inplace=True)
    return df
