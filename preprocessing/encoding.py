"""
preprocessing/encoding.py
------------------------------
Pure functions for encoding categorical columns.
Used by agents/encoding_agent.py.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

LOW_CARDINALITY_THRESHOLD = 10


def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One-hot encodes a single column and drops the original."""
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
    df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    return df


def label_encode(df: pd.DataFrame, column: str) -> tuple:
    """
    Label-encodes a single column.

    Returns:
        (DataFrame with encoded column, fitted LabelEncoder instance)
    """
    df = df.copy()
    le = LabelEncoder()
    df[column] = df[column].astype(str)
    df[column] = le.fit_transform(df[column])
    return df, le


def encode_categorical_columns(df: pd.DataFrame, exclude_columns: list = None,
                                threshold: int = LOW_CARDINALITY_THRESHOLD) -> tuple:
    """
    Encodes all categorical columns automatically:
      - low-cardinality (<= threshold unique values) -> one-hot encoding
      - high-cardinality -> label encoding

    Args:
        df: input DataFrame
        exclude_columns: columns to skip (e.g. the target column)
        threshold: cutoff for choosing one-hot vs label encoding

    Returns:
        (encoded DataFrame, encoders dict) — encoders dict must be saved
        so PredictionAgent can apply the SAME encoding to new data later.
    """
    df = df.copy()
    exclude_columns = exclude_columns or []
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclude_columns]

    encoders = {}

    for col in categorical_cols:
        n_unique = df[col].nunique()

        if n_unique <= threshold:
            df = one_hot_encode(df, col)
            encoders[col] = {"type": "onehot"}
        else:
            df, le = label_encode(df, col)
            encoders[col] = {"type": "label", "encoder": le}

    return df, encoders


def apply_saved_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Applies previously-fitted encoders (from training time) to new/unseen data.
    Handles unseen categories gracefully by mapping them to the first known class.
    """
    df = df.copy()

    for col, enc_info in encoders.items():
        if col not in df.columns:
            continue

        if enc_info["type"] == "label":
            le = enc_info["encoder"]
            df[col] = df[col].astype(str)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            df[col] = le.transform(df[col])

        elif enc_info["type"] == "onehot":
            df = one_hot_encode(df, col)

    return df
