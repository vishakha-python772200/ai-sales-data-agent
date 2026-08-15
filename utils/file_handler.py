"""
utils/file_handler.py
----------------------
Safe, reusable file IO helpers (CSV, JSON, pickle) used by all agents.
"""

import json
import os
import pickle
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file safely with logging."""
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded CSV: {filepath} | shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV at {filepath}: {e}")
        raise


def save_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """Save a DataFrame to CSV, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=index)
        logger.info(f"Saved CSV: {filepath} | shape={df.shape}")
    except Exception as e:
        logger.error(f"Failed to save CSV at {filepath}: {e}")
        raise


def save_json(data: dict, filepath: str) -> None:
    """Save a dictionary as JSON."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"Saved JSON: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON at {filepath}: {e}")
        raise


def read_json(filepath: str) -> dict:
    """Read a JSON file into a dictionary."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to read JSON at {filepath}: {e}")
        raise


def save_pickle(obj, filepath: str) -> None:
    """Save any Python object (e.g. trained model) as a pickle file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle object: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save pickle at {filepath}: {e}")
        raise


def load_pickle(filepath: str):
    """Load a pickled Python object."""
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded pickle object: {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load pickle at {filepath}: {e}")
        raise
