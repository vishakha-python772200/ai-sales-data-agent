"""
config.py
----------
Central configuration file for the AI Agent pipeline.
All file paths, constants, and settings are defined here so that
no agent hardcodes a path. Change paths here only.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file (API keys, secrets) into environment
load_dotenv()

# ---------------------------------------------------------------------------
# API KEYS / SECRETS  (loaded from .env — never hardcode keys here)
# ---------------------------------------------------------------------------
# Optional: only needed if you plug in an LLM later (e.g. for AI-generated
# natural-language insights in insight_agent.py). Not required for the
# core ML pipeline (cleaning/EDA/training/prediction) to work.
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# ---------------------------------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# DATA PATHS
# ---------------------------------------------------------------------------
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PREDICTIONS_DATA_DIR = os.path.join(BASE_DIR, "data", "predictions")

RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "sales_dataset.csv")
CLEANED_DATA_FILE = os.path.join(CLEANED_DATA_DIR, "cleaned_sales_dataset.csv")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "processed_sales_dataset.csv")
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DATA_DIR, "predictions_output.csv")

# ---------------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------------
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "models", "trained_models")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")
MODEL_FILE = os.path.join(SAVED_MODELS_DIR, "best_model.pkl")
MODEL_METADATA_FILE = os.path.join(SAVED_MODELS_DIR, "model_metadata.json")

# ---------------------------------------------------------------------------
# REPORTS & OUTPUTS
# ---------------------------------------------------------------------------
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EDA_REPORT_FILE = os.path.join(REPORTS_DIR, "eda_report.pdf")
INSIGHT_REPORT_FILE = os.path.join(REPORTS_DIR, "insight_report.pdf")
MODEL_REPORT_FILE = os.path.join(REPORTS_DIR, "model_report.pdf")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
CLEANED_CSV_DIR = os.path.join(OUTPUTS_DIR, "cleaned_csv")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# PIPELINE SETTINGS
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Sales"          # <-- tumcha column name "Sales" (capital S) ahey
TEST_SIZE = 0.2
RANDOM_STATE = 42

# task_type: "regression" or "classification"
TASK_TYPE = "regression"

# columns to always drop (IDs, names, etc.) - edit as needed
COLUMNS_TO_DROP = []

# ---------------------------------------------------------------------------
# ENSURE ALL DIRECTORIES EXIST
# ---------------------------------------------------------------------------
_ALL_DIRS = [
    RAW_DATA_DIR, CLEANED_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DATA_DIR,
    TRAINED_MODELS_DIR, SAVED_MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR,
    GRAPHS_DIR, CLEANED_CSV_DIR, PREDICTIONS_DIR, LOGS_DIR
]

for _dir in _ALL_DIRS:
    os.makedirs(_dir, exist_ok=True)