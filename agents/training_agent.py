"""
agents/training_agent.py
----------------------------
Takes the best_model_name chosen by ModelSelectionAgent, fits it on the
FULL training set, and saves the trained model + metadata to disk so it
can be reused later by PredictionAgent without retraining.
"""

import json
from datetime import datetime

import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from agents.base_agent import BaseAgent
from utils.file_handler import save_pickle, save_json
import config

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


class TrainingAgent(BaseAgent):
    """Fits the chosen best model on the full training set and saves it."""

    def __init__(self):
        super().__init__(name="TrainingAgent")

    def _instantiate_model(self, model_name: str):
        registry = CLASSIFICATION_MODELS if config.TASK_TYPE == "classification" else REGRESSION_MODELS
        if model_name not in registry:
            raise ValueError(f"Unknown model name '{model_name}' for task_type='{config.TASK_TYPE}'")

        model_class = registry[model_name]
        try:
            return model_class(random_state=config.RANDOM_STATE)
        except TypeError:
            # Some models (e.g. LinearRegression) don't accept random_state
            return model_class()

    def run(self, best_model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
            model_path: str = None) -> dict:

        model = self._instantiate_model(best_model_name)
        self.logger.info(f"Training {best_model_name} on {X_train.shape[0]} rows, {X_train.shape[1]} features...")

        model.fit(X_train, y_train)

        model_path = model_path or config.MODEL_FILE
        save_pickle(model, model_path)

        metadata = {
            "model_name": best_model_name,
            "task_type": config.TASK_TYPE,
            "trained_at": datetime.now().isoformat(),
            "n_features": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "n_training_rows": X_train.shape[0],
        }
        save_json(metadata, config.MODEL_METADATA_FILE)

        self.logger.info(f"Model trained and saved to {model_path}")
        return {"model": model, "model_path": model_path, "metadata": metadata}
