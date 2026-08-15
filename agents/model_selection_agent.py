"""
agents/model_selection_agent.py
-----------------------------------
Splits the data into train/test sets and tries several candidate
algorithms using cross-validation, then picks the best one based on
task type (regression -> R2, classification -> accuracy).

The actual final training (fit on full training set) happens in
training_agent.py — this agent's job is ONLY to pick the winner.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from agents.base_agent import BaseAgent
from utils.file_handler import save_json
from utils.validator import validate_dataframe_not_empty, validate_target_column
import config

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=config.RANDOM_STATE),
    "RandomForestRegressor": RandomForestRegressor(random_state=config.RANDOM_STATE, n_estimators=100),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
    "RandomForestClassifier": RandomForestClassifier(random_state=config.RANDOM_STATE, n_estimators=100),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
}


class ModelSelectionAgent(BaseAgent):
    """Splits data and selects the best-performing candidate model."""

    def __init__(self):
        super().__init__(name="ModelSelectionAgent")

    def run(self, df: pd.DataFrame, target_column: str = None) -> dict:
        validate_dataframe_not_empty(df, context="ModelSelectionAgent input")
        target_column = target_column or config.TARGET_COLUMN
        validate_target_column(df, target_column)

        df = df.copy()
        # Safety net: drop any leftover non-numeric columns (should already be encoded)
        X = df.drop(columns=[target_column])
        X = X.select_dtypes(include=[np.number])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )

        candidates = CLASSIFICATION_MODELS if config.TASK_TYPE == "classification" else REGRESSION_MODELS
        scoring = "accuracy" if config.TASK_TYPE == "classification" else "r2"

        scores = {}
        for name, model in candidates.items():
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
                scores[name] = round(float(np.mean(cv_scores)), 4)
                self.logger.info(f"{name}: mean {scoring}={scores[name]}")
            except Exception as e:
                self.logger.warning(f"{name} failed during cross-validation: {e}")
                scores[name] = float("-inf")

        best_model_name = max(scores, key=scores.get)
        self.logger.info(f"Best model selected: {best_model_name} (score={scores[best_model_name]})")

        save_json(scores, f"{config.REPORTS_DIR}/model_selection_scores.json")

        return {
            "best_model_name": best_model_name,
            "scores": scores,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
