"""
agents/evaluation_agent.py
------------------------------
Evaluates the trained model on the held-out test set and produces
task-appropriate metrics (regression vs classification).
"""

import numpy as np # for numerical value handling sathi 
import pandas as pd

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # this is all model evalutions techniques this techniques used after train our model 
    
)

from agents.base_agent import BaseAgent
from utils.file_handler import save_json
import config


class EvaluationAgent(BaseAgent):
    """Computes evaluation metrics for the trained model on test data."""

    def __init__(self):
        super().__init__(name="EvaluationAgent")

    def run(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        predictions = model.predict(X_test)

        if config.TASK_TYPE == "classification":
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            }
        else:
            mse = mean_squared_error(y_test, predictions)
            metrics = {
                "r2_score": round(float(r2_score(y_test, predictions)), 4),
                "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
                "rmse": round(float(np.sqrt(mse)), 4),
            }

        self.logger.info(f"Evaluation metrics ({config.TASK_TYPE}): {metrics}")

        save_json(metrics, f"{config.REPORTS_DIR}/evaluation_metrics.json")
        return metrics
