"""
agents/insight_agent.py
---------------------------
Turns model results into plain-language, business-readable insights:
 - top feature importances (if the model supports it)
 - simple textual summary of model performance
Saves insights as JSON so ReportAgent can pull them into the final PDF.
"""

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.file_handler import save_json
import config


class InsightAgent(BaseAgent):
    """Generates human-readable insights from the trained model and metrics."""

    def __init__(self):
        super().__init__(name="InsightAgent")

    def run(self, model, feature_names: list, metrics: dict) -> dict:
        insights = {"summary": [], "top_features": []}

        # 1. Feature importance (tree-based models) or coefficients (linear models)
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef).flatten() if hasattr(coef, "flatten") else np.abs(coef)

        if importances is not None and len(importances) == len(feature_names):
            ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            top_5 = ranked[:5]
            insights["top_features"] = [{"feature": f, "importance": round(float(v), 4)} for f, v in top_5]

            for feature, importance in top_5:
                insights["summary"].append(
                    f"'{feature}' has a strong influence on the prediction (importance score: {round(float(importance), 3)})."
                )
        else:
            insights["summary"].append("Feature importance is not available for this model type.")

        # 2. Performance summary
        if config.TASK_TYPE == "regression":
            r2 = metrics.get("r2_score")
            if r2 is not None:
                if r2 >= 0.8:
                    insights["summary"].append(f"Model explains {r2*100:.1f}% of the variance — strong fit.")
                elif r2 >= 0.5:
                    insights["summary"].append(f"Model explains {r2*100:.1f}% of the variance — moderate fit, consider more features.")
                else:
                    insights["summary"].append(f"Model only explains {r2*100:.1f}% of the variance — weak fit, review data quality or features.")
        else:
            acc = metrics.get("accuracy")
            if acc is not None:
                insights["summary"].append(f"Model achieved {acc*100:.1f}% accuracy on the test set.")

        self.logger.info(f"Generated {len(insights['summary'])} insight statements.")
        save_json(insights, f"{config.REPORTS_DIR}/insights.json")
        return insights
