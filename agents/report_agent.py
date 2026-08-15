"""
agents/report_agent.py
---------------------------
Compiles a final consolidated PDF report combining:
 - model selection scores
 - evaluation metrics
 - insights (top features, summary text)
Uses matplotlib's PdfPages so no extra system dependency is required.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from agents.base_agent import BaseAgent
import config


class ReportAgent(BaseAgent):
    """Builds the final model/insight PDF report."""

    def __init__(self):
        super().__init__(name="ReportAgent")

    def _add_text_page(self, pdf, title: str, lines: list):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis("off")
        ax.text(0.05, 0.95, title, fontsize=16, fontweight="bold", transform=ax.transAxes)

        y = 0.88
        for line in lines:
            ax.text(0.05, y, f"- {line}", fontsize=10, transform=ax.transAxes, wrap=True)
            y -= 0.04
            if y < 0.05:
                break

        pdf.savefig(fig)
        plt.close(fig)

    def run(self, model_scores: dict, metrics: dict, insights: dict,
            report_path: str = None) -> str:
        report_path = report_path or config.MODEL_REPORT_FILE
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with PdfPages(report_path) as pdf:
            # Page 1: Model selection scores
            score_lines = [f"{name}: {score}" for name, score in model_scores.items()]
            self._add_text_page(pdf, "Model Selection Scores", score_lines)

            # Page 2: Evaluation metrics
            metric_lines = [f"{k}: {v}" for k, v in metrics.items() if not isinstance(v, list)]
            self._add_text_page(pdf, "Evaluation Metrics", metric_lines)

            # Page 3: Insights
            self._add_text_page(pdf, "Key Insights", insights.get("summary", []))

        self.logger.info(f"Final model report saved to: {report_path}")
        return report_path
