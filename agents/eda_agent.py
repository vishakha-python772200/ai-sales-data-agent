"""
agents/eda_agent.py
----------------------
Performs Exploratory Data Analysis: generates distribution plots,
a correlation heatmap, and saves everything into outputs/graphs/.
Also compiles a simple PDF report using matplotlib's PdfPages
(no extra system dependencies needed).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, safe for servers/streamlit pdf  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # this library used for to create a  data visulization pdf this is help for developer 
# multiple graph sathi useeful he  

from agents.base_agent import BaseAgent # this is useful for to call baseagent 
from utils.validator import validate_dataframe_not_empty # checking this if dataframe is empty ya not 
import config # this is used for configration values store 


class EDAAgent(BaseAgent):
    """Generates EDA plots and a consolidated PDF report."""

    def __init__(self):
        super().__init__(name="EDAAgent") # generated eda report and saved as pdf 

    def run(self, df: pd.DataFrame, report_path: str = None) -> str: 
        validate_dataframe_not_empty(df, context="EDAAgent input")

        os.makedirs(config.GRAPHS_DIR, exist_ok=True)  # Graphs save karanyasathi folder create karto. Folder already asel tar error yet nahi (exist_ok=True).

        report_path = report_path or config.EDA_REPORT_FILE
        os.makedirs(os.path.dirname(report_path), exist_ok=True) # pdf saved honyacha path geto ani to folder nasel new create karto 

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() # DataFrame madhil fakta numeric columns (int, float) select karto ani tyanchi names list madhye convert karto.
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        with PdfPages(report_path) as pdf:

            # ---- 1. Numeric distributions ----
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[col].hist(bins=30, ax=ax, color="#4C72B0", edgecolor="black")
                ax.set_title(f"Distribution: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                fig.tight_layout()
                fig.savefig(os.path.join(config.GRAPHS_DIR, f"dist_{col}.png"))
                pdf.savefig(fig)
                plt.close(fig)

            # ---- 2. Correlation heatmap ----
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                corr = df[numeric_cols].corr()
                im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(numeric_cols)))
                ax.set_yticks(range(len(numeric_cols)))
                ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
                ax.set_yticklabels(numeric_cols, fontsize=7)
                fig.colorbar(im, ax=ax)
                ax.set_title("Correlation Heatmap")
                fig.tight_layout()
                fig.savefig(os.path.join(config.GRAPHS_DIR, "correlation_heatmap.png"))
                pdf.savefig(fig)
                plt.close(fig)

            # ---- 3. Top categorical value counts ----
            for col in categorical_cols:
                top_vals = df[col].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(6, 4))
                top_vals.plot(kind="bar", ax=ax, color="#55A868")
                ax.set_title(f"Top values: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(os.path.join(config.GRAPHS_DIR, f"cat_{col}.png"))
                pdf.savefig(fig)
                plt.close(fig)

        self.logger.info(f"EDA report saved to: {report_path}")
        self.logger.info(f"EDA graphs saved to: {config.GRAPHS_DIR}")
        return report_path
