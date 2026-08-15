"""
agents/data_summary_agent.py
this files create summary.
Generates a quick statistical & structural summary of the incoming dataset.
Runs right after data is loaded (raw or cleaned) to give a snapshot before
deeper processing begins.
# here we used json bcz json file  saved data nested formate naturallly support to dict format 
"""

import pandas as pd

from agents.base_agent import BaseAgent
from utils.file_handler import save_json # for saving  json file 
from utils.validator import validate_dataframe_not_empty # dataframe empty ahey ke nahi he check karysathi we can create this functions 
import config  # import kelyla values na configration access karysathi.


class DataSummaryAgent(BaseAgent): # new class created  from baseagent 
    """Produces a summary dict describing shape, dtypes, nulls, and stats."""

    def __init__(self):
        super().__init__(name="DataSummaryAgent")

    def run(self, df: pd.DataFrame, save_path: str = None) -> dict:# he input dataframe gheto and dictonery return karto
        validate_dataframe_not_empty(df, context="DataSummaryAgent input")

        summary = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]}, 
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().mean() * 100).round(2).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_summary": df.describe(include="number").round(2).to_dict(),
            "categorical_summary": {
                col: df[col].value_counts().head(5).to_dict()
                for col in df.select_dtypes(include=["object", "category"]).columns
            },
        }

        self.logger.info(
            f"Data summary generated: {summary['shape']['rows']} rows, "
            f"{summary['shape']['columns']} columns, "
            f"{summary['duplicate_rows']} duplicate rows found."
        )

        output_path = save_path or f"{config.REPORTS_DIR}/data_summary.json"
        save_json(summary, output_path)

        return summary
