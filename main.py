"""
main.py
---------
Pipeline Orchestrator. Runs all agents IN ORDER, passing the output of
one agent as the input of the next. This is the single entry point for
running the full ML pipeline from the command line:

    python main.py

Each agent is called via .execute() (never .run() directly) so that
errors are caught, logged, and timed automatically by BaseAgent.
"""


import sys
import pandas as pd

import config
from utils.logger import get_logger
from utils.file_handler import read_csv

from agents.data_summary_agent import DataSummaryAgent
from agents.cleaning_agent import CleaningAgent
from agents.eda_agent import EDAAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.encoding_agent import EncodingAgent
from agents.feature_selection_agent import FeatureSelectionAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.prediction_agent import PredictionAgent
from agents.insight_agent import InsightAgent
from agents.report_agent import ReportAgent
from agents.export_agent import ExportAgent

logger = get_logger("main")


def _force_numeric_safety_net(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Safety net: forces every feature column (everything except the target)
    to numeric. Any stray garbage value that slipped past cleaning/encoding
    (e.g. a stray backslash or other unparseable string) becomes NaN, then
    gets filled with that column's median so training never crashes on it.
    """
    df = df.copy()
    for col in df.columns:
        if col == target_column:
            continue
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().any():
                df[col] = coerced

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def run_pipeline(raw_data_path: str = None, target_column: str = None) -> dict:
    """
    Runs the full pipeline end-to-end.

    Args:
        raw_data_path: path to the raw CSV. Defaults to config.RAW_DATA_FILE.
        target_column: name of the target column. Defaults to config.TARGET_COLUMN.

    Returns:
        dict with all intermediate results, or {"success": False, "error": ...}
        if any stage fails (pipeline stops at the first failure).
    """
    raw_data_path = raw_data_path or config.RAW_DATA_FILE
    target_column = target_column or config.TARGET_COLUMN

    pipeline_results = {}

    # ---------------------------------------------------------------
    # STAGE 0: Load raw data
    # ---------------------------------------------------------------
    try:
        df = read_csv(raw_data_path)
    except Exception as e:
        logger.error(f"Pipeline stopped: could not read raw data. {e}")
        return {"success": False, "error": str(e), "stage": "load_raw_data"}

    # ---------------------------------------------------------------
    # STAGE 1: Data Summary
    # ---------------------------------------------------------------
    r = DataSummaryAgent().execute(df)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "DataSummaryAgent"}
    pipeline_results["data_summary"] = r["result"]

    # ---------------------------------------------------------------
    # STAGE 2: Cleaning
    # ---------------------------------------------------------------
    r = CleaningAgent().execute(df)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "CleaningAgent"}
    cleaned_df = r["result"]

    # Save cleaned data to disk + record path so the UI can offer a download
    cleaned_df.to_csv(config.CLEANED_DATA_FILE, index=False)
    pipeline_results["cleaned_data_path"] = config.CLEANED_DATA_FILE

    # ---------------------------------------------------------------
    # STAGE 3: EDA
    # ---------------------------------------------------------------
    r = EDAAgent().execute(cleaned_df)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "EDAAgent"}
    pipeline_results["eda_report_path"] = r["result"]

    # ---------------------------------------------------------------
    # STAGE 4: Feature Engineering
    # ---------------------------------------------------------------
    r = FeatureEngineeringAgent().execute(cleaned_df)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "FeatureEngineeringAgent"}
    fe_df = r["result"]

    # Drop raw datetime columns (already converted into year/month/day/weekday)
    datetime_cols = fe_df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    if datetime_cols:
        fe_df = fe_df.drop(columns=datetime_cols)

    # ---------------------------------------------------------------
    # STAGE 5: Encoding
    # ---------------------------------------------------------------
    r = EncodingAgent().execute(fe_df)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "EncodingAgent"}
    encoded_df = r["result"]

    # ---------------------------------------------------------------
    # STAGE 6: Feature Selection
    # ---------------------------------------------------------------
    r = FeatureSelectionAgent().execute(encoded_df, target_column=target_column)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "FeatureSelectionAgent"}
    selected_df = r["result"]

    # ---------------------------------------------------------------
    # SAFETY NET: force every feature column to numeric before modeling.
    # Catches any stray non-numeric garbage (e.g. a lone backslash) that
    # slipped past cleaning/encoding and would otherwise crash TrainingAgent.
    # ---------------------------------------------------------------
    selected_df = _force_numeric_safety_net(selected_df, target_column)

    # ---------------------------------------------------------------
    # STAGE 7: Model Selection
    # ---------------------------------------------------------------
    r = ModelSelectionAgent().execute(selected_df, target_column=target_column)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "ModelSelectionAgent"}
    ms_result = r["result"]

    # ---------------------------------------------------------------
    # STAGE 8: Training
    # ---------------------------------------------------------------
    r = TrainingAgent().execute(ms_result["best_model_name"], ms_result["X_train"], ms_result["y_train"])
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "TrainingAgent"}
    train_result = r["result"]

    # ---------------------------------------------------------------
    # STAGE 9: Evaluation
    # ---------------------------------------------------------------
    r = EvaluationAgent().execute(train_result["model"], ms_result["X_test"], ms_result["y_test"])
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "EvaluationAgent"}
    metrics = r["result"]

    # ---------------------------------------------------------------
    # STAGE 10: Insights
    # ---------------------------------------------------------------
    r = InsightAgent().execute(train_result["model"], list(ms_result["X_train"].columns), metrics)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "InsightAgent"}
    insights = r["result"]

    # ---------------------------------------------------------------
    # STAGE 11: Final Report
    # ---------------------------------------------------------------
    r = ReportAgent().execute(ms_result["scores"], metrics, insights)
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "ReportAgent"}
    pipeline_results["model_report_path"] = r["result"]

    # ---------------------------------------------------------------
    # STAGE 12: Export
    # ---------------------------------------------------------------
    r = ExportAgent().execute()
    if not r["success"]:
        return {"success": False, "error": r["error"], "stage": "ExportAgent"}
    pipeline_results["export_path"] = r["result"]

    pipeline_results.update({
        "success": True,
        "metrics": metrics,
        "insights": insights,
        "best_model_name": ms_result["best_model_name"],
        "model_scores": ms_result["scores"],
    })

    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    return pipeline_results


if __name__ == "__main__":
    result = run_pipeline()

    if result.get("success"):
        print("\nPipeline finished successfully!")
        print(f"Best model     : {result['best_model_name']}")
        print(f"Metrics        : {result['metrics']}")
        print(f"Report saved to: {result['model_report_path']}")
        print(f"Export zip     : {result['export_path']}")
        sys.exit(0)
    else:
        print(f"\nPipeline FAILED at stage: {result.get('stage')}")
        print(f"Error: {result.get('error')}")
        sys.exit(1)
