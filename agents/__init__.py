"""
agents package
----------------
Exposes all agent classes for easy importing in main.py:

    from agents import (
        DataSummaryAgent, CleaningAgent, EDAAgent,
        FeatureEngineeringAgent, EncodingAgent, FeatureSelectionAgent,
        ModelSelectionAgent, TrainingAgent, EvaluationAgent,
        PredictionAgent, InsightAgent, ReportAgent, ExportAgent
    )
"""

from agents.base_agent import BaseAgent
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

__all__ = [
    "BaseAgent",
    "DataSummaryAgent",
    "CleaningAgent",
    "EDAAgent",
    "FeatureEngineeringAgent",
    "EncodingAgent",
    "FeatureSelectionAgent",
    "ModelSelectionAgent",
    "TrainingAgent",
    "EvaluationAgent",
    "PredictionAgent",
    "InsightAgent",
    "ReportAgent",
    "ExportAgent",
]
