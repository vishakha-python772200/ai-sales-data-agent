"""
agents/export_agent.py
---------------------------
Final stage: gathers all pipeline outputs (cleaned data, predictions,
trained model, reports) and copies them into the outputs/ folder for
easy delivery/download. Also creates a single zip archive.
"""

import os
import shutil
from datetime import datetime

from agents.base_agent import BaseAgent
import config


class ExportAgent(BaseAgent):
    """Collects all final artifacts into outputs/ and zips them."""

    def __init__(self):
        super().__init__(name="ExportAgent")

    def _safe_copy(self, src: str, dest_dir: str):
        if os.path.exists(src):
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src, dest_dir)
            self.logger.info(f"Copied {src} -> {dest_dir}")
        else:
            self.logger.warning(f"Skipped export, file not found: {src}")

    def run(self, zip_output: bool = True) -> str:
        # 1. Cleaned data
        self._safe_copy(config.CLEANED_DATA_FILE, config.CLEANED_CSV_DIR)

        # 2. Predictions
        self._safe_copy(config.PREDICTIONS_FILE, config.PREDICTIONS_DIR)

        # 3. Trained model + metadata
        model_export_dir = os.path.join(config.OUTPUTS_DIR, "model")
        self._safe_copy(config.MODEL_FILE, model_export_dir)
        self._safe_copy(config.MODEL_METADATA_FILE, model_export_dir)

        # 4. Reports
        reports_export_dir = os.path.join(config.OUTPUTS_DIR, "reports")
        for report_file in [config.EDA_REPORT_FILE, config.INSIGHT_REPORT_FILE, config.MODEL_REPORT_FILE]:
            self._safe_copy(report_file, reports_export_dir)

        result_path = config.OUTPUTS_DIR

        if zip_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_base_name = os.path.join(config.BASE_DIR, f"pipeline_export_{timestamp}")
            zip_path = shutil.make_archive(zip_base_name, "zip", config.OUTPUTS_DIR)
            self.logger.info(f"Created export zip: {zip_path}")
            result_path = zip_path

        self.logger.info("Export complete. All artifacts available in outputs/")
        return result_path
