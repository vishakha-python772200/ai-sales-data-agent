"""
server.py
---------
Flask backend for the Pipeline UI.

It does three things:
1. Serves static/index.html
2. Accepts a CSV upload
3. Runs the pipeline in a background thread and exposes /api/status
   so the frontend can poll and animate stage-by-stage progress.

IMPORTANT — this file does NOT replace or rewrite main.py's pipeline
logic. It calls a single wrapper function, run_pipeline_for_ui(),
which you add ALONGSIDE your existing main.py code (see the block
at the bottom of this file for exactly what to add and where).
"""

import os
import threading
import traceback
import uuid

import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

import config           # your existing config.py
import main as pipeline  # your existing main.py

app = Flask(__name__, static_folder="static", static_url_path="")

UPLOAD_DIR = os.path.join(os.getcwd(), "data", "raw")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory run status. Single-run-at-a-time is enough for a local tool;
# if you need multiple concurrent runs later, key this dict by a run_id.
STATUS = {
    "state": "idle",       # idle | processing | success | error
    "stage_index": -1,
    "message": "",
    "results": None,
}

STAGE_COUNT = 12  # keep in sync with the STAGES array in index.html


# ---------------------------------------------------------------------
# Static page
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file"}), 400

    safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(save_path)
    return jsonify({"path": save_path})


# ---------------------------------------------------------------------
# Run pipeline (background thread + polling status)
# ---------------------------------------------------------------------
@app.route("/api/run", methods=["POST"])
def run():
    if STATUS["state"] == "processing":
        return jsonify({"error": "A pipeline run is already in progress"}), 409

    body = request.get_json(force=True) or {}
    csv_path = body.get("path")
    target_column = body.get("target_column")

    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"error": "Uploaded file not found on server"}), 400
    if not target_column:
        return jsonify({"error": "target_column is required"}), 400

    STATUS.update({"state": "processing", "stage_index": 0, "message": "Starting…", "results": None})

    thread = threading.Thread(target=_run_pipeline_thread, args=(csv_path, target_column), daemon=True)
    thread.start()
    return jsonify({"started": True})


def _stage_callback(index, name):
    """Passed into run_pipeline_for_ui so it can report progress as it goes."""
    STATUS["stage_index"] = index
    STATUS["message"] = f"Running: {name}"


def _run_pipeline_thread(csv_path, target_column):
    try:
        results = pipeline.run_pipeline_for_ui(
            csv_path=csv_path,
            target_column=target_column,
            on_stage=_stage_callback,
        )
        STATUS.update({
            "state": "success",
            "stage_index": STAGE_COUNT,
            "message": "Pipeline complete",
            "results": results,
        })
    except Exception as exc:
        traceback.print_exc()
        STATUS.update({
            "state": "error",
            "message": str(exc),
        })


@app.route("/api/status")
def status():
    return jsonify(STATUS)


# ---------------------------------------------------------------------
# Downloads — run_pipeline_for_ui() returns the real, already-saved file
# paths in results["downloads"], so this endpoint just serves whatever
# path it's given, restricted to the project folder for safety.
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.getcwd())


@app.route("/api/download-file")
def download_file():
    path = request.args.get("path")
    if not path or not os.path.abspath(path).startswith(PROJECT_ROOT):
        return jsonify({"error": "Invalid path"}), 400
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


# ---------------------------------------------------------------------
# Predict on a new CSV using the already-trained model
# ---------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True) or {}
    csv_path = body.get("path")
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"error": "File not found on server"}), 400

    try:
        predictions_df, output_path = pipeline.predict_for_ui(csv_path)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    preview = predictions_df.head(20).to_dict(orient="records")
    return jsonify({
        "preview": preview,
        "download": f"/api/download-file?path={output_path}",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)


# ---------------------------------------------------------------------
# ADD THIS TO main.py — paste it AFTER your existing run_pipeline()
# function (do not touch run_pipeline() itself). It reuses the exact
# same agents, in the exact same order, just with tick() calls added
# for the UI's progress bar, and it accepts a custom csv_path.
#
# Needs `import os` at the top of main.py if it isn't already there.
# ---------------------------------------------------------------------
"""
def run_pipeline_for_ui(csv_path, target_column, on_stage=None):
    stages = [
        "Ingest", "Clean", "EDA", "Feature Eng.", "Encode",
        "Select features", "Select model", "Train", "Evaluate",
        "Insights", "Report", "Export",
    ]

    def tick(i):
        if on_stage:
            on_stage(i, stages[i])

    pipeline_results = {}

    tick(0)
    try:
        df = read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Could not read raw data: {e}")

    r = DataSummaryAgent().execute(df)
    if not r["success"]:
        raise RuntimeError(f"DataSummaryAgent: {r['error']}")
    pipeline_results["data_summary"] = r["result"]

    tick(1)
    r = CleaningAgent().execute(df)
    if not r["success"]:
        raise RuntimeError(f"CleaningAgent: {r['error']}")
    cleaned_df = r["result"]
    cleaned_df.to_csv(config.CLEANED_DATA_FILE, index=False)
    pipeline_results["cleaned_data_path"] = config.CLEANED_DATA_FILE

    tick(2)
    r = EDAAgent().execute(cleaned_df)
    if not r["success"]:
        raise RuntimeError(f"EDAAgent: {r['error']}")
    pipeline_results["eda_report_path"] = r["result"]

    tick(3)
    r = FeatureEngineeringAgent().execute(cleaned_df)
    if not r["success"]:
        raise RuntimeError(f"FeatureEngineeringAgent: {r['error']}")
    fe_df = r["result"]
    datetime_cols = fe_df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    if datetime_cols:
        fe_df = fe_df.drop(columns=datetime_cols)

    tick(4)
    r = EncodingAgent().execute(fe_df)
    if not r["success"]:
        raise RuntimeError(f"EncodingAgent: {r['error']}")
    encoded_df = r["result"]

    tick(5)
    r = FeatureSelectionAgent().execute(encoded_df, target_column=target_column)
    if not r["success"]:
        raise RuntimeError(f"FeatureSelectionAgent: {r['error']}")
    selected_df = r["result"]

    tick(6)
    r = ModelSelectionAgent().execute(selected_df, target_column=target_column)
    if not r["success"]:
        raise RuntimeError(f"ModelSelectionAgent: {r['error']}")
    ms_result = r["result"]

    tick(7)
    r = TrainingAgent().execute(ms_result["best_model_name"], ms_result["X_train"], ms_result["y_train"])
    if not r["success"]:
        raise RuntimeError(f"TrainingAgent: {r['error']}")
    train_result = r["result"]

    tick(8)
    r = EvaluationAgent().execute(train_result["model"], ms_result["X_test"], ms_result["y_test"])
    if not r["success"]:
        raise RuntimeError(f"EvaluationAgent: {r['error']}")
    metrics = r["result"]

    tick(9)
    r = InsightAgent().execute(train_result["model"], list(ms_result["X_train"].columns), metrics)
    if not r["success"]:
        raise RuntimeError(f"InsightAgent: {r['error']}")
    insights = r["result"]

    tick(10)
    r = ReportAgent().execute(ms_result["scores"], metrics, insights)
    if not r["success"]:
        raise RuntimeError(f"ReportAgent: {r['error']}")
    model_report_path = r["result"]

    tick(11)
    r = ExportAgent().execute()
    if not r["success"]:
        raise RuntimeError(f"ExportAgent: {r['error']}")
    export_path = r["result"]

    # ---- shape it the way the UI expects ----
    metrics_out = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            metrics_out[k] = round(v, 4) if isinstance(v, float) else v

    insights_out = insights if isinstance(insights, list) else [str(insights)]

    return {
        "metrics": metrics_out,
        "insights": insights_out,
        "downloads": {
            "cleaned_csv": f"/api/download-file?path={pipeline_results['cleaned_data_path']}",
            "eda_report": f"/api/download-file?path={pipeline_results['eda_report_path']}",
            "model_report": f"/api/download-file?path={model_report_path}",
            "export": f"/api/download-file?path={export_path}",
        },
    }


def predict_for_ui(csv_path):
    df = read_csv(csv_path)
    r = PredictionAgent().execute(df)
    if not r["success"]:
        raise RuntimeError(f"PredictionAgent: {r['error']}")
    predictions_df = r["result"]

    output_path = os.path.join(os.path.dirname(config.CLEANED_DATA_FILE), "predictions.csv")
    predictions_df.to_csv(output_path, index=False)
    return predictions_df, output_path
"""