"""
app.py
--------
Streamlit UI for the AI Agent pipeline — Apple-inspired design.
Run with:  streamlit run app.py
"""

import os
import time
import pandas as pd
import streamlit as st

import config
from utils.file_handler import save_csv
from main import run_pipeline
from agents.prediction_agent import PredictionAgent

st.set_page_config(page_title="Sales Data Agent", page_icon="◎", layout="centered")

if "app_state" not in st.session_state:
    st.session_state.app_state = "idle"
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "predict_result" not in st.session_state:
    st.session_state.predict_result = None

HUES = {"idle": 214, "processing": 32, "success": 152, "error": 6}
hue = HUES[st.session_state.app_state]

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# NOTE: every rule below is on its own single line with zero blank lines.
# Streamlit's markdown parser can split a raw <style> HTML block on blank
# lines, which leaks the CSS as visible page text instead of applying it.
css = f"""
<style>
:root {{ --hue: {hue}; --accent: hsl(var(--hue), 90%, 58%); --accent-soft: hsla(var(--hue), 90%, 58%, 0.14); --accent-glow: hsla(var(--hue), 90%, 58%, 0.45); }}
html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
.stApp {{ background: radial-gradient(ellipse 120% 60% at 50% -10%, #101018 0%, #000000 55%); color: #F5F5F7; }}
.hero-title {{ font-size: 42px; font-weight: 800; letter-spacing: -0.03em; text-align:center; margin-bottom: 6px; margin-top: 12px; }}
.hero-sub {{ text-align:center; color: rgba(245,245,247,0.6); font-size: 16px; margin-bottom: 22px; }}
.state-tag {{ display:flex; align-items:center; justify-content:center; gap:8px; margin: 0 auto 28px; font-family:'JetBrains Mono', monospace; font-size:12px; color: var(--accent); background: var(--accent-soft); border:1px solid rgba(255,255,255,0.09); padding: 7px 16px; border-radius:100px; width: fit-content; transition: all 0.5s ease; }}
.state-dot {{ width:6px; height:6px; border-radius:50%; background: var(--accent); box-shadow: 0 0 8px var(--accent-glow); }}
.glass-card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09); border-radius: 20px; padding: 28px 26px; margin-bottom: 20px; backdrop-filter: blur(20px); }}
.section-label {{ font-family:'JetBrains Mono', monospace; font-size:11px; color: rgba(245,245,247,0.4); text-transform:uppercase; letter-spacing:0.1em; margin-bottom: 16px; }}
.field-label {{ font-family:'JetBrains Mono', monospace; font-size:11px; color: rgba(245,245,247,0.45); text-transform:uppercase; letter-spacing:0.08em; margin: 18px 0 6px; }}
[data-testid="stFileUploader"] {{ border: 1.5px dashed rgba(255,255,255,0.16); border-radius: 16px; padding: 10px; background: rgba(255,255,255,0.02); transition: border-color 0.3s ease; }}
[data-testid="stFileUploader"]:hover {{ border-color: var(--accent); }}
[data-testid="stFileUploaderDropzone"] {{ background: transparent; }}
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span {{ color: rgba(245,245,247,0.55) !important; }}
.stTextInput input, .stTextInput input::placeholder {{ color: #F5F5F7 !important; }}
.stTextInput input {{ background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.14) !important; border-radius: 12px !important; }}
.stTextInput input:focus {{ border-color: var(--accent) !important; }}
[data-testid="stSelectbox"] > div > div {{ background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.14) !important; border-radius: 12px !important; color: #F5F5F7 !important; }}
[data-testid="stSelectbox"] label {{ color: rgba(245,245,247,0.6) !important; }}
[data-baseweb="popover"] {{ background: #14141a !important; }}
[data-baseweb="menu"] {{ background: #14141a !important; }}
[data-baseweb="menu"] li {{ color: #F5F5F7 !important; }}
[data-baseweb="menu"] li:hover {{ background: rgba(255,255,255,0.08) !important; }}
div.stButton > button, div.stDownloadButton > button {{ font-family: 'Inter', sans-serif; font-weight: 600; font-size: 15px; letter-spacing: -0.01em; color: white; background: linear-gradient(135deg, hsl(var(--hue), 90%, 58%), hsl(calc(var(--hue) + 30), 90%, 50%)); border: none; border-radius: 100px; padding: 12px 28px; box-shadow: 0 4px 24px -6px var(--accent-glow); transition: transform 0.2s cubic-bezier(.22,1,.36,1), box-shadow 0.4s ease; width: 100%; }}
div.stButton > button:hover, div.stDownloadButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 6px 30px -4px var(--accent-glow); color: white; }}
div.stButton > button:active, div.stDownloadButton > button:active {{ transform: scale(0.97); }}
div.stButton > button:disabled {{ opacity: 0.35; box-shadow: none; }}
div.stDownloadButton > button {{ background: linear-gradient(135deg, hsl(214, 90%, 58%), hsl(244, 90%, 50%)); }}
[data-testid="stMetric"] {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09); border-radius: 16px; padding: 16px 18px; }}
[data-testid="stMetricValue"] {{ background: linear-gradient(135deg, hsl(152,70%,60%), hsl(172,70%,50%)); -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 700; }}
.insight-line {{ padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 14px; color: rgba(245,245,247,0.75); display:flex; gap: 10px; }}
.insight-line:last-child {{ border-bottom: none; }}
.insight-marker {{ color: var(--accent); font-weight: 700; }}
.download-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }}
.mega-download {{ background: linear-gradient(135deg, hsla(152,70%,45%,0.18), hsla(172,70%,45%,0.10)); border: 1px solid hsla(152,70%,55%,0.35); border-radius: 20px; padding: 26px; text-align: center; margin-top: 8px; }}
.mega-download h3 {{ font-size: 18px; margin-bottom: 4px; }}
.mega-download p {{ color: rgba(245,245,247,0.6); font-size: 13px; margin-bottom: 18px; }}
hr {{ border-color: rgba(255,255,255,0.08); }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

state_labels = {"idle": "Idle", "processing": "Processing", "success": "Complete", "error": "Error"}
st.markdown('<div class="hero-title">Sales Data Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload raw sales data. Get it cleaned, analyzed, modeled — automatically.</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="state-tag"><span class="state-dot"></span>{state_labels[st.session_state.app_state]}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">01 · Upload</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Raw sales CSV", type=["csv"], label_visibility="collapsed")

target_column = None
preview_df = None

if uploaded_file is not None:
    preview_df = pd.read_csv(uploaded_file)
    st.dataframe(preview_df.head(5), use_container_width=True)

    st.markdown('<div class="field-label">Target column — what should the model predict?</div>', unsafe_allow_html=True)
    columns = list(preview_df.columns)
    # default to config.TARGET_COLUMN only if it's an actual column in this file
    default_index = columns.index(config.TARGET_COLUMN) if config.TARGET_COLUMN in columns else 0
    target_column = st.selectbox(
        "Target column",
        options=columns,
        index=default_index,
        label_visibility="collapsed",
    )

run_clicked = st.button("Run pipeline", disabled=(uploaded_file is None), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------
if run_clicked and uploaded_file is not None and target_column:
    st.session_state.app_state = "processing"
    st.session_state.target_column_choice = target_column
    st.rerun()

if st.session_state.app_state == "processing" and uploaded_file is not None:
    chosen_target = st.session_state.get("target_column_choice", target_column)
    save_csv(preview_df, config.RAW_DATA_FILE)

    stages = [
        "Loading data", "Summarizing", "Cleaning", "Running EDA", "Engineering features",
        "Encoding", "Selecting features", "Selecting model", "Training",
        "Evaluating", "Generating insights", "Building report", "Exporting",
    ]
    progress = st.progress(0, text=stages[0])
    for i, label in enumerate(stages[:-1]):
        progress.progress(int((i + 1) / len(stages) * 100), text=label + "…")
        time.sleep(0.15)

    result = run_pipeline(target_column=chosen_target)
    progress.progress(100, text="Done")

    st.session_state.pipeline_result = result
    st.session_state.app_state = "success" if result.get("success") else "error"
    st.rerun()

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
result = st.session_state.pipeline_result
if result is not None:
    if result.get("success"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">02 · Results</div>', unsafe_allow_html=True)

        cols = st.columns(min(len(result["metrics"]) + 1, 4))
        for i, (k, v) in enumerate(result["metrics"].items()):
            with cols[i % len(cols)]:
                st.metric(k, f"{v:.4f}" if isinstance(v, float) else v)
        with cols[len(result["metrics"]) % len(cols)]:
            st.metric("Best model", result["best_model_name"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Key insights**")
        insight_lines = result.get("insights", {}).get("summary", [])
        if insight_lines:
            for line in insight_lines:
                st.markdown(f'<div class="insight-line"><span class="insight-marker">→</span>{line}</div>', unsafe_allow_html=True)
        else:
            st.caption("No insights returned.")
        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------------------------------------
        # Downloads — dedicated cleaned-data button + combined bundle
        # -------------------------------------------------------------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">03 · Downloads</div>', unsafe_allow_html=True)

        cleaned_path = result.get("cleaned_data_path")
        report_path = result.get("model_report_path")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if cleaned_path and os.path.exists(cleaned_path):
                with open(cleaned_path, "rb") as f:
                    st.download_button(
                        "🧹 Cleaned data (.csv)",
                        f,
                        file_name="cleaned_sales_data.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            else:
                st.caption("Cleaned data not available.")
        with dcol2:
            if report_path and os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button(
                        "📄 Model report (.pdf)",
                        f,
                        file_name="model_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.caption("Report not available.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="mega-download">', unsafe_allow_html=True)
        st.markdown('<h3>📦 Get everything</h3>', unsafe_allow_html=True)
        st.markdown('<p>Cleaned data, model, and report bundled into one zip</p>', unsafe_allow_html=True)

        export_path = result.get("export_path")
        if export_path and os.path.exists(export_path):
            with open(export_path, "rb") as f:
                st.download_button(
                    "⬇️  Download everything (.zip)",
                    f,
                    file_name="sales_agent_export.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.caption("Export bundle not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.error(f"Pipeline failed at stage: **{result.get('stage')}**")
        st.code(result.get("error"))
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Predict on new data
# ---------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">04 · Predict on new data</div>', unsafe_allow_html=True)

if not os.path.exists(config.MODEL_FILE):
    st.caption("Run the pipeline above first to train a model.")
else:
    new_file = st.file_uploader("New data (no target column)", type=["csv"], key="predict_upload", label_visibility="collapsed")

    if new_file is not None:
        new_df = pd.read_csv(new_file)
        st.dataframe(new_df.head(), use_container_width=True)

        if st.button("Generate predictions", use_container_width=True):
            with st.spinner("Predicting…"):
                agent_result = PredictionAgent().execute(new_df)
            st.session_state.predict_result = agent_result
            st.rerun()

    predict_result = st.session_state.predict_result
    if predict_result is not None:
        if predict_result.get("success"):
            result_df = predict_result["result"]
            st.success(f"Generated {len(result_df)} predictions.")
            st.dataframe(result_df, use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download predictions (.csv)",
                csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.error(f"Prediction failed: {predict_result.get('error')}")

st.markdown('</div>', unsafe_allow_html=True)
