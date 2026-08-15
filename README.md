# 🤖 AI Sales Data Agent

A production-style, multi-agent pipeline that automates the full ML workflow
for sales data: cleaning → EDA → feature engineering → encoding →
feature selection → model selection → training → evaluation → insights →
reporting → prediction.

## 📁 Project Structure

```
ai_agent/
├── app.py                  # Streamlit UI
├── main.py                 # Pipeline orchestrator (CLI entry point)
├── config.py                # All paths, constants, settings
├── requirements.txt
├── .env                      # Your API keys (never commit this)
├── .env.example               # Safe template to share
│
├── data/                     # Snapshots of data at each pipeline stage
│   ├── raw/  cleaned/  processed/  predictions/
│
├── agents/                   # Orchestration layer (one class per pipeline stage)
├── preprocessing/            # Pure, reusable logic functions (no agent dependency)
├── models/                    # Saved trained models + metadata
├── reports/                    # Generated PDF reports
├── outputs/                     # Final deliverables (cleaned csv, predictions, zipped export)
├── logs/                          # Daily pipeline logs
└── utils/                          # logger, validator, file_handler helpers
```

## ⚙️ Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your .env (optional, only needed for future LLM features)
cp .env.example .env
```

## ▶️ How to Run

### Option A — Command line (full pipeline)
1. Put your raw CSV at `data/raw/sales_dataset.csv`
2. Set `TARGET_COLUMN` and `TASK_TYPE` in `config.py`
3. Run:
   ```bash
   python main.py
   ```

### Option B — Streamlit UI (recommended for interactive use)
```bash
streamlit run app.py
```
Then open the local URL shown in the terminal, upload your CSV, and click
**Run Full Pipeline**.

## 🔑 About the `.env` file

The pipeline itself (cleaning, training, prediction) does **not** require
any API key — everything runs locally with scikit-learn. The `.env` file
is included for when you plug in an LLM later (e.g. to have
`insight_agent.py` generate richer natural-language insights).

Keys currently supported:
```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

## 🧩 Adding your own logic

- New cleaning step → add a function to `preprocessing/`, call it from the
  matching agent in `agents/`.
- New model candidate → add it to `REGRESSION_MODELS` / `CLASSIFICATION_MODELS`
  dict in `agents/model_selection_agent.py` and `agents/training_agent.py`.
- New engineered feature → add a function to `preprocessing/feature_engineering.py`.

## 📝 Notes

- Change `config.TARGET_COLUMN` to match your dataset's label column.
- Change `config.TASK_TYPE` to `"regression"` or `"classification"`.
- Logs are written daily to `logs/pipeline_YYYYMMDD.log`.
