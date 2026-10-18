# Agent Instructions

## Project Overview
This is a machine-learning Streamlit application that predicts customer shopping preferences
(Online / Store / Hybrid) using the provided CSV dataset.

## Workflow
1. **Data** lives in `data/dataset.csv` — do not modify it.
2. **Training** is done by `train_model.py`:
   - Loads `data/dataset.csv`
   - Identifies numeric and categorical features automatically
   - Compares multiple classifiers via cross-validation
   - Saves the best pipeline + metadata to `models/model.pkl`
3. **App** is `app.py` — Streamlit UI that loads `models/model.pkl` and runs predictions.

## Rules
- Use **relative paths** only (via `pathlib.Path(__file__).parent`).
- Never hard-code prediction logic — always use `pipeline.predict()`.
- All API keys / secrets go in environment variables; see `.env.example`.
- Do not import Flask, Django, or use Jupyter Notebooks.
- `requirements.txt` must stay in sync with all imports used.

## Retraining
Run `python train_model.py` to regenerate `models/model.pkl`.
The app will automatically pick up the new model on next load (cache is cleared on restart).

## Adding a New Model
Add an entry to the `candidates` dict in `train_model.py`. The comparison loop handles the rest.

## Deployment
- Streamlit Community Cloud: push to GitHub, connect repo at share.streamlit.io.
- Render: add a web service with start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
