# ImdbPredictionDLRL — Sequential Text Prediction System (IMDB Movie Reviews)

**Activity-2 · Problem 1 · Dataset: IMDB · Solution 8 (Hybrid comparison: RNN vs BiRNN vs Deep RNN vs LSTM vs GRU)**

## What this is
A clean IMDb-themed (black + yellow) web application that classifies movie reviews as
**POSITIVE / NEGATIVE**, compares 5 recurrent architectures, shows a step-by-step
simulation of how the prediction was produced, and ships a full project report.

## Layout
- `app.py` — Flask application entry point
  - `app_flask.py` — Flask routes and API
  - **Section 1 · Studio Mode**: abstract input/output form (no internals shown)
  - **Section 2 · Lab Mode**: step-by-step simulation (tokenize → sequences → pad → embed → recurrent unroll → classify)
  - **Section 3 · Report**: what the project is, how it was built, metrics, gradient analysis
- `src/data_loader2.py` — loads IMDb data from local files, Hugging Face, or Stanford
- `src/preprocess.py` — cleaning, tokenization, vocab, numericalization, padding
- `src/rnn_cells.py`, `src/rnn_bi.py`, `src/rnn_lstm.py`, `src/rnn_gru.py` — recurrent architectures implemented in NumPy and trained with Adam
- `src/train_model.py` — full training pipeline (production TF-IDF+LR model + recurrent comparison)
- `src/inference.py` — loads artifacts, predicts, explains, builds simulation trace
- `models/` — trained artifacts used by the app (recreated by training)
- `PROJECT_REPORT.md` — full written report

## Quickstart
```powershell
py -m pip install -r requirements.txt
py app.py
```
Then open http://127.0.0.1:5000. The checked-in model artifacts let the app
start without training. To retrain the models, run `py -m src.train_model`;
training downloads or loads IMDb data and saves the new artifacts under `models/`.
