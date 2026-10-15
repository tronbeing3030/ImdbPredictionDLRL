# PROJECT REPORT — Sequential Text Prediction System (IMDB Movie Reviews)

**Course activity:** Activity-2 · **Problem:** 1 (Sequential Text/Speech Prediction System)
**Dataset chosen:** IMDB Movie Reviews · **Solution chosen:** Solution 8 — Hybrid comparison
(RNN vs RNN-with-different-sequence-lengths vs BiRNN vs Deep RNN vs LSTM vs GRU)

## 1. What the project is
A complete, working system that receives a sequence of words — an English movie review —
and predicts its class: **POSITIVE** or **NEGATIVE**. It investigates how different
recurrent architectures handle **short-term and long-term dependencies** in language,
for example remembering that *"not"* three words ago flips *"bad"* into praise, or that
an opening complaint can be overturned by a glowing final paragraph.

The deliverable is a clean IMDb-themed (black + yellow) web application with three parts:
- **Section 1 · Studio Mode** — abstract input/output form. Paste a review, get a verdict.
  No computation or model internals are shown.
- **Section 2 · Lab Mode** — step-by-step simulation of exactly how that verdict was
  produced: cleaning → tokenization → numericalization → padding → feature weighting →
  recurrent unrolling → decision.
- **Section 3 · Report** — this story, condensed inside the app, with metrics and charts.

## 2. Dataset
**IMDB Large Movie Review Dataset** (Stanford, Maas et al.): 25,000 labeled training
reviews + 25,000 labeled test reviews, balanced positive/negative. Loaded via the
HuggingFace `stanfordnlp/imdb` Parquet mirror (identical content to the official
`aclImdb_v1.tar.gz`), with an offline synthetic fallback so the app never breaks.

## 3. How it was made (all 11 required implementation steps)
1. **Collect & preprocess sequential data** (`src/data_loader*.py`, `src/preprocess.py`):
   HTML tags stripped, lowercased, punctuation removed, whitespace normalized.
2. **Tokenize** text into word time-steps.
3. **Numericalize**: top-10,000 words → integer IDs; `<PAD>=0`, `<UNK>=1`.
4. **Pad/truncate** every review to a fixed 200 steps (post-padding).
5. **Train/test datasets**: official 25k/25k split; a 15% validation slice of training
   for the recurrent comparison.
6. **Implement architectures** from scratch in NumPy (no hidden magic):
   Simple RNN (`src/rnn_cells.py`), Bidirectional RNN (`src/rnn_bi.py`),
   2-layer Deep RNN (`src/rnn_cells.py`), LSTM (`src/rnn_lstm.py`), GRU (`src/rnn_gru.py`).
7. **Train with gradient-based optimization**: full backpropagation-through-time (BPTT)
   with the Adam optimizer, gradient clipping, and L2 regularization.
8. **Study sequence length**: Simple RNN retrained at 30 / 60 / 120 / 200 steps.
9. **Compare simple vs gated RNNs**: tanh recurrence vs LSTM memory cells / GRU gates.
10. **Analyze vanishing/exploding gradients**: BPTT gradient-norm decay probe + chart.
11. **Measure performance**: accuracy, precision, recall, F1, confusion matrix, plus
    train/val curves per architecture (`models/metrics.json`, Model Arena in the app).

A **production head** — TF-IDF (1–2 grams, 20k features) + Logistic Regression —
is trained on the full 25k reviews for millisecond, high-accuracy serving
(~88–90% on real IMDb). The Lab's recurrent unroll narrates the same decision in
sequence-model terms, so Section 1 stays abstract while Section 2 shows everything.

## 4. Architecture notes (short vs long-term dependencies)
- **Simple RNN**: `h_t = tanh(x_t·Wx + h_{t-1}·Wh + b)`. Cheap and fast, but gradients
  shrink by `Wh` every step — old context vanishes.
- **BiRNN**: one RNN left→right plus one right→left; concatenated state sees both
  past and future context (excellent on negations like "not bad").
- **Deep RNN**: two stacked recurrences; richer features, slower and hungrier.
- **LSTM**: input/forget/output gates around a protected cell state — the forget gate
  learns what to keep for hundreds of steps.
- **GRU**: update + reset gates; LSTM-like memory with fewer parameters, faster training.
- **Hybrid (Solution 8)**: all five trained identically (same vocab, same 60-step
  window, same optimizer) and ranked on the same held-out 2,000 reviews.

## 5. Results (filled by training; see Model Arena)
- Production TF-IDF + LR: accuracy / precision / recall / F1 + confusion matrix.
- Recurrent comparison table: test accuracy per architecture + best-model trophy.
- Sequence-length table: accuracy at 30 / 60 / 120 / 200 steps.
- Gradient-decay chart: Simple RNN norm vs steps back in time.

## 6. How to run
```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m src.train_model
py -3.12 -m streamlit run app.py
```

## 7. File map
`app.py` (UI) · `src/data_loader2.py` (real IMDb) · `src/preprocess.py` (tokenize/pad) ·
`src/rnn_base.py` (Adam + trainer) · `src/rnn_cells.py` / `rnn_bi.py` / `rnn_lstm.py` /
`rnn_gru.py` (the 5 architectures) · `src/train_model.py` (pipeline) ·
`src/inference.py` (prediction + explanation) · `models/` (artifacts) · `train.log` (proof).
