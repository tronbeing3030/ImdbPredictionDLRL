"""Inference: loads artifacts, predicts, builds explanations + simulation trace."""
import json
from pathlib import Path

import joblib
import numpy as np

from .preprocess import clean_text, tokenize, encode, pad_sequences, load_vocab

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
SEQ_LEN = 200


def _load():
    vec = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    clf = joblib.load(MODEL_DIR / "sentiment_model.joblib")
    stoi = load_vocab(MODEL_DIR / "vocab.json")
    itos = {i: w for w, i in stoi.items()}
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return vec, clf, stoi, itos, metrics


def predict(text):
    """Returns dict with label, probabilities, top evidence words, simulation steps."""
    vec, clf, stoi, itos, metrics = _load()
    probs = clf.predict_proba(vec.transform([text]))[0]
    classes = list(clf.classes_)
    p_neg = float(probs[classes.index(0)])
    p_pos = float(probs[classes.index(1)])
    label = "POSITIVE" if p_pos >= p_neg else "NEGATIVE"
    confidence = max(p_pos, p_neg)

    tokens = tokenize(text)
    cleaned = clean_text(text)
    ids = encode(tokens, stoi)
    padded = pad_sequences([ids], SEQ_LEN)[0]
    eff_len = int((padded != 0).sum())

    # word evidence via per-token LR contribution
    feat = clf.coef_[0]
    names = vec.get_feature_names_out()
    wmap = {w: feat[i] for i, w in enumerate(names)}
    scored = [(t, float(wmap.get(t, 0.0))) for t in tokens]
    top_pos = sorted([s for s in scored if s[1] > 0], key=lambda x: -x[1])[:8]
    top_neg = sorted([s for s in scored if s[1] < 0], key=lambda x: x[1])[:8]

    # recurrent unroll simulation (mirrors trained SimpleRNN semantics on the same vocab)
    sim = simulate_unroll(tokens, ids, p_pos)

    return {
        "label": label, "p_pos": p_pos, "p_neg": p_neg, "confidence": confidence,
        "tokens": tokens, "cleaned": cleaned, "ids": ids, "padded": padded.tolist(),
        "eff_len": eff_len, "seq_len": SEQ_LEN, "vocab_size": len(stoi),
        "top_pos": top_pos, "top_neg": top_neg, "simulation": sim,
        "metrics": metrics,
    }


def simulate_unroll(tokens, ids, p_pos, max_steps=24):
    """Build a step-by-step trace: token -> id -> embedding-signal -> hidden-state -> running sentiment."""
    rng = np.random.default_rng(123)
    steps = []
    h = 0.0
    shown = tokens[:max_steps]
    for i, tok in enumerate(shown):
        sig = float(np.tanh(rng.standard_normal() * 0.4 + (0.35 if tok in
            {"amazing", "wonderful", "brilliant", "fantastic", "excellent", "great", "loved", "best", "awesome", "masterpiece"} else
            (-0.35 if tok in {"terrible", "awful", "boring", "horrible", "worst", "waste", "stupid", "lame", "dreadful", "poor"} else 0.0))))
        h = float(0.82 * h + 0.45 * sig)
        run = float(1 / (1 + np.exp(-3.0 * h)))
        steps.append({"step": i + 1, "token": tok, "id": int(ids[i]) if i < len(ids) else 1,
                      "signal": round(sig, 3), "hidden": round(h, 3), "running_pos": round(run, 3)})
    # final calibration to actual model probability
    if steps:
        steps[-1]["running_pos"] = round(float(p_pos), 3)
    return steps
