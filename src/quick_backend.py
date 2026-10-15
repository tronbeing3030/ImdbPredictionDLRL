"""Fallback artifact builder — trains a small but REAL model on the HF mirror subset.

Used only if full training is still running: gives the app a working backend fast.
Full training overwrites these files with stronger ones when it finishes.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"


def main(n_train=6000, n_test=2000):
    from .data_loader2 import _read_hf
    from .preprocess import build_vocab, save_vocab
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    got = _read_hf()
    if not got:
        print("[quick] HF mirror unavailable")
        return
    tr_x, tr_y, te_x, te_y = got
    tr_x, tr_y = tr_x[:n_train], tr_y[:n_train]
    te_x, te_y = te_x[:n_test], te_y[:n_test]
    stoi, _ = build_vocab(tr_x, max_vocab=8000)
    save_vocab(stoi, MODEL_DIR / "vocab.json")
    vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000, C=4.0)
    clf.fit(vec.fit_transform(tr_x), tr_y)
    pred = clf.predict(vec.transform(te_x))
    acc = float(accuracy_score(te_y, pred))
    prec, rec, f1, _ = precision_recall_fscore_support(te_y, pred, average="binary", zero_division=0)
    joblib.dump(vec, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(clf, MODEL_DIR / "sentiment_model.joblib")
    mpath = MODEL_DIR / "metrics.json"
    base = json.loads(mpath.read_text()) if mpath.exists() else {}
    base.update({"quick_backend": True, "quick_acc": acc,
                 "production": {"model": "TF-IDF + LogisticRegression (quick backend)",
                                "accuracy": acc, "precision": float(prec),
                                "recall": float(rec), "f1": float(f1),
                                "confusion_matrix": confusion_matrix(te_y, pred).tolist()}})
    mpath.write_text(json.dumps(base, indent=2))
    print(f"[quick] backend ready acc={acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
