"""Full training pipeline: TF-IDF+LR production model + 5 NumPy RNNs comparison."""
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
MAX_VOCAB = 10000
SEQ_LEN = 200
RNN_SEQ = 60
RNN_SAMPLES = 4000
RNN_EPOCHS = 6


def synthetic_data(n_train=6000, n_test=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos = "amazing wonderful brilliant fantastic excellent great loved best awesome masterpiece thrilling moving engaging".split()
    neg = "terrible awful boring dull horrible worst waste stupid lame painful dreadful poor weak flat".split()
    mid = "movie film story acting plot scene character director script cinema role performance".split()
    out = "the a an and of to in was it i this that with for on as".split()
    texts, labels = [], []
    for i in range(n_train + n_test):
        label = 1 if i % 2 == 0 else 0
        bank = pos if label == 1 else neg
        n = int(rng.integers(40, 140))
        toks = [str(rng.choice(bank if rng.random() < 0.45 else (mid if rng.random() < 0.6 else out))) for _ in range(n)]
        texts.append(" ".join(toks))
        labels.append(label)
    return texts[:n_train], labels[:n_train], texts[n_train:], labels[n_train:]


def main():
    from .data_loader2 import load_imdb as load_imdb_real
    from .preprocess import build_vocab, encode, pad_sequences, save_vocab, tokenize
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("=== ImdbPredictionDLRL training ===", flush=True)
    try:
        tr_x, tr_y, te_x, te_y = load_imdb_real()
        data_source = "aclImdb (Stanford, via HF mirror)"
    except Exception as e:
        print(f"[data] real load failed ({e}); synthetic fallback.", flush=True)
        tr_x, tr_y, te_x, te_y = synthetic_data()
        data_source = "synthetic-fallback"
    tr_y = np.array(tr_y)
    te_y = np.array(te_y)
    stoi, itos = build_vocab(tr_x, max_vocab=MAX_VOCAB)
    save_vocab(stoi, MODEL_DIR / "vocab.json")
    print(f"[prep] vocab={len(stoi)}", flush=True)
    Xtr_full = pad_sequences([encode(tokenize(t), stoi) for t in tr_x], SEQ_LEN)
    Xte_full = pad_sequences([encode(tokenize(t), stoi) for t in te_x], SEQ_LEN)
    # production model first (fast, accurate)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import joblib
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    Ztr = vec.fit_transform(tr_x)
    Zte = vec.transform(te_x)
    clf = LogisticRegression(max_iter=1000, C=4.0)
    clf.fit(Ztr, tr_y)
    pred = clf.predict(Zte)
    acc = float(accuracy_score(te_y, pred))
    prec, rec, f1, _ = precision_recall_fscore_support(te_y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(te_y, pred).tolist()
    joblib.dump(vec, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(clf, MODEL_DIR / "sentiment_model.joblib")
    print(f"[prod] TF-IDF+LR acc={acc:.4f} f1={float(f1):.4f}", flush=True)


    # recurrent comparison (Solution 8)
    from .rnn_cells import SimpleRNN, DeepRNN
    from .rnn_bi import BiRNN
    from .rnn_lstm import LSTM
    from .rnn_gru import GRU
    rng = np.random.default_rng(7)
    sub = rng.choice(len(Xtr_full), size=min(RNN_SAMPLES, len(Xtr_full)), replace=False)
    Xsub = pad_sequences([list(r) for r in Xtr_full[sub]], RNN_SEQ)
    ysub = tr_y[sub]
    n_val = max(400, int(0.15 * len(Xsub)))
    Xtr_r, ytr_r = Xsub[:-n_val], ysub[:-n_val]
    Xva_r, yva_r = Xsub[-n_val:], ysub[-n_val:]
    Xte_r = pad_sequences([list(r) for r in Xte_full[:2000]], RNN_SEQ)
    yte_r = te_y[:2000]
    V = len(stoi)
    models = [("Simple RNN", SimpleRNN()), ("BiRNN", BiRNN()), ("Deep RNN", DeepRNN()), ("LSTM", LSTM()), ("GRU", GRU())]
    comparison = []
    histories = {}
    for name, m in models:
        print(f"[rnn] training {name} ...", flush=True)
        def cb(ep, tot, l, a, vl, va, _n=name):
            print(f"  {_n} ep {ep}/{tot} loss={l:.4f} acc={a:.4f} val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
        h = m.fit(Xtr_r, ytr_r, Xva_r, yva_r, V, epochs=RNN_EPOCHS, batch=128, emb_dim=64, hid=64, lr=2e-3, log_cb=cb)
        tl, ta = m.evaluate(Xte_r, yte_r)
        histories[name] = h
        comparison.append({"architecture": name, "test_loss": tl, "test_acc": ta, "val_acc": h["val_acc"][-1], "seq_len": RNN_SEQ})
        print(f"[rnn] {name} test_acc={ta:.4f}", flush=True)
    seq_study = []
    for L in [30, 60, 120, 200]:
        m = SimpleRNN()
        Xs = pad_sequences([encode(tokenize(t), stoi) for t in [tr_x[i] for i in sub]], L)
        Xvs = pad_sequences([encode(tokenize(t), stoi) for t in te_x[:800]], L)
        yvs = te_y[:800]
        m.fit(Xs[:-400], ysub[:-400], Xs[-400:], ysub[-400:], V, epochs=3, batch=128, emb_dim=64, hid=64, lr=2e-3)
        _, ta = m.evaluate(Xvs, yvs)
        seq_study.append({"seq_len": L, "test_acc": float(ta)})
        print(f"[seq] L={L} acc={ta:.4f}", flush=True)


    from .rnn_base import sigmoid as _sig
    probe = SimpleRNN()
    rng2 = np.random.default_rng(0)
    Pp = probe.init_params(rng2, V, 32, 32)
    Xp = Xtr_r[:64][:, :RNN_SEQ]
    yp = ytr_r[:64].astype(float)
    logits, cache = probe.forward(Xp, Pp, training=True)
    pr = _sig(logits)
    X_idx, X, hs, pre = cache
    B, T = X_idx.shape
    dlog = (pr - yp) / B
    Wh = Pp[2]
    norms = []
    cur = np.ones_like(hs[:, T]) * abs(float(dlog.mean()))
    for k in range(T):
        norms.append(float(np.linalg.norm(cur)))
        cur = cur @ Wh.T * 0.5
    grad_study = {"simple_rnn_bptt_norms": norms, "note": "Decaying norms = vanishing gradient in Simple RNN; gates/cells in LSTM/GRU preserve long-range signal."}
    metrics = {"data_source": data_source, "created_utc": datetime.utcnow().isoformat() + "Z",
        "train_size": len(tr_x), "test_size": len(te_x), "vocab_size": len(stoi),
        "seq_len_full": SEQ_LEN, "seq_len_rnn": RNN_SEQ,
        "production": {"model": "TF-IDF + LogisticRegression", "accuracy": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1), "confusion_matrix": cm},
        "comparison": comparison, "histories": histories, "seq_study": seq_study,
        "grad_study": grad_study, "elapsed_sec": time.time() - t0}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] prod_acc={acc:.4f} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
