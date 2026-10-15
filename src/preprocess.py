"""Cleaning / tokenization / numericalization / padding."""
import re
import json
from collections import Counter

HTML_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")


def clean_text(s: str) -> str:
    s = s.lower()
    s = HTML_RE.sub(" ", s)
    s = s.replace("\\n", " ")
    s = NON_ALPHA_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str):
    return clean_text(s).split()


def build_vocab(texts, max_vocab=10000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    most = [w for w, c in counter.most_common() if c >= min_freq][: max_vocab - 2]
    stoi = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(most, start=2):
        stoi[w] = i
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


def encode(tokens, stoi, unk=1):
    return [stoi.get(t, unk) for t in tokens]


def pad_sequences(seqs, maxlen, pad=0, trunc="post"):
    import numpy as np
    out = np.full((len(seqs), maxlen), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        if len(s) > maxlen:
            s = s[:maxlen] if trunc == "post" else s[-maxlen:]
        out[i, : len(s)] = s
    return out


def save_vocab(stoi, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stoi, f)


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {k: int(v) for k, v in d.items()}
