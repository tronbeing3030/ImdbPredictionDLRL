"""IMDB data loader — downloads aclImdb once and parses it."""
import os
import ssl
import tarfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"
ROOT = DATA_DIR / "aclImdb"


def _download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ROOT.exists():
        return
    if not ARCHIVE.exists():
        print(f"[data] downloading IMDB (~84MB) ...", flush=True)
        ctx = ssl.create_default_context()
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as r, open(ARCHIVE, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print("[data] download complete.", flush=True)
    print("[data] extracting ...", flush=True)
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        tf.extractall(DATA_DIR)
    print("[data] extracted.", flush=True)


def _read_split(split):
    texts, labels = [], []
    for label, name in ((1, "pos"), (0, "neg")):
        folder = ROOT / split / name
        for f in sorted(folder.glob("*.txt")):
            try:
                texts.append(f.read_text(encoding="utf-8"))
            except Exception:
                texts.append(f.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label)
    return texts, labels


def load_imdb():
    _download()
    tr_x, tr_y = _read_split("train")
    te_x, te_y = _read_split("test")
    print(f"[data] train={len(tr_x)} test={len(te_x)}", flush=True)
    return tr_x, tr_y, te_x, te_y

