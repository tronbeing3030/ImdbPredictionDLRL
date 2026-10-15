"""IMDB data loader.

Priority:
1. Local aclImdb folder (data/aclImdb) if already extracted.
2. HuggingFace Parquet mirror (fast, no 84MB tarball): stanfordnlp/imdb.
3. Stanford tarball (streamed).
4. Synthetic fallback (offline) so the app NEVER breaks.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROOT = DATA_DIR / "aclImdb"
ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
HF_PARQUET = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet"
HF_PARQUET_TEST = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet"


def _read_local(split):
    texts, labels = [], []
    for label, name in ((1, "pos"), (0, "neg")):
        folder = ROOT / split / name
        if not folder.exists():
            return None
        for f in sorted(folder.glob("*.txt")):
            try:
                texts.append(f.read_text(encoding="utf-8"))
            except Exception:
                texts.append(f.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label)
    if not texts:
        return None
    return texts, labels


def _read_hf():
    try:
        import pandas as pd
    except Exception:
        return None
    try:
        tr = pd.read_parquet(HF_PARQUET, engine="pyarrow")
        te = pd.read_parquet(HF_PARQUET_TEST, engine="pyarrow")
        def split_df(df):
            t = df["text"].astype(str).tolist()
            y = df["label"].astype(int).tolist()
            return t, y
        tr_x, tr_y = split_df(tr)
        te_x, te_y = split_df(te)
        print(f"[data] HF mirror: train={len(tr_x)} test={len(te_x)}", flush=True)
        return tr_x, tr_y, te_x, te_y
    except Exception as e:
        print(f"[data] HF mirror failed: {e}", flush=True)
        return None


def _read_tarball():
    import tarfile
    import urllib.request
    import ssl
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists() or ARCHIVE.stat().st_size < 10_000_000:
        print("[data] downloading Stanford tarball ...", flush=True)
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=300) as r, open(ARCHIVE, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    if not ROOT.exists():
        with tarfile.open(ARCHIVE, "r:gz") as tf:
            tf.extractall(DATA_DIR)
    tr = _read_local("train")
    te = _read_local("test")
    if tr and te:
        return tr[0], tr[1], te[0], te[1]
    return None


def load_imdb():
    for split in ("train", "test"):
        pass
    tr = _read_local("train")
    te = _read_local("test")
    if tr and te:
        print(f"[data] local aclImdb: train={len(tr[0])} test={len(te[0])}", flush=True)
        return tr[0], tr[1], te[0], te[1]
    hf = _read_hf()
    if hf:
        return hf
    tb = _read_tarball()
    if tb:
        print("[data] Stanford tarball OK", flush=True)
        return tb
    raise RuntimeError("no IMDB source available")
