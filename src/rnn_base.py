"""Shared base + Adam for NumPy recurrent models."""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, clip=5.0):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = np.clip(g, -clip, clip)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def xavier(rng, a, b, scale=1.0):
    return rng.standard_normal((a, b)) * np.sqrt(scale / max(a, b))


class RNNBase:
    name = "base"

    def init_params(self, rng, V, D, H):
        raise NotImplementedError

    def forward(self, X_idx, P, training=True):
        raise NotImplementedError

    def backward(self, P, cache, y_true, y_prob):
        raise NotImplementedError

    def fit(self, Xtr, ytr, Xva, yva, vocab_size, epochs=6, batch=256,
            emb_dim=64, hid=64, lr=2e-3, seed=0, log_cb=None, l2=1e-5):
        rng = np.random.default_rng(seed)
        params = self.init_params(rng, vocab_size, emb_dim, hid)
        opt = Adam(params, lr=lr)
        n = len(Xtr)
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        for ep in range(epochs):
            perm = rng.permutation(n)
            tl, tc, tot = 0.0, 0, 0
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb, yb = Xtr[idx], ytr[idx].astype(float)
                logits, cache = self.forward(xb, params, training=True)
                prob = sigmoid(logits)
                eps = 1e-7
                loss = -(yb * np.log(prob + eps) + (1 - yb) * np.log(1 - prob + eps)).mean()
                grads = self.backward(params, cache, yb, prob)
                grads = [g + l2 * p for g, p in zip(grads, params)]
                opt.step(grads)
                tl += loss * len(xb)
                tc += ((prob >= 0.5).astype(int) == yb.astype(int)).sum()
                tot += len(xb)
            vl, va = self.evaluate(Xva, yva, params)
            history["loss"].append(float(tl / tot))
            history["acc"].append(float(tc / tot))
            history["val_loss"].append(float(vl))
            history["val_acc"].append(float(va))
            if log_cb:
                log_cb(ep + 1, epochs, tl / tot, tc / tot, vl, va)
        self.trained_params = params
        return history

    def evaluate(self, X, y, params=None):
        params = self.trained_params if params is None else params
        logits, _ = self.forward(X, params, training=False)
        prob = sigmoid(logits)
        eps = 1e-7
        yf = y.astype(float)
        loss = float(-(yf * np.log(prob + eps) + (1 - yf) * np.log(1 - prob + eps)).mean())
        acc = float((((prob >= 0.5).astype(int)) == yf.astype(int)).mean())
        return loss, acc

    def predict_proba(self, X):
        logits, _ = self.forward(X, self.trained_params, training=False)
        return sigmoid(logits)
