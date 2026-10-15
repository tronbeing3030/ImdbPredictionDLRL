"""Simple RNN + Deep 2-layer RNN."""
import numpy as np
from .rnn_base import RNNBase, xavier


class SimpleRNN(RNNBase):
    name = "Simple RNN"

    def init_params(self, rng, V, D, H):
        return [xavier(rng, V, D) * 0.5, xavier(rng, D, H),
                xavier(rng, H, H, scale=0.5), np.zeros(H),
                xavier(rng, H, 1, scale=0.5), np.zeros(1)]

    def forward(self, X_idx, P, training=True):
        E, Wx, Wh, b, Wo, bo = P
        B, T = X_idx.shape
        H = Wh.shape[0]
        X = E[X_idx]
        hs = np.zeros((B, T + 1, H))
        pre = np.zeros((B, T, H))
        h = np.zeros((B, H))
        for t in range(T):
            a = X[:, t] @ Wx + h @ Wh + b
            pre[:, t] = a
            h = np.tanh(a)
            hs[:, t + 1] = h
        logits = (hs[:, T] @ Wo).ravel() + bo[0]
        return logits, (X_idx, X, hs, pre)

    def backward(self, P, cache, y_true, y_prob):
        E, Wx, Wh, b, Wo, bo = P
        X_idx, X, hs, pre = cache
        B, T = X_idx.shape
        dlog = (y_prob - y_true) / B
        dWo = hs[:, T].T @ dlog[:, None]
        dbo = np.array([dlog.sum()])
        dh_next = dlog[:, None] @ Wo.T
        dWx = np.zeros_like(Wx)
        dWh = np.zeros_like(Wh)
        db = np.zeros_like(b)
        dE = np.zeros_like(E)
        for t in reversed(range(T)):
            dt = (1 - np.tanh(pre[:, t]) ** 2) * dh_next
            dWx += X[:, t].T @ dt
            dWh += hs[:, t].T @ dt
            db += dt.sum(0)
            dX = dt @ Wx.T
            for bi in range(B):
                dE[X_idx[bi, t]] += dX[bi]
            dh_next = dt @ Wh.T
        return [dE, dWx, dWh, db, dWo, dbo]


class DeepRNN(RNNBase):
    """Two stacked Simple-RNN layers."""
    name = "Deep RNN"

    def init_params(self, rng, V, D, H):
        return [xavier(rng, V, D) * 0.5,
                xavier(rng, D, H), xavier(rng, H, H, scale=0.5), np.zeros(H),
                xavier(rng, H, H), xavier(rng, H, H, scale=0.5), np.zeros(H),
                xavier(rng, H, 1, scale=0.5), np.zeros(1)]

    def forward(self, X_idx, P, training=True):
        E, Wx1, Wh1, b1, Wx2, Wh2, b2, Wo, bo = P
        B, T = X_idx.shape
        H = Wh1.shape[0]
        X = E[X_idx]
        h1 = np.zeros((B, H))
        h2 = np.zeros((B, H))
        H1 = np.zeros((B, T + 1, H))
        H2 = np.zeros((B, T + 1, H))
        A1 = np.zeros((B, T, H))
        A2 = np.zeros((B, T, H))
        for t in range(T):
            a1 = X[:, t] @ Wx1 + h1 @ Wh1 + b1
            h1 = np.tanh(a1)
            a2 = h1 @ Wx2 + h2 @ Wh2 + b2
            h2 = np.tanh(a2)
            A1[:, t], H1[:, t + 1] = a1, h1
            A2[:, t], H2[:, t + 1] = a2, h2
        logits = (H2[:, T] @ Wo).ravel() + bo[0]
        return logits, (X_idx, X, H1, A1, H2, A2)

    def backward(self, P, cache, y_true, y_prob):
        E, Wx1, Wh1, b1, Wx2, Wh2, b2, Wo, bo = P
        X_idx, X, H1, A1, H2, A2 = cache
        B, T = X_idx.shape
        dlog = (y_prob - y_true) / B
        dWo = H2[:, T].T @ dlog[:, None]
        dbo = np.array([dlog.sum()])
        dh2 = dlog[:, None] @ Wo.T
        dh1_next = np.zeros_like(dh2)
        dWx1 = np.zeros_like(Wx1); dWh1 = np.zeros_like(Wh1); db1 = np.zeros_like(b1)
        dWx2 = np.zeros_like(Wx2); dWh2 = np.zeros_like(Wh2); db2 = np.zeros_like(b2)
        dE = np.zeros_like(E)
        for t in reversed(range(T)):
            dt2 = (1 - np.tanh(A2[:, t]) ** 2) * dh2
            dWx2 += H1[:, t + 1].T @ dt2
            dWh2 += H2[:, t].T @ dt2
            db2 += dt2.sum(0)
            dh1 = dt2 @ Wx2.T + dh1_next
            dt1 = (1 - np.tanh(A1[:, t]) ** 2) * dh1
            dWx1 += X[:, t].T @ dt1
            dWh1 += H1[:, t].T @ dt1
            db1 += dt1.sum(0)
            for bi in range(B):
                dE[X_idx[bi, t]] += dt1[bi] @ Wx1.T
            dh1_next = dt1 @ Wh1.T
            dh2 = dt2 @ Wh2.T
        return [dE, dWx1, dWh1, db1, dWx2, dWh2, db2, dWo, dbo]
