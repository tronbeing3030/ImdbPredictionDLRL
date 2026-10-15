"""GRU from scratch."""
import numpy as np
from .rnn_base import RNNBase, sigmoid, xavier


class GRU(RNNBase):
    name = "GRU"

    def init_params(self, rng, V, D, H):
        def m(a, b, scale=1.0):
            return xavier(rng, a, b, scale=scale)
        return [m(V, D) * 0.5,
                m(D, H), m(H, H, scale=0.5), np.zeros(H),
                m(D, H), m(H, H, scale=0.5), np.zeros(H),
                m(D, H), m(H, H, scale=0.5), np.zeros(H),
                m(H, 1, scale=0.5), np.zeros(1)]

    def forward(self, X_idx, P, training=True):
        (E, Wxz, Whz, bz, Wxr, Whr, br, Wxh, Whh, bh, Wo, bout) = P
        B, T = X_idx.shape
        H = Whz.shape[0]
        X = E[X_idx]
        h = np.zeros((B, H))
        HS = np.zeros((B, T + 1, H))
        steps = []
        for t in range(T):
            xt = X[:, t]
            z = sigmoid(xt @ Wxz + h @ Whz + bz)
            r = sigmoid(xt @ Wxr + h @ Whr + br)
            hh = np.tanh(xt @ Wxh + (r * h) @ Whh + bh)
            h = (1 - z) * h + z * hh
            HS[:, t + 1] = h
            steps.append((xt, z, r, hh))
        logits = (HS[:, T] @ Wo).ravel() + bout[0]
        return logits, (X_idx, X, HS, steps)

    def backward(self, P, cache, y_true, y_prob):
        (E, Wxz, Whz, bz, Wxr, Whr, br, Wxh, Whh, bh, Wo, bout) = P
        X_idx, X, HS, steps = cache
        B, T = X_idx.shape
        dlog = (y_prob - y_true) / B
        dWo = HS[:, T].T @ dlog[:, None]
        dbout = np.array([dlog.sum()])
        dh = dlog[:, None] @ Wo.T
        grads = [np.zeros_like(p) for p in P]
        dE = grads[0]
        for t in reversed(range(T)):
            xt, z, r, hh = steps[t]
            h_prev = HS[:, t]
            dz = dh * (hh - h_prev) * z * (1 - z)
            dhh = dh * z * (1 - hh ** 2)
            dr = (dhh @ Whh.T) * h_prev * r * (1 - r)
            dh_prev = dh * (1 - z) + (dhh @ Whh.T) * r
            dh_prev += dz @ Whz.T + dr @ Whr.T
            dxt = dz @ Wxz.T + dr @ Wxr.T + dhh @ Wxh.T
            grads[1] += xt.T @ dz; grads[2] += h_prev.T @ dz; grads[3] += dz.sum(0)
            grads[4] += xt.T @ dr; grads[5] += h_prev.T @ dr; grads[6] += dr.sum(0)
            grads[7] += xt.T @ dhh; grads[8] += (r * h_prev).T @ dhh; grads[9] += dhh.sum(0)
            for bi_ in range(B):
                dE[X_idx[bi_, t]] += dxt[bi_]
            dh = dh_prev
        grads[10] = dWo
        grads[11] = dbout
        return grads
