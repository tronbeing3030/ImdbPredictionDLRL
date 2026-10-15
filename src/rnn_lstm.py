"""LSTM from scratch."""
import numpy as np
from .rnn_base import RNNBase, sigmoid, xavier


class LSTM(RNNBase):
    name = "LSTM"

    def init_params(self, rng, V, D, H):
        def m(a, b, scale=1.0):
            return xavier(rng, a, b, scale=scale)
        return [m(V, D) * 0.5,
                m(D, H), m(H, H, scale=0.5), np.zeros(H),
                m(D, H), m(H, H, scale=0.5), np.zeros(H),
                m(D, H), m(H, H, scale=0.5), np.zeros(H),
                m(D, H), m(H, H, scale=0.5), np.zeros(H),
                m(H, 1, scale=0.5), np.zeros(1)]

    def forward(self, X_idx, P, training=True):
        (E, Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo,
         Wxg, Whg, bg, Wo, bout) = P
        B, T = X_idx.shape
        H = Whi.shape[0]
        X = E[X_idx]
        h = np.zeros((B, H)); c = np.zeros((B, H))
        HS = np.zeros((B, T + 1, H)); CS = np.zeros((B, T + 1, H))
        cache_t = []
        for t in range(T):
            xt = X[:, t]
            i = sigmoid(xt @ Wxi + h @ Whi + bi)
            f = sigmoid(xt @ Wxf + h @ Whf + bf)
            o = sigmoid(xt @ Wxo + h @ Who + bo)
            g = np.tanh(xt @ Wxg + h @ Whg + bg)
            c = f * c + i * g
            h = o * np.tanh(c)
            HS[:, t + 1], CS[:, t + 1] = h, c
            cache_t.append((xt, i, f, o, g, h.copy(), c.copy()))
        logits = (HS[:, T] @ Wo).ravel() + bout[0]
        return logits, (X_idx, X, HS, CS, cache_t)

    def backward(self, P, cache, y_true, y_prob):
        (E, Wxi, Whi, bi, Wxf, Whf, bf, Wxo, Who, bo,
         Wxg, Whg, bg, Wo, bout) = P
        X_idx, X, HS, CS, cache_t = cache
        B, T = X_idx.shape
        H = Whi.shape[0]
        dlog = (y_prob - y_true) / B
        dWo = HS[:, T].T @ dlog[:, None]
        dbout = np.array([dlog.sum()])
        dh = dlog[:, None] @ Wo.T
        dc = np.zeros((B, H))
        grads = [np.zeros_like(p) for p in P]
        dE = grads[0]
        for t in reversed(range(T)):
            xt, i, f, o, g, h_prev_ignored, c_cur = cache_t[t]
            h_prev = HS[:, t]; c_prev = CS[:, t]
            tanh_c = np.tanh(c_cur)
            dh = dh + np.zeros_like(dh)
            do = dh * tanh_c * o * (1 - o)
            dc = dh * o * (1 - tanh_c ** 2) + dc
            di = dc * g * i * (1 - i)
            df = dc * c_prev * f * (1 - f)
            dg = dc * i * (1 - g ** 2)
            dc = dc * f
            grads[1] += xt.T @ di; grads[2] += h_prev.T @ di; grads[3] += di.sum(0)
            grads[4] += xt.T @ df; grads[5] += h_prev.T @ df; grads[6] += df.sum(0)
            grads[7] += xt.T @ do; grads[8] += h_prev.T @ do; grads[9] += do.sum(0)
            grads[10] += xt.T @ dg; grads[11] += h_prev.T @ dg; grads[12] += dg.sum(0)
            dxt = di @ Wxi.T + df @ Wxf.T + do @ Wxo.T + dg @ Wxg.T
            dh = di @ Whi.T + df @ Whf.T + do @ Who.T + dg @ Whg.T
            for bi_ in range(B):
                dE[X_idx[bi_, t]] += dxt[bi_]
        grads[13] = dWo
        grads[14] = dbout
        return grads
