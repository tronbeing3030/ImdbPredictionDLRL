"""Bidirectional RNN."""
import numpy as np
from .rnn_base import RNNBase, xavier


class BiRNN(RNNBase):
    name = "Bidirectional RNN"

    def init_params(self, rng, V, D, H):
        return [xavier(rng, V, D) * 0.5,
                xavier(rng, D, H), xavier(rng, H, H, scale=0.5), np.zeros(H),
                xavier(rng, D, H), xavier(rng, H, H, scale=0.5), np.zeros(H),
                xavier(rng, 2 * H, 1, scale=0.5), np.zeros(1)]

    def forward(self, X_idx, P, training=True):
        E, Wxf, Whf, bf, Wxb, Whb, bb, Wo, bo = P
        B, T = X_idx.shape
        H = Whf.shape[0]
        X = E[X_idx]
        hf = np.zeros((B, H)); hb = np.zeros((B, H))
        HF = np.zeros((B, T + 1, H)); HB = np.zeros((B, T + 1, H))
        AF = np.zeros((B, T, H)); AB = np.zeros((B, T, H))
        for t in range(T):
            af = X[:, t] @ Wxf + hf @ Whf + bf
            hf = np.tanh(af)
            AF[:, t], HF[:, t + 1] = af, hf
        for t in reversed(range(T)):
            ab = X[:, t] @ Wxb + hb @ Whb + bb
            hb = np.tanh(ab)
            AB[:, t], HB[:, t] = ab, hb
        feat = np.concatenate([HF[:, T], HB[:, 0]], axis=1)
        logits = (feat @ Wo).ravel() + bo[0]
        return logits, (X_idx, X, HF, AF, HB, AB, feat)

    def backward(self, P, cache, y_true, y_prob):
        E, Wxf, Whf, bf, Wxb, Whb, bb, Wo, bo = P
        X_idx, X, HF, AF, HB, AB, feat = cache
        B, T = X_idx.shape
        dlog = (y_prob - y_true) / B
        dWo = feat.T @ dlog[:, None]
        dbo = np.array([dlog.sum()])
        dfeat = dlog[:, None] @ Wo.T
        H = Whf.shape[0]
        dhf = dfeat[:, :H]
        dhb = dfeat[:, H:]
        dWxf = np.zeros_like(Wxf); dWhf = np.zeros_like(Whf); dbf = np.zeros_like(bf)
        dWxb = np.zeros_like(Wxb); dWhb = np.zeros_like(Whb); dbb = np.zeros_like(bb)
        dE = np.zeros_like(E)
        for t in reversed(range(T)):
            dt = (1 - np.tanh(AF[:, t]) ** 2) * dhf
            dWxf += X[:, t].T @ dt; dWhf += HF[:, t].T @ dt; dbf += dt.sum(0)
            for bi in range(B):
                dE[X_idx[bi, t]] += dt[bi] @ Wxf.T
            dhf = dt @ Whf.T
        for t in range(T):
            dt = (1 - np.tanh(AB[:, t]) ** 2) * dhb
            dWxb += X[:, t].T @ dt; dWhb += HB[:, t + 1].T @ dt; dbb += dt.sum(0)
            for bi in range(B):
                dE[X_idx[bi, t]] += dt[bi] @ Wxb.T
            dhb = dt @ Whb.T
        return [dE, dWxf, dWhf, dbf, dWxb, dWhb, dbb, dWo, dbo]
