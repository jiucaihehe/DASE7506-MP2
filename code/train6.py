"""Deterministic full-batch group-softmax: no sampling noise."""
import json
import time
import numpy as np

from train import get_data
from train2 import acc_hard, group_mask_hard


def fit_fullbatch(X, y, n_groups, lam, lr, epochs):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = ((X - mu) / sd).reshape(n_groups, 4, -1)
    Y = y.reshape(n_groups, 4)
    w = np.zeros(X.shape[1]); b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = 0.0; vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        s = Xs @ w + b                       # (G,4)
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True)
        S = (p * Y).sum(axis=1, keepdims=True) + 1e-12
        ds = p - Y * p / S                   # (G,4)
        gw = (Xs * ds[:, :, None]).sum(axis=(0, 1)) / n_groups + lam * w
        gb = float(ds.sum() / n_groups)
        mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw ** 2
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
        w -= lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
    return w, b, mu, sd


def main():
    Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
    va_hard = group_mask_hard(yva)
    tr_hard = group_mask_hard(ytr)
    ng_tr = len(ytr) // 4
    for lam in (0.01, 0.02, 0.03):
        for lr, epochs in ((0.05, 300), (0.05, 800), (0.02, 1500)):
            t0 = time.perf_counter()
            w, b, mu, sd = fit_fullbatch(Xtr, ytr, ng_tr, lam, lr, epochs)
            dt = time.perf_counter() - t0
            str_ = ((Xtr - mu) / sd) @ w + b
            sva = ((Xva - mu) / sd) @ w + b
            print(json.dumps({'tag': f'fb lam={lam} lr={lr} ep={epochs}', 'sec': round(dt, 1),
                              'train_hard': acc_hard(str_, tr_answers, ytr, 'argmax', tr_hard),
                              'val_all': acc_hard(sva, va_answers, yva),
                              'val_hard': acc_hard(sva, va_answers, yva, 'argmax', va_hard)}), flush=True)


if __name__ == '__main__':
    main()
