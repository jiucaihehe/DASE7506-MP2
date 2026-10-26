"""Round: independent-agreement features, block-wise L2, hard-group reweighting."""
import json
import numpy as np

from train import get_data
from train2 import acc_hard, group_mask_hard


def fit(X, y, n_groups, lam, lr=0.02, epochs=1500, group_w=None):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = ((X - mu) / sd).reshape(n_groups, 4, -1)
    Y = y.reshape(n_groups, 4)
    if group_w is None:
        group_w = np.ones(n_groups)
    gw4 = group_w[:, None]
    wsum = group_w.sum()
    lam_vec = lam if np.ndim(lam) else np.full(X.shape[1], float(lam))
    w = np.zeros(X.shape[1]); b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = 0.0; vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        s = Xs @ w + b
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True)
        S = (p * Y).sum(axis=1, keepdims=True) + 1e-12
        ds = (p - Y * p / S) * gw4
        gw = (Xs * ds[:, :, None]).sum(axis=(0, 1)) / wsum + lam_vec * w
        gb = float(ds.sum() / wsum)
        mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw ** 2
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
        w -= lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
    return w, b, mu, sd


def main():
    Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
    n_hand = Xtr.shape[1] - 1024
    print('hand dim:', n_hand)
    va_hard = group_mask_hard(yva)
    tr_hard = group_mask_hard(ytr)
    ng = len(ytr) // 4

    def report(tag, w, b, mu, sd):
        print(json.dumps({'tag': tag,
                          'train_hard': acc_hard(((Xtr - mu) / sd) @ w + b, tr_answers, ytr, 'argmax', tr_hard),
                          'val_all': acc_hard(((Xva - mu) / sd) @ w + b, va_answers, yva),
                          'val_hard': acc_hard(((Xva - mu) / sd) @ w + b, va_answers, yva, 'argmax', va_hard)}), flush=True)

    # A: new features, baseline lam
    w, b, mu, sd = fit(Xtr, ytr, ng, 0.02)
    report('A new-feats lam=0.02', w, b, mu, sd)

    # B: block-wise lambda: hand 0.005, emb 0.02/0.05
    for lh, le in ((0.005, 0.02), (0.005, 0.05), (0.01, 0.05)):
        lam_vec = np.concatenate([np.full(n_hand, lh), np.full(1024, le)])
        w, b, mu, sd = fit(Xtr, ytr, ng, lam_vec)
        report(f'B block-lam {lh}/{le}', w, b, mu, sd)

    # C: upweight groups where the correct answer is a minority
    Y4 = ytr.reshape(ng, 4)
    # need answers to know cluster sizes; approximate: correct count <= 2 and ...
    ncorr = Y4.sum(axis=1)
    gw = np.ones(ng)
    gw[(ncorr == 1)] = 1.5   # single correct candidate groups
    w, b, mu, sd = fit(Xtr, ytr, ng, 0.02, group_w=gw)
    report('C upweight ncorr==1 x1.5', w, b, mu, sd)
    gw2 = np.ones(ng); gw2[(ncorr == 1)] = 2.0
    w, b, mu, sd = fit(Xtr, ytr, ng, 0.02, group_w=gw2)
    report('C upweight ncorr==1 x2', w, b, mu, sd)


if __name__ == '__main__':
    main()
