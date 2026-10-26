"""Group-softmax model: directly optimizes P(pick a correct candidate).

For a group with scores s and correct set C, loss = -log(sum_{j in C} softmax(s)_j).
Linear model s = X @ w + b, trained with Adam, L2. Numpy only.
"""
import json

import numpy as np

from train import get_data, SEED
from train2 import acc_hard, group_mask_hard


def group_softmax_fit(X, y, n_groups, lam=1e-3, lr=0.05, epochs=300, seed=SEED, batch=512):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(X.shape[1])
    b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = 0.0; vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n_groups)
        for start in range(0, n_groups, batch):
            gids = order[start:start + batch]
            t += 1
            gw = np.zeros_like(w); gb = 0.0
            for g in gids:
                xg = Xs[4 * g:4 * g + 4]
                yg = y[4 * g:4 * g + 4]
                s = xg @ w + b
                s = s - s.max()
                p = np.exp(s); p /= p.sum()
                S = float((p * yg).sum()) + 1e-12
                ds = p - yg * p / S  # dL/ds
                gw += xg.T @ ds
                gb += ds.sum()
            gw = gw / len(gids) + lam * w
            gb /= len(gids)
            mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw ** 2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
            w -= lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps)
            b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
    return w, b, mu, sd


def scores(X, w, b, mu, sd):
    return (X - mu) / sd @ w + b


def main():
    Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
    n_hand = Xtr.shape[1] - 1024
    va_hard = group_mask_hard(yva)
    tr_hard = group_mask_hard(ytr)
    ng_tr = len(ytr) // 4

    def report(tag, w, b, mu, sd):
        str_ = scores(Xtr, w, b, mu, sd)
        sva = scores(Xva, w, b, mu, sd)
        # turn scores into pseudo-probs for acc helper (argmax only used)
        print(json.dumps({
            'tag': tag,
            'train_hard': acc_hard(str_, tr_answers, ytr, 'argmax', tr_hard),
            'val_all': acc_hard(sva, va_answers, yva),
            'val_hard': acc_hard(sva, va_answers, yva, 'argmax', va_hard),
        }), flush=True)

    for lam, lr, epochs, subset in ((1e-2, 0.05, 200, 'hand'), (5e-3, 0.05, 200, 'all'),
                                    (1e-2, 0.05, 200, 'all'), (2e-2, 0.05, 200, 'all'),
                                    (1e-2, 0.05, 400, 'all'), (3e-2, 0.05, 200, 'all'),
                                    (5e-2, 0.05, 200, 'all')):
        if subset == 'hand':
            w, b, mu, sd = group_softmax_fit(Xtr[:, :n_hand], ytr, ng_tr, lam, lr, epochs)
            print(json.dumps({'tag': f'gs hand lam={lam}',
                              'train_hard': acc_hard(scores(Xtr[:, :n_hand], w, b, mu, sd), tr_answers, ytr, 'argmax', tr_hard),
                              'val_all': acc_hard(scores(Xva[:, :n_hand], w, b, mu, sd), va_answers, yva),
                              'val_hard': acc_hard(scores(Xva[:, :n_hand], w, b, mu, sd), va_answers, yva, 'argmax', va_hard)}), flush=True)
        else:
            w, b, mu, sd = group_softmax_fit(Xtr, ytr, ng_tr, lam, lr, epochs)
            report(f'gs all lam={lam} lr={lr} ep={epochs}', w, b, mu, sd)


if __name__ == '__main__':
    main()
