"""Approach 5: smooth margin listwise loss + permutation augmentation.

Margin loss per group: softplus(LSE(wrong scores) - LSE(correct scores))
(smooth approximation of 'best correct must beat best wrong').
Augmentation: train on K seeded permutations of each group's candidate order.
"""
import json
import numpy as np

from train import get_data
from train2 import acc_hard, group_mask_hard

Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
va_hard = group_mask_hard(yva); tr_hard = group_mask_hard(ytr)
ng = len(ytr) // 4


def fit_margin(X, y, n_groups, lam=0.02, lr=0.02, epochs=1500, perms=1, temp=1.0, seed=7506):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = ((X - mu) / sd).reshape(n_groups, 4, -1)
    Y = y.reshape(n_groups, 4)
    if perms > 1:
        X0, Y0 = Xs, Y
        for _ in range(perms - 1):
            perm = np.stack([rng.permutation(4) for _ in range(X0.shape[0])])
            Xs = np.concatenate([Xs, np.take_along_axis(X0, perm[:, :, None], axis=1)], axis=0)
            Y = np.concatenate([Y, np.take_along_axis(Y0, perm, axis=1)], axis=0)
        n_groups = Xs.shape[0]
    w = np.zeros(X.shape[1]); b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = 0.0; vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        s = Xs @ w + b
        sc = np.where(Y > 0, s, -1e9)
        sw = np.where(Y > 0, -1e9, s)
        lc = temp * np.log(np.exp(sc / temp).sum(axis=1) + 1e-12)   # LSE over correct
        lw = temp * np.log(np.exp(sw / temp).sum(axis=1) + 1e-12)   # LSE over wrong
        m = lw - lc
        loss = np.logaddexp(0.0, m)                                  # softplus
        dloss = 1.0 / (1.0 + np.exp(-m))                             # d softplus/dm
        # d lw/d s_j: softmax over wrong
        pw = np.where(Y > 0, 0.0, np.exp(sw - lw[:, None]))
        pc = np.where(Y > 0, np.exp(sc - lc[:, None]), 0.0)
        ds = (pw - pc) * dloss[:, None]
        # skip all-correct groups (no wrong candidates)
        has_wrong = (Y.sum(axis=1) < 4)
        ds = ds * has_wrong[:, None]
        gw = (Xs * ds[:, :, None]).sum(axis=(0, 1)) / n_groups + lam * w
        gb = float(ds.sum() / n_groups)
        mw = b1*mw + (1-b1)*gw; vw = b2*vw + (1-b2)*gw**2
        mb = b1*mb + (1-b1)*gb; vb = b2*vb + (1-b2)*gb**2
        w -= lr * (mw/(1-b1**t)) / (np.sqrt(vw/(1-b2**t)) + eps)
        b -= lr * (mb/(1-b1**t)) / (np.sqrt(vb/(1-b2**t)) + eps)
    return w, b, mu, sd


def report(tag, w, b, mu, sd):
    print(json.dumps({'tag': tag,
                      'train_hard': acc_hard(((Xtr - mu) / sd) @ w + b, tr_answers, ytr, 'argmax', tr_hard),
                      'val_all': acc_hard(((Xva - mu) / sd) @ w + b, va_answers, yva),
                      'val_hard': acc_hard(((Xva - mu) / sd) @ w + b, va_answers, yva, 'argmax', va_hard)}), flush=True)

for tag, kw in (
    ('margin perms=1 temp=1', dict(perms=1, temp=1.0)),
    ('margin perms=1 temp=0.5', dict(perms=1, temp=0.5)),
    ('margin perms=4 aug', dict(perms=4, temp=1.0)),
    ('margin perms=4 temp=0.5', dict(perms=4, temp=0.5)),
):
    w, b, mu, sd = fit_margin(Xtr, ytr, ng, **kw)
    report(tag, w, b, mu, sd)
