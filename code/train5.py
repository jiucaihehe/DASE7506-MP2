"""Seed robustness + weight-averaged ensemble for group-softmax."""
import json
import numpy as np

from train import get_data, SEED
from train2 import acc_hard, group_mask_hard
from train4 import group_softmax_fit, scores


def main():
    Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
    va_hard = group_mask_hard(yva)
    tr_hard = group_mask_hard(ytr)
    ng_tr = len(ytr) // 4

    ws, bs = [], []
    mu = sd = None
    for seed in (7506, 1, 2, 3, 4):
        w, b, mu, sd = group_softmax_fit(Xtr, ytr, ng_tr, lam=2e-2, lr=0.05, epochs=200, seed=seed)
        sva = scores(Xva, w, b, mu, sd)
        print(json.dumps({'seed': seed,
                          'val_all': acc_hard(sva, va_answers, yva),
                          'val_hard': acc_hard(sva, va_answers, yva, 'argmax', va_hard)}), flush=True)
        ws.append(w); bs.append(b)
    w_avg = np.mean(ws, axis=0); b_avg = float(np.mean(bs))
    sva = scores(Xva, w_avg, b_avg, mu, sd)
    str_ = scores(Xtr, w_avg, b_avg, mu, sd)
    print(json.dumps({'tag': 'avg-5-seeds',
                      'train_hard': acc_hard(str_, tr_answers, ytr, 'argmax', tr_hard),
                      'val_all': acc_hard(sva, va_answers, yva),
                      'val_hard': acc_hard(sva, va_answers, yva, 'argmax', va_hard)}))
    # ensemble by averaging per-candidate softmax probabilities instead of weights
    probs = []
    for (w, b) in zip(ws, bs):
        s = scores(Xva, w, b, mu, sd).reshape(-1, 4)
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True)
        probs.append(p.ravel())
    p_ens = np.mean(probs, axis=0)
    print(json.dumps({'tag': 'ens-5-probs',
                      'val_all': acc_hard(p_ens, va_answers, yva),
                      'val_hard': acc_hard(p_ens, va_answers, yva, 'argmax', va_hard)}))


if __name__ == '__main__':
    main()
