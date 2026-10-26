"""Round 3: group-normalized features + hard-group sample weighting."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

from train import get_data, SEED
from train2 import acc_hard, group_mask_hard


def fit_eval(clf, X_tr, X_va, ytr, yva, tr_answers, va_answers, tr_hard, va_hard, tag, sample_weight=None):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9
    clf.fit((X_tr - mu) / sd, ytr, sample_weight=sample_weight)
    ptr = clf.predict_proba((X_tr - mu) / sd)[:, 1]
    pva = clf.predict_proba((X_va - mu) / sd)[:, 1]
    print(json.dumps({
        'tag': tag,
        'train_hard': acc_hard(ptr, tr_answers, ytr, 'argmax', tr_hard),
        'val_all': acc_hard(pva, va_answers, yva),
        'val_hard': acc_hard(pva, va_answers, yva, 'argmax', va_hard),
    }), flush=True)
    return pva


def main():
    Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
    n_hand = Xtr.shape[1] - 1024
    print('hand dim now:', n_hand)
    va_hard = group_mask_hard(yva)
    tr_hard = group_mask_hard(ytr)
    hard_idx = np.concatenate([np.arange(4 * i, 4 * i + 4) for i in np.where(tr_hard)[0]])
    w_hard = np.ones(len(ytr)); w_hard[hard_idx] = 2.0

    args = (Xtr, Xva, ytr, yva, tr_answers, va_answers, tr_hard, va_hard)
    fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), *args, 'LR hand+emb groupnorm')
    fit_eval(LogisticRegression(C=0.1, max_iter=2000, random_state=SEED), *args, 'LR C=0.1')
    fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), *args, 'LR w-hard x2', sample_weight=w_hard)
    hard_idx_all = hard_idx
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    mu, sd = Xtr[hard_idx_all].mean(axis=0), Xtr[hard_idx_all].std(axis=0) + 1e-9
    clf.fit((Xtr[hard_idx_all] - mu) / sd, ytr[hard_idx_all])
    pva = clf.predict_proba((Xva - mu) / sd)[:, 1]
    print(json.dumps({'tag': 'LR hard-only fit', 'val_all': acc_hard(pva, va_answers, yva),
                      'val_hard': acc_hard(pva, va_answers, yva, 'argmax', va_hard)}))
    # hand-only with groupnorm (no embedding)
    fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
             Xtr[:, :n_hand], Xva[:, :n_hand], ytr, yva, tr_answers, va_answers, tr_hard, va_hard, 'LR hand-only groupnorm')


if __name__ == '__main__':
    main()
