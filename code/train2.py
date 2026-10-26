"""Model comparison focused on HARD groups (mixed correctness), the test-like
proxy, plus ablations for the report. Train split only for fitting."""
import json
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from common import DATA
from train import acc_from_probs, get_data, labels_for, SEED


def group_mask_hard(y):
    n = len(y) // 4
    return np.array([0 < y[4 * i:4 * i + 4].sum() < 4 for i in range(n)])


def acc_hard(probs, answers, y, strategy='argmax', mask=None):
    n = len(answers)
    correct = total = 0
    for i in range(n):
        if mask is not None and not mask[i]:
            continue
        p = probs[4 * i:4 * i + 4]
        if strategy == 'argmax':
            c = int(np.argmax(p))
        elif strategy == 'majority':
            counts = Counter(a for a in answers[i] if a is not None)
            c = next((j for j, a in enumerate(answers[i]) if a == counts.most_common(1)[0][0]), 0) if counts else 0
        correct += int(y[4 * i + c] > 0)
        total += 1
    return round(100 * correct / total, 2)


def main():
    Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
    n_hand = Xtr.shape[1] - 1024
    va_hard = group_mask_hard(yva)
    tr_hard = group_mask_hard(ytr)
    print(f'val groups: {len(va_hard)}, hard: {int(va_hard.sum())}')
    print(f'train groups: {len(tr_hard)}, hard: {int(tr_hard.sum())}')

    # majority baseline on hard subsets
    print('majority val-all:', acc_hard(np.zeros(len(yva)), va_answers, yva, 'majority'),
          'val-hard:', acc_hard(np.zeros(len(yva)), va_answers, yva, 'majority', va_hard))

    def fit_eval(clf, X_tr, X_va, tag, scale=True):
        mu, sd = (X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9) if scale else (0, 1)
        clf.fit((X_tr - mu) / sd, ytr)
        pva = clf.predict_proba((X_va - mu) / sd)[:, 1] if hasattr(clf, 'predict_proba') else None
        ptr = clf.predict_proba((X_tr - mu) / sd)[:, 1]
        print(json.dumps({
            'tag': tag,
            'train_all': acc_hard(ptr, tr_answers, ytr),
            'train_hard': acc_hard(ptr, tr_answers, ytr, 'argmax', tr_hard),
            'val_all': acc_hard(pva, va_answers, yva),
            'val_hard': acc_hard(pva, va_answers, yva, 'argmax', va_hard),
        }), flush=True)
        return pva

    # ablations
    fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), Xtr[:, :n_hand], Xva[:, :n_hand], 'LR hand')
    fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), Xtr[:, n_hand:], Xva[:, n_hand:], 'LR emb-only')
    pva_lr = fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), Xtr, Xva, 'LR hand+emb')
    fit_eval(LogisticRegression(C=0.1, max_iter=2000, random_state=SEED), Xtr, Xva, 'LR hand+emb C=0.1')
    fit_eval(HistGradientBoostingClassifier(random_state=SEED, max_iter=100, learning_rate=0.05,
                                            max_depth=3, min_samples_leaf=100),
             Xtr[:, :n_hand], Xva[:, :n_hand], 'GBM hand tuned', scale=False)
    fit_eval(MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=300, random_state=SEED,
                           early_stopping=True, validation_fraction=0.1),
             Xtr, Xva, 'MLP-64 a=1')

    # simple ensemble LR + majority bonus on hard cases
    # score = logit(p) + alpha * agree ; tune alpha on val-hard
    va_logits = np.log(pva_lr / (1 - pva_lr + 1e-12) + 1e-12)
    # recompute agree counts for val
    best = None
    for alpha in (-0.5, 0.0, 0.5, 1.0):
        probs = np.zeros_like(pva_lr)
        for i in range(len(va_answers)):
            for j in range(4):
                agree = sum(1 for k in range(4) if k != j and va_answers[i][k] is not None
                            and va_answers[i][k] == va_answers[i][j])
                probs[4 * i + j] = va_logits[4 * i + j] + alpha * agree
        a_all = acc_hard(probs, va_answers, yva)
        a_h = acc_hard(probs, va_answers, yva, 'argmax', va_hard)
        print(json.dumps({'tag': f'LR+{alpha}*agree', 'val_all': a_all, 'val_hard': a_h}))


if __name__ == '__main__':
    main()
