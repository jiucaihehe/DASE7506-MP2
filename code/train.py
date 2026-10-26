"""Train candidate-correctness models and compare selection strategies.

Development protocol: train on the train split only, select methods on
validation only. Test references are never touched here.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import build_matrix

SEED = 7506
CACHE = Path('runs/feature_cache.npz')


def labels_for(split, bank):
    gold = {r['id']: parse_answer(r['answer']) for r in read_jsonl(DATA / f'gsm8k_{split}.jsonl')}
    y = np.zeros(len(bank) * 4)
    answers = []
    for i, row in enumerate(bank):
        ca = [parse_answer(c['text']) for c in row['candidates']]
        answers.append(ca)
        for j in range(4):
            y[4 * i + j] = float(ca[j] is not None and ca[j] == gold[row['id']])
    return y, answers


def acc_from_probs(probs, answers, y, strategy):
    n = len(answers)
    correct = 0
    for i in range(n):
        p = probs[4 * i:4 * i + 4]
        if strategy == 'argmax':
            c = int(np.argmax(p))
        elif strategy == 'cluster':
            mass = {}
            for j in range(4):
                key = answers[i][j] if answers[i][j] is not None else f'__none{j}'
                mass[key] = mass.get(key, 0.0) + p[j]
            best_key = max(mass, key=mass.get)
            c = max((j for j in range(4) if (answers[i][j] if answers[i][j] is not None else f'__none{j}') == best_key),
                    key=lambda j: p[j])
        elif strategy == 'majority':
            counts = Counter(a for a in answers[i] if a is not None)
            c = next((j for j, a in enumerate(answers[i]) if a == counts.most_common(1)[0][0]), 0) if counts else 0
        correct += int(y[4 * i + c] > 0)
    return 100 * correct / n


def get_data():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z['Xtr'], z['Xva'], z['ytr'], z['yva'], z['tr_answers'].tolist(), z['va_answers'].tolist()
    tr_bank, tr_arrays = load_inputs('train', DATA)
    va_bank, va_arrays = load_inputs('validation', DATA)
    ytr, tr_answers = labels_for('train', tr_bank)
    yva, va_answers = labels_for('validation', va_bank)
    Xtr = build_matrix(tr_bank, tr_arrays, True)
    Xva = build_matrix(va_bank, va_arrays, True)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, Xtr=Xtr, Xva=Xva, ytr=ytr, yva=yva,
             tr_answers=np.array(tr_answers, dtype=object), va_answers=np.array(va_answers, dtype=object))
    return Xtr, Xva, ytr, yva, tr_answers, va_answers


def evaluate(tag, ptr, pva, ytr, yva, tr_answers, va_answers):
    out = {'tag': tag}
    for split, probs, answers, y in (('train', ptr, tr_answers, ytr), ('val', pva, va_answers, yva)):
        for strat in ('argmax', 'cluster'):
            out[f'{split}_{strat}'] = round(acc_from_probs(probs, answers, y, strat), 2)
    print(json.dumps(out), flush=True)
    return out


def main():
    Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
    n_hand = Xtr.shape[1] - 1024

    def fit_eval(clf, X_tr, X_va, tag, scale=False, sample_idx=None):
        Xt, yt = X_tr, ytr
        if sample_idx is not None:
            Xt, yt = Xt[sample_idx], yt[sample_idx]
        if scale:
            mu, sd = Xt.mean(axis=0), Xt.std(axis=0) + 1e-9
            clf.fit((Xt - mu) / sd, yt)
            ptr = clf.predict_proba((X_tr - mu) / sd)[:, 1]
            pva = clf.predict_proba((X_va - mu) / sd)[:, 1]
        else:
            clf.fit(Xt, yt)
            ptr = clf.predict_proba(X_tr)[:, 1]
            pva = clf.predict_proba(X_va)[:, 1]
        return evaluate(tag, ptr, pva, ytr, yva, tr_answers, va_answers), pva

    # hard-group subset (mixed correctness only)
    hard = [i for i in range(len(ytr) // 4) if 0 < ytr[4 * i:4 * i + 4].sum() < 4]
    hard_idx = np.concatenate([np.arange(4 * i, 4 * i + 4) for i in hard])

    fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
             Xtr[:, :n_hand], Xva[:, :n_hand], 'LR hand', scale=True)
    fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
             Xtr, Xva, 'LR hand+emb C=1', scale=True)
    fit_eval(LogisticRegression(C=0.05, max_iter=2000, random_state=SEED),
             Xtr, Xva, 'LR hand+emb C=0.05', scale=True)
    fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
             Xtr, Xva, 'LR hand+emb hard-only', scale=True, sample_idx=hard_idx)
    r_gbm, pva_gbm = fit_eval(HistGradientBoostingClassifier(random_state=SEED, max_iter=300),
                              Xtr[:, :n_hand], Xva[:, :n_hand], 'GBM hand')
    r_gbm2, pva_gbm2 = fit_eval(HistGradientBoostingClassifier(random_state=SEED, max_iter=300),
                                Xtr, Xva, 'GBM hand+emb')
    # ensemble: LR(emb) + GBM(hand)
    _, pva_lr = fit_eval(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
                         Xtr, Xva, 'LR again (for ensemble)', scale=True)
    for w in (0.3, 0.5, 0.7):
        evaluate(f'ens LR*{w}+GBM*{1-w}', None if False else np.zeros(len(ytr)), w * pva_lr + (1 - w) * pva_gbm,
                 ytr, yva, tr_answers, va_answers)


if __name__ == '__main__':
    main()
