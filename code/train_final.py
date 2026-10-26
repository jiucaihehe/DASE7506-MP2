"""Final training: deterministic full-batch group-softmax linear model.

Objective per question group: -log(sum of softmax probabilities over the
correct candidates) — directly matches the scoring rule (any correct pick
earns the point). All-correct groups contribute zero loss and are ignored
automatically.

Fits on the train split only. Writes runs/model_bundle.npz (float32) plus a
config/training record. Reproduces with fixed zero init and full-batch Adam:
no randomness anywhere in this script.

Usage:  python train_final.py
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import build_matrix

LAM, LR, EPOCHS = 0.02, 0.02, 1500
OUT = Path('runs/model_bundle.npz')

HAND_NAMES = [
    'log_len', 'log_digits', 'log_eq', 'has_boxed', 'has_hash', 'has_answer_kw', 'log_newlines',
    'parsed_ok', 'agree', 'all_agree', 'emb_norm', 'sim_centroid', 'sim_mean', 'sim_max',
    'position', 'hedges', 'q_overlap', 'q_len', 'q_nums',
    'log_abs_val', 'is_integer', 'is_zero', 'in_question_vals',
]
FEATURE_NAMES = HAND_NAMES + [f'z_{n}' for n in HAND_NAMES] + [f'emb_{i}' for i in range(1024)]


def labels(bank, split):
    gold = {r['id']: parse_answer(r['answer']) for r in read_jsonl(DATA / f'gsm8k_{split}.jsonl')}
    y = np.zeros(len(bank) * 4)
    answers = []
    for i, row in enumerate(bank):
        ca = [parse_answer(c['text']) for c in row['candidates']]
        answers.append(ca)
        for j in range(4):
            y[4 * i + j] = float(ca[j] is not None and ca[j] == gold[row['id']])
    return y, answers


def fit(X, y, n_groups, lam=LAM, lr=LR, epochs=EPOCHS):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = ((X - mu) / sd).reshape(n_groups, 4, -1)
    Y = y.reshape(n_groups, 4)
    w = np.zeros(X.shape[1]); b = 0.0
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = 0.0; vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        s = Xs @ w + b
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True)
        S = (p * Y).sum(axis=1, keepdims=True) + 1e-12
        ds = p - Y * p / S
        gw = (Xs * ds[:, :, None]).sum(axis=(0, 1)) / n_groups + lam * w
        gb = float(ds.sum() / n_groups)
        mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw ** 2
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
        w -= lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
    return w, b, mu, sd


def pick(scores):
    return scores.reshape(-1, 4).argmax(axis=1)


def accuracy(choices, y):
    return float(np.mean([y[4 * i + c] for i, c in enumerate(choices)]))


def main():
    t0 = time.perf_counter()
    tr_bank, tr_arrays = load_inputs('train', DATA)
    va_bank, va_arrays = load_inputs('validation', DATA)
    ytr, _ = labels(tr_bank, 'train')
    yva, _ = labels(va_bank, 'validation')
    Xtr = build_matrix(tr_bank, tr_arrays, include_embedding=True)
    Xva = build_matrix(va_bank, va_arrays, include_embedding=True)
    assert Xtr.shape[1] == len(FEATURE_NAMES), (Xtr.shape, len(FEATURE_NAMES))

    w, b, mu, sd = fit(Xtr, ytr, len(tr_bank))
    train_sec = time.perf_counter() - t0

    ch_tr = pick(((Xtr - mu) / sd) @ w + b)
    ch_va = pick(((Xva - mu) / sd) @ w + b)
    hard_tr = np.array([0 < ytr[4 * i:4 * i + 4].sum() < 4 for i in range(len(tr_bank))])
    hard_va = np.array([0 < yva[4 * i:4 * i + 4].sum() < 4 for i in range(len(va_bank))])
    metrics = {
        'train_acc_all': round(100 * accuracy(ch_tr, ytr), 2),
        'train_acc_hard': round(100 * accuracy(ch_tr[hard_tr], ytr.reshape(-1, 4)[hard_tr].ravel()), 2),
        'val_acc_all': round(100 * accuracy(ch_va, yva), 2),
        'val_acc_hard': round(100 * accuracy(ch_va[hard_va], yva.reshape(-1, 4)[hard_va].ravel()), 2),
        'train_seconds': round(train_sec, 1),
        'hyperparams': {'lam': LAM, 'lr': LR, 'epochs': EPOCHS, 'objective': 'group-softmax',
                        'init': 'zeros', 'optimizer': 'full-batch Adam', 'seed': 'deterministic (no sampling)'},
    }
    print(json.dumps(metrics, indent=2))

    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, w=w.astype(np.float32), b=np.float32(b),
             mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    Path('runs/train_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

    # top hand-feature weights for the report
    order = np.argsort(-np.abs(w[:92]))
    print('\nTop hand-feature weights:')
    for k in order[:15]:
        print(f'  {FEATURE_NAMES[k]:>16s}  {w[k]:+.3f}')
    asset_bytes = OUT.stat().st_size
    print(f'\nasset: {OUT} = {asset_bytes} bytes ({asset_bytes / 2**20:.4f} MiB)')


if __name__ == '__main__':
    main()
