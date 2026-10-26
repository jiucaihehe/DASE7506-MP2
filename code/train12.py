"""Approach 1: reasoning-chain coherence features."""
import json
import numpy as np

from common import DATA, load_inputs
from train import get_data
from train2 import acc_hard, group_mask_hard
from train6 import fit_fullbatch
from chain import chain_features

tr_bank, tr_arrays = load_inputs('train', DATA)
va_bank, va_arrays = load_inputs('validation', DATA)

def chain_matrix(bank):
    rows = []
    for row in bank:
        C = np.array([chain_features(row['question'], c['text']) for c in row['candidates']])
        Z = (C - C.mean(0, keepdims=True)) / (C.std(0, keepdims=True) + 1e-9)
        rows.append(np.hstack([C, Z]))
    return np.vstack(rows)

Ctr, Cva = chain_matrix(tr_bank), chain_matrix(va_bank)

Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
va_hard = group_mask_hard(yva); tr_hard = group_mask_hard(ytr)
ng = len(ytr) // 4

# signal check
Y = ytr
mg = Ctr[:, 0] > 0.3
print('P(correct | magic_ratio>0.3):', round(float(Y[mg].mean()), 4), 'n =', int(mg.sum()))
print('P(correct | magic_ratio<=0.3):', round(float(Y[~mg].mean()), 4))
fg = Ctr[:, 3] == 1
print('P(correct | final_grounded=1):', round(float(Y[fg].mean()), 4), 'n =', int(fg.sum()))
print('P(correct | final_grounded=0):', round(float(Y[~fg].mean()), 4))

Xtr2 = np.hstack([Xtr, Ctr]); Xva2 = np.hstack([Xva, Cva])
for lam in (0.02, 0.05):
    w, b, mu, sd = fit_fullbatch(Xtr2, ytr, ng, lam=lam, lr=0.02, epochs=1500)
    sva = ((Xva2 - mu) / sd) @ w + b
    print(json.dumps({'tag': f'chain lam={lam}',
                      'train_hard': acc_hard(((Xtr2 - mu) / sd) @ w + b, tr_answers, ytr, 'argmax', tr_hard),
                      'val_all': acc_hard(sva, va_answers, yva),
                      'val_hard': acc_hard(sva, va_answers, yva, 'argmax', va_hard)}), flush=True)
