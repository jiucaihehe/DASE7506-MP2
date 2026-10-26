"""Round: arithmetic-verification features."""
import json
import numpy as np

from common import DATA, load_inputs
from train import get_data
from train2 import acc_hard, group_mask_hard
from train6 import fit_fullbatch
from arith import arith_features

tr_bank, tr_arrays = load_inputs('train', DATA)
va_bank, va_arrays = load_inputs('validation', DATA)

def arith_matrix(bank):
    rows = []
    for row in bank:
        A = np.array(arith_features([c['text'] for c in row['candidates']]))
        Z = (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-9)
        rows.append(np.hstack([A, Z]))
    return np.vstack(rows)

Atr, Ava = arith_matrix(tr_bank), arith_matrix(va_bank)
np.savez('runs/arith_cache.npz', Atr=Atr, Ava=Ava)

Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
va_hard = group_mask_hard(yva); tr_hard = group_mask_hard(ytr)
ng = len(ytr) // 4

Y = ytr.reshape(-1)
hw = Atr[:, 3]
print('P(correct | has_wrong=1):', round(float(Y[hw == 1].mean()), 4), 'n =', int((hw == 1).sum()))
print('P(correct | has_wrong=0):', round(float(Y[hw == 0].mean()), 4), 'n =', int((hw == 0).sum()))
print('coverage (n_checked>0):', round(float((Atr[:, 0] > 0).mean()), 3))

def report(tag, w, b, mu, sd, Xtr_, Xva_):
    print(json.dumps({'tag': tag,
                      'train_hard': acc_hard(((Xtr_ - mu) / sd) @ w + b, tr_answers, ytr, 'argmax', tr_hard),
                      'val_all': acc_hard(((Xva_ - mu) / sd) @ w + b, va_answers, yva),
                      'val_hard': acc_hard(((Xva_ - mu) / sd) @ w + b, va_answers, yva, 'argmax', va_hard)}), flush=True)

Xtr2 = np.hstack([Xtr, Atr]); Xva2 = np.hstack([Xva, Ava])
for lam in (0.02, 0.05):
    w, b, mu, sd = fit_fullbatch(Xtr2, ytr, ng, lam=lam, lr=0.02, epochs=1500)
    report(f'arith lam={lam}', w, b, mu, sd, Xtr2, Xva2)
