"""Round: ensemble of group-softmax and pointwise-LR probabilities."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

from train import get_data, SEED
from train2 import acc_hard, group_mask_hard
from train6 import fit_fullbatch

Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
va_hard = group_mask_hard(yva); tr_hard = group_mask_hard(ytr)
ng = len(ytr) // 4

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
lr_clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit((Xtr - mu) / sd, ytr)
p_lr = lr_clf.predict_proba((Xva - mu) / sd)[:, 1].reshape(-1, 4)
p_lr /= p_lr.sum(axis=1, keepdims=True)

w, b, mu2, sd2 = fit_fullbatch(Xtr, ytr, ng, lam=0.02, lr=0.02, epochs=1500)
s = (((Xva - mu2) / sd2) @ w + b).reshape(-1, 4)
s -= s.max(axis=1, keepdims=True)
p_gs = np.exp(s); p_gs /= p_gs.sum(axis=1, keepdims=True)

for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    p = a * p_gs + (1 - a) * p_lr
    print(json.dumps({'tag': f'ens a={a} (gs share)',
                      'val_all': acc_hard(p.ravel(), va_answers, yva),
                      'val_hard': acc_hard(p.ravel(), va_answers, yva, 'argmax', va_hard)}), flush=True)
