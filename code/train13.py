"""Approach 2: nonlinear pairwise preference model (GBM, Bradley-Terry scoring)."""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from train import get_data, SEED
from train2 import acc_hard, group_mask_hard

Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
n_hand = Xtr.shape[1] - 1024
va_hard = group_mask_hard(yva); tr_hard = group_mask_hard(ytr)

def pairs(X, y, answers, n):
    """Build ordered-pair dataset from groups with mixed labels (train only)."""
    P, T = [], []
    for i in range(n):
        yg = y[4*i:4*i+4]
        if 0 < yg.sum() < 4:
            for j in range(4):
                for k in range(4):
                    if j != k and yg[j] != yg[k]:
                        P.append(np.concatenate([X[4*i+j] - X[4*i+k], X[4*i+j]]))
                        T.append(int(yg[j] == 1))
    return np.array(P), np.array(T)

Xh_tr, Xh_va = Xtr[:, :n_hand], Xva[:, :n_hand]
Xp, yp = pairs(Xh_tr, ytr, tr_answers, len(ytr)//4)
print('pairs:', Xp.shape, 'pos rate:', round(float(yp.mean()), 3))

clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200, learning_rate=0.05,
                                     max_depth=3, min_samples_leaf=200, l2_regularization=1.0)
clf.fit(Xp, yp)

def bt_scores(X, n):
    """Bradley-Terry: score_j = mean_k P(j beats k)."""
    S = np.zeros((n, 4))
    for i in range(n):
        for j in range(4):
            for k in range(4):
                if j != k:
                    f = np.concatenate([X[4*i+j] - X[4*i+k], X[4*i+j]])[None]
                    S[i, j] += clf.predict_proba(f)[0, 1]
    return (S / 3).ravel()

for split, X, y_, answers, hard in (('train', Xh_tr, ytr, tr_answers, tr_hard),
                                    ('val', Xh_va, yva, va_answers, va_hard)):
    s = bt_scores(X, len(y_)//4)
    print(json.dumps({'split': split,
                      'all': acc_hard(s, answers, y_),
                      'hard': acc_hard(s, answers, y_, 'argmax', hard)}), flush=True)
