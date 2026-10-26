"""Approach 4: kNN retrieval scoring on PCA-64 embeddings (non-parametric)."""
import json
import numpy as np
from sklearn.decomposition import PCA

from train import get_data, SEED
from train2 import acc_hard, group_mask_hard
from train6 import fit_fullbatch

Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
n_hand = Xtr.shape[1] - 1024
va_hard = group_mask_hard(yva); tr_hard = group_mask_hard(ytr)
ng = len(ytr) // 4

Etr, Eva = Xtr[:, n_hand:], Xva[:, n_hand:]
pca = PCA(n_components=64, random_state=SEED).fit(Etr)
Ptr = pca.transform(Etr); Pva = pca.transform(Eva)
Utr = Ptr / np.maximum(np.linalg.norm(Ptr, axis=1, keepdims=True), 1e-12)
Uva = Pva / np.maximum(np.linalg.norm(Pva, axis=1, keepdims=True), 1e-12)

def knn_scores(k):
    sim = Uva @ Utr.T  # (Nva*4, Ntr*4), chunked to save memory
    out = np.zeros(sim.shape[0])
    B = 512
    for s in range(0, sim.shape[0], B):
        blk = sim[s:s+B]
        idx = np.argpartition(-blk, k, axis=1)[:, :k]
        for r in range(blk.shape[0]):
            wsim = blk[r, idx[r]]
            wsim = np.maximum(wsim, 0) ** 2
            out[s+r] = float((wsim * ytr[idx[r]]).sum() / (wsim.sum() + 1e-12))
    return out

for k in (10, 50, 200):
    s = knn_scores(k)
    print(json.dumps({'tag': f'knn k={k}',
                      'val_all': acc_hard(s, va_answers, yva),
                      'val_hard': acc_hard(s, va_answers, yva, 'argmax', va_hard)}), flush=True)

# combine with base model probability
w, b, mu, sd = fit_fullbatch(Xtr, ytr, ng, lam=0.02, lr=0.02, epochs=1500)
sg = (((Xva - mu) / sd) @ w + b).reshape(-1, 4)
sg -= sg.max(axis=1, keepdims=True)
p_gs = np.exp(sg); p_gs /= p_gs.sum(axis=1, keepdims=True)
s_knn = knn_scores(50).reshape(-1, 4)
for a in (0.5, 0.7, 0.85):
    p = a * p_gs + (1 - a) * s_knn
    print(json.dumps({'tag': f'gs*{a}+knn*{1-a}',
                      'val_all': acc_hard(p.ravel(), va_answers, yva),
                      'val_hard': acc_hard(p.ravel(), va_answers, yva, 'argmax', va_hard)}), flush=True)
