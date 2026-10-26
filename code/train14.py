"""Approach 3: cluster-level answer model (two-stage).

Stage 1: group-softmax over ANSWER CLUSTERS using cluster-aggregate features
(padded to 4 clusters per group). Stage 2: within the winning cluster, pick
the candidate with the highest base group-softmax score.
"""
import json
import numpy as np

from common import DATA, load_inputs, parse_answer
from train import get_data
from train2 import acc_hard, group_mask_hard
from train6 import fit_fullbatch
from features import group_features

tr_bank, tr_arrays = load_inputs('train', DATA)
va_bank, va_arrays = load_inputs('validation', DATA)
Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
va_hard = group_mask_hard(yva); tr_hard = group_mask_hard(ytr)
ng_tr = len(ytr) // 4

# base candidate model for within-cluster selection
wb, bb, mub, sdb = fit_fullbatch(Xtr, ytr, ng_tr, lam=0.02, lr=0.02, epochs=1500)

def clusterize(bank, arrays, X, gold_y=None):
    """Return per-group cluster feature matrices (4 x F, padded), cluster->candidate map,
    and cluster correctness labels (train only)."""
    all_F, all_lab, all_map = [], [], []
    for i, row in enumerate(bank):
        texts = [c['text'] for c in row['candidates']]
        ans = [parse_answer(t) for t in texts]
        emb = arrays['embeddings'][i].astype(np.float64)
        unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        clusters = {}
        for j in range(4):
            key = ans[j] if ans[j] is not None else f'none{j}'
            clusters.setdefault(key, []).append(j)
        keys = sorted(clusters, key=lambda k: -len(clusters[k]))
        F = np.zeros((4, 12)); lab = np.zeros(4); cmap = {}
        for ci, key in enumerate(keys[:4]):
            members = clusters[key]
            E = unit[members]
            spread = float(np.mean([E[a] @ E[b_] for a in range(len(members)) for b_ in range(a+1, len(members))])) if len(members) > 1 else 0.0
            hf = group_features(texts, emb, row['question'])  # (4, 46+) raw+z per candidate
            mh = hf[members]
            val = 0.0
            if not key.startswith('none'):
                try:
                    val = float(np.log1p(abs(float(key.split('/')[0]) / (float(key.split('/')[1]) if '/' in key else 1.0))))
                except Exception:
                    pass
            F[ci] = [len(members), len(members)/4, spread,
                     float(mh[:, 0].mean()), float(mh[:, 0].max()),      # log_len mean/max
                     float(mh[:, 1].mean()), float(mh[:, 2].mean()),      # digits, eqs
                     float(mh[:, 15].mean()),                              # hedges
                     float(mh[:, 12].mean()),                              # sim_centroid
                     float(mh[:, 19].mean()),                              # log_abs_val
                     val,
                     float(mh[:, 3].max())]                                # has_boxed any
            cmap[ci] = members
            if gold_y is not None:
                lab[ci] = float(any(gold_y[4*i+j] > 0 for j in members))
        all_F.append(F); all_lab.append(lab); all_map.append((keys[:4], cmap))
    return np.array(all_F), np.array(all_lab), all_map

Ftr, Ltr, _ = clusterize(tr_bank, tr_arrays, Xtr, ytr)
Fva, _, map_va = clusterize(va_bank, va_arrays, Xva)

# group-softmax over clusters (masked)
def fit_clusters(F, L, lam=0.05, lr=0.02, epochs=1500):
    G, K, D = F.shape
    mu, sd = F.reshape(-1, D).mean(0), F.reshape(-1, D).std(0) + 1e-9
    Fs = (F - mu) / sd
    mask = (F.sum(axis=2) != 0)  # real clusters (padded rows are all zero)
    w = np.zeros(D); b = 0.0
    mw = np.zeros(D); vw = np.zeros(D); mb = 0.0; vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs+1):
        s = Fs @ w + b
        s = np.where(mask, s, -1e9)
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s) * mask; p /= p.sum(axis=1, keepdims=True)
        S = (p * L).sum(axis=1, keepdims=True) + 1e-12
        ds = p - L * p / S
        gw = (Fs * ds[:, :, None]).sum(axis=(0, 1)) / G + lam * w
        gb = float(ds.sum() / G)
        mw = b1*mw + (1-b1)*gw; vw = b2*vw + (1-b2)*gw**2
        mb = b1*mb + (1-b1)*gb; vb = b2*vb + (1-b2)*gb**2
        w -= lr * (mw/(1-b1**t)) / (np.sqrt(vw/(1-b2**t)) + eps)
        b -= lr * (mb/(1-b1**t)) / (np.sqrt(vb/(1-b2**t)) + eps)
    return w, b, mu, sd

wc, bc, muc, sdc = fit_clusters(Ftr, Ltr)

# inference: pick winning cluster, then best base-score candidate inside
s_base = ((Xva - mub) / sdb) @ wb + bb
Fvs = (Fva - muc) / sdc
sc = Fvs @ wc + bc
choices = []
for i in range(len(va_bank)):
    keys, cmap = map_va[i]
    n_real = len(keys)
    win = int(np.argmax(sc[i, :n_real]))
    members = cmap[win]
    choices.append(max(members, key=lambda j: s_base[4*i+j]))
acc = 100*np.mean([yva[4*i+c] for i, c in enumerate(choices)])
acc_h = 100*np.mean([yva[4*i+choices[i]] for i in np.where(va_hard)[0]])
print(json.dumps({'tag': 'cluster-two-stage', 'val_all': round(float(acc), 2), 'val_hard': round(float(acc_h), 2)}))

# cluster oracle: how often is the winning cluster correct at all?
win_correct = []
for i in range(len(va_bank)):
    keys, cmap = map_va[i]
    win = int(np.argmax(sc[i, :len(keys)]))
    win_correct.append(any(yva[4*i+j] > 0 for j in cmap[win]))
print('winning-cluster correctness:', round(100*float(np.mean(win_correct)), 2))
