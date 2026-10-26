"""Round: consistency features + tiny MLP with group-softmax + PCA ablation."""
import json
import numpy as np

from train import get_data
from train2 import acc_hard, group_mask_hard
from train6 import fit_fullbatch


def fit_mlp_gs(X, y, n_groups, hidden=32, lam=1e-2, lr=0.02, epochs=800):
    rng = np.random.default_rng(7506)  # init only; training is full-batch deterministic
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = ((X - mu) / sd).reshape(n_groups, 4, -1)
    Y = y.reshape(n_groups, 4)
    d = X.shape[1]
    W1 = rng.normal(0, 0.05, (d, hidden)); b1 = np.zeros(hidden)
    w2 = np.zeros(hidden); b2 = 0.0
    params = [W1, b1, w2, np.array([b2])]
    ms = [np.zeros_like(p) for p in params]; vs = [np.zeros_like(p) for p in params]
    B1, B2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        H = np.tanh(Xs @ W1 + b1)            # (G,4,h)
        s = H @ w2 + b2                      # (G,4)
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s); p /= p.sum(axis=1, keepdims=True)
        S = (p * Y).sum(axis=1, keepdims=True) + 1e-12
        ds = p - Y * p / S
        gw2 = (H * ds[:, :, None]).sum(axis=(0, 1)) / n_groups + lam * w2
        gb2 = ds.sum() / n_groups
        dH = ds[:, :, None] * w2[None, None, :] * (1 - H ** 2)
        gW1 = (Xs.transpose(0, 2, 1) @ dH).sum(axis=0) / n_groups + lam * W1
        gb1 = dH.sum(axis=(0, 1)) / n_groups
        grads = [gW1, gb1, gw2, np.array([gb2])]
        for i in range(4):
            ms[i] = B1 * ms[i] + (1 - B1) * grads[i]
            vs[i] = B2 * vs[i] + (1 - B2) * grads[i] ** 2
            params[i] -= lr * (ms[i] / (1 - B1 ** t)) / (np.sqrt(vs[i] / (1 - B2 ** t)) + eps)
    return params, mu, sd


def mlp_scores(X, params, mu, sd):
    W1, b1, w2, b2 = params[0], params[1], params[2], float(params[3][0])
    return np.tanh(((X - mu) / sd) @ W1 + b1) @ w2 + b2


def main():
    Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
    n_hand = Xtr.shape[1] - 1024
    print('feature dim:', Xtr.shape[1], 'hand:', n_hand)
    va_hard = group_mask_hard(yva)
    tr_hard = group_mask_hard(ytr)
    ng_tr = len(ytr) // 4

    def report(tag, str_, sva):
        print(json.dumps({'tag': tag,
                          'train_hard': acc_hard(str_, tr_answers, ytr, 'argmax', tr_hard),
                          'val_all': acc_hard(sva, va_answers, yva),
                          'val_hard': acc_hard(sva, va_answers, yva, 'argmax', va_hard)}), flush=True)

    # linear group-softmax with new features (reference plateau setting)
    w, b, mu, sd = fit_fullbatch(Xtr, ytr, ng_tr, lam=0.02, lr=0.02, epochs=1500)
    report('linear new-feats', ((Xtr - mu) / sd) @ w + b, ((Xva - mu) / sd) @ w + b)

    # hand-only / emb-only ablations with group-softmax
    w2_, b2_, mu2, sd2 = fit_fullbatch(Xtr[:, :n_hand], ytr, ng_tr, lam=0.02, lr=0.02, epochs=1500)
    report('gs hand-only', ((Xtr[:, :n_hand] - mu2) / sd2) @ w2_ + b2_, ((Xva[:, :n_hand] - mu2) / sd2) @ w2_ + b2_)
    w3_, b3_, mu3, sd3 = fit_fullbatch(Xtr[:, n_hand:], ytr, ng_tr, lam=0.02, lr=0.02, epochs=1500)
    report('gs emb-only', ((Xtr[:, n_hand:] - mu3) / sd3) @ w3_ + b3_, ((Xva[:, n_hand:] - mu3) / sd3) @ w3_ + b3_)

    # PCA on embedding block
    from sklearn.decomposition import PCA
    Xtr_e, Xva_e = Xtr[:, n_hand:], Xva[:, n_hand:]
    pca = PCA(n_components=128, random_state=7506).fit(Xtr_e)
    Xtr_p = np.hstack([Xtr[:, :n_hand], pca.transform(Xtr_e)])
    Xva_p = np.hstack([Xva[:, :n_hand], pca.transform(Xva_e)])
    w4, b4, mu4, sd4 = fit_fullbatch(Xtr_p, ytr, ng_tr, lam=0.02, lr=0.02, epochs=1500)
    report('gs hand+PCA128', ((Xtr_p - mu4) / sd4) @ w4 + b4, ((Xva_p - mu4) / sd4) @ w4 + b4)

    # tiny MLP with group-softmax
    for h, lam in ((32, 1e-2), (32, 3e-2), (64, 3e-2)):
        params, mu5, sd5 = fit_mlp_gs(Xtr, ytr, ng_tr, hidden=h, lam=lam, lr=0.02, epochs=800)
        report(f'mlp h={h} lam={lam}', mlp_scores(Xtr, params, mu5, sd5), mlp_scores(Xva, params, mu5, sd5))


if __name__ == '__main__':
    main()
