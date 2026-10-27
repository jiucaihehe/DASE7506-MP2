"""Train-only CV of question-conditioned scoring, separate from frozen files."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import json
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from common import DATA, load_inputs
from features import build_matrix
from train_final import labels
from experiment_context import accuracy

OUT = Path('runs/bilinear_experiment')


def fit_adam(x, y, extra_penalty=1., epochs=1500):
    """Same zero-init Adam as the frozen baseline, efficient matrix gradients."""
    mu = x.mean((0, 1)); sd = x.std((0, 1)) + 1e-9
    z = ((x - mu) / sd).reshape(-1, x.shape[-1])
    w = np.zeros(z.shape[1]); m = np.zeros_like(w); v = np.zeros_like(w)
    penalty = np.full(len(w), .02); penalty[1070:] = extra_penalty
    for step in range(1, epochs + 1):
        s = (z @ w).reshape(-1, 4)
        s -= s.max(1, keepdims=True)
        p = np.exp(s); p /= p.sum(1, keepdims=True)
        ds = p - y * p / ((y * p).sum(1, keepdims=True) + 1e-12)
        grad = z.T @ ds.ravel() / len(y) + penalty * w
        m = .9 * m + .1 * grad; v = .999 * v + .001 * grad ** 2
        w -= .02 * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
    return dict(w=w, mu=mu, sd=sd)


def score(x, model):
    return ((x - model['mu']) / model['sd']) @ model['w']


def projection(x, dim):
    e = x[..., 46:]; c = e.mean(1)
    cm = c.mean(0)
    vals, vecs = np.linalg.eigh((c - cm).T @ (c - cm) / len(c))
    cp = vecs[:, -dim:] / np.sqrt(np.maximum(vals[-dim:], 1e-8))
    r = (e - c[:, None, :]).reshape(-1, e.shape[-1])
    vals, vecs = np.linalg.eigh(r.T @ r / len(r))
    rp = vecs[:, -dim:] / np.sqrt(np.maximum(vals[-dim:], 1e-8))
    return dict(cm=cm, cp=cp, rp=rp)


def expand(x, proj, kind, dim):
    e = x[..., 46:]; c = e.mean(1)
    context = (c - proj['cm']) @ proj['cp'][:, -dim:]
    if kind == 'bilinear':
        residual = (e - c[:, None, :]) @ proj['rp'][:, -dim:]
    elif kind == 'hand_gate':
        residual = x[..., :46]
    else:
        raise ValueError(kind)
    products = (context[:, None, :, None] * residual[:, :, None, :]).reshape(len(x), 4, -1)
    return np.concatenate([x, products], axis=-1)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    start = time.perf_counter()
    bank, arrays = load_inputs('train', DATA)
    x = build_matrix(bank, arrays).reshape(len(bank), 4, -1)
    y = labels(bank, 'train')[0].reshape(-1, 4).astype(bool)
    configs = [{'kind': 'baseline', 'dim': 0, 'penalty': 1.}]
    configs += [dict(kind=k, dim=d, penalty=p) for k in ('bilinear', 'hand_gate')
                for d in (8, 16) for p in (.2, 1.)]
    (OUT / 'protocol.json').write_text(json.dumps({'seed': 7506, 'folds': 3,
        'configs': configs, 'selection': 'pooled train OOF hard correct; earliest breaks ties'}, indent=2))
    oof = np.zeros((len(configs), len(y), 4))
    for fold, (tr, held) in enumerate(StratifiedKFold(3, shuffle=True, random_state=7506).split(x, y.sum(1))):
        proj = projection(x[tr], 16)
        for idx, cfg in enumerate(configs):
            xt = x[tr] if idx == 0 else expand(x[tr], proj, cfg['kind'], cfg['dim'])
            xv = x[held] if idx == 0 else expand(x[held], proj, cfg['kind'], cfg['dim'])
            model = fit_adam(xt, y[tr], cfg['penalty'])
            oof[idx, held] = score(xv, model)
        print(json.dumps({'fold': fold, 'seconds': time.perf_counter() - start,
                          'accuracies': [accuracy(s[held], y[held])['hard_pct'] for s in oof]}), flush=True)
    metrics = [dict(config=c, oof=accuracy(s, y)) for c, s in zip(configs, oof)]
    winner = max(range(len(configs)), key=lambda i: metrics[i]['oof']['hard_correct'])
    selection = {'winner_index': winner, 'winner': configs[winner], 'results': metrics}
    (OUT / 'cv_selection.json').write_text(json.dumps(selection, indent=2))
    np.savez(OUT / 'oof_diagnostics.npz', scores=oof, labels=y)
    print(json.dumps(selection, indent=2), flush=True)
    # Exactly one selected configuration is evaluated on validation.
    cfg = configs[winner]
    proj = projection(x, 16)
    xt = x if winner == 0 else expand(x, proj, cfg['kind'], cfg['dim'])
    model = fit_adam(xt, y, cfg['penalty'])
    np.savez(OUT / 'selected_bundle.npz', **model, **proj, kind=np.array(cfg['kind']), dim=cfg['dim'])
    vb, va = load_inputs('validation', DATA)
    vx = build_matrix(vb, va).reshape(len(vb), 4, -1)
    vy = labels(vb, 'validation')[0].reshape(-1, 4).astype(bool)
    xv = vx if winner == 0 else expand(vx, proj, cfg['kind'], cfg['dim'])
    new = score(xv, model)
    with np.load('runs/model_bundle.npz') as f:
        old = ((vx - f['mu']) / f['sd']) @ f['w']
    a = vy[np.arange(len(vy)), old.argmax(1)]
    b = vy[np.arange(len(vy)), new.argmax(1)]
    result = dict(config=cfg, baseline=accuracy(old, vy), selected=accuracy(new, vy),
                  rescued=int((~a & b).sum()), regressed=int((a & ~b).sum()),
                  seconds=time.perf_counter()-start)
    (OUT / 'validation_results.json').write_text(json.dumps(result, indent=2))
    np.savez(OUT / 'validation_diagnostics.npz', frozen_scores=old, selected_scores=new, labels=vy)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
