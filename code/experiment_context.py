"""Train-only, group-disjoint CV of contextual embeddings and quadratic reliability.

No test data is loaded. Validation is opened only after the CV choice is saved.
Run from code/: ../.venv/Scripts/python.exe experiment_context.py
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from common import DATA, load_inputs, check_data
from features import build_matrix
from train_final import labels

OUT = Path('runs/context_experiment')
SEED = 7506


def accuracy(s, y):
    chosen = y[np.arange(len(y)), s.argmax(1)]
    hard = (y.sum(1) > 0) & (y.sum(1) < 4)
    return {'all_correct': int(chosen.sum()), 'all_n': len(y),
            'hard_correct': int(chosen[hard].sum()), 'hard_n': int(hard.sum()),
            'all_pct': float(100 * chosen.mean()),
            'hard_pct': float(100 * chosen[hard].mean())}


def fit_rank(x, y, lam=.02):
    mu = x.mean((0, 1)); sd = x.std((0, 1)) + 1e-9
    z = ((x - mu) / sd).reshape(-1, x.shape[-1])
    # A constant feature has no effect on within-group probabilities.
    n = len(y)
    def objective(w):
        s = (z @ w).reshape(n, 4)
        all_lse = logsumexp(s, axis=1, keepdims=True)
        correct_lse = logsumexp(np.where(y, s, -np.inf), axis=1, keepdims=True)
        p = np.exp(s - all_lse)
        q = np.where(y, np.exp(s - correct_lse), 0)
        loss = np.mean(all_lse - correct_lse) + .5 * lam * (w @ w)
        grad = z.T @ (p - q).ravel() / n + lam * w
        return loss, grad
    fit = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                   options={'maxiter': 250, 'ftol': 1e-11, 'gtol': 1e-7})
    return {'w': fit.x, 'mu': mu, 'sd': sd}


def rank_scores(x, model):
    return ((x - model['mu']) / model['sd']) @ model['w']


def contextual(x, mode, scale=None):
    hand, e = x[..., :46], x[..., 46:]
    r = e - e.mean(1, keepdims=True)
    if scale is None:
        scale = np.sqrt(np.mean(r * r, axis=(0, 1))) + 1e-8
    if mode == 'relative':
        # Per-coordinate shrinkage prevents tiny within-group differences exploding.
        r = r / np.sqrt(np.mean(r * r, axis=1, keepdims=True) + scale ** 2)
        return np.concatenate([hand, r], axis=-1), scale
    if mode == 'direction':
        # Candidate residual direction, independent of group semantic dispersion.
        r = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-8)
        return np.concatenate([hand, r], axis=-1), scale
    if mode == 'augmented':
        r = r / np.sqrt(np.mean(r * r, axis=1, keepdims=True) + scale ** 2)
        return np.concatenate([x, r], axis=-1), scale
    raise ValueError(mode)


def qda_fit(x, y, dim, shrink):
    e = x[..., 46:]
    r = (e - e.mean(1, keepdims=True)).reshape(-1, e.shape[-1])
    # Unsupervised PCA of residuals, fitted on the training fold only.
    eig, vec = np.linalg.eigh(r.T @ r / len(r))
    p = vec[:, -dim:] / np.sqrt(np.maximum(eig[-dim:], 1e-10))
    z = (r @ p).reshape(len(y), 4, dim)
    hard = (y.sum(1) > 0) & (y.sum(1) < 4)
    z, target = z[hard].reshape(-1, dim), y[hard].ravel()
    means, precisions, constants = [], [], []
    for value in (False, True):
        a = z[target == value]
        m = a.mean(0)
        c = (a - m).T @ (a - m) / len(a)
        c = (1 - shrink) * c + shrink * np.eye(dim)
        sign, ld = np.linalg.slogdet(c)
        assert sign > 0
        means.append(m); precisions.append(np.linalg.inv(c)); constants.append(-.5 * ld)
    return {'p': p, 'means': np.array(means), 'precisions': np.array(precisions),
            'constants': np.array(constants)}


def qda_scores(x, model):
    e = x[..., 46:]; r = e - e.mean(1, keepdims=True)
    z = r @ model['p']; scores = []
    for m, q, c in zip(model['means'], model['precisions'], model['constants']):
        d = z - m
        scores.append(-.5 * np.sum((d @ q) * d, axis=-1) + c)
    return scores[1] - scores[0]


def normalized(s):
    s = s - s.mean(1, keepdims=True)
    return s / max(float(np.sqrt(np.mean(s * s))), 1e-8)


def main():
    tick = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    check_data(DATA, ('train', 'validation'))
    bank, arrays = load_inputs('train', DATA)
    y = labels(bank, 'train')[0].reshape(-1, 4).astype(bool)
    x = build_matrix(bank, arrays).reshape(len(bank), 4, -1)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, y.sum(1)))
    # Predeclared finite search. The validation split does not affect these choices.
    configs = [{'kind': 'baseline'}]
    configs += [{'kind': m} for m in ('relative', 'direction', 'augmented')]
    configs += [{'kind': 'qda', 'dim': d, 'shrink': sh, 'alpha': a}
                for d in (64, 128, 256) for sh in (.5, .9) for a in (.15, .35)]
    (OUT / 'protocol.json').write_text(json.dumps({'seed': SEED, 'folds': 3,
        'selection': 'pooled training OOF hard accuracy; tie: earliest configuration',
        'configs': configs, 'test_access': False}, indent=2))
    oof = np.zeros((len(configs), len(y), 4))
    for fold, (tr, held) in enumerate(folds):
        base = fit_rank(x[tr], y[tr])
        bs = rank_scores(x[held], base)
        train_bs = rank_scores(x[tr], base)
        bscale = np.sqrt(np.mean((train_bs - train_bs.mean(1, keepdims=True)) ** 2))
        oof[0, held] = bs
        for idx, cfg in enumerate(configs[1:4], 1):
            xt, scale = contextual(x[tr], cfg['kind'])
            xv, _ = contextual(x[held], cfg['kind'], scale)
            oof[idx, held] = rank_scores(xv, fit_rank(xt, y[tr]))
        for d in (64, 128, 256):
            for sh in (.5, .9):
                model = qda_fit(x[tr], y[tr], d, sh)
                train_qs = qda_scores(x[tr], model)
                qscale = np.sqrt(np.mean((train_qs - train_qs.mean(1, keepdims=True)) ** 2))
                qs = qda_scores(x[held], model) / qscale
                for idx, cfg in enumerate(configs):
                    if cfg['kind'] == 'qda' and cfg['dim'] == d and cfg['shrink'] == sh:
                        oof[idx, held] = bs / bscale + cfg['alpha'] * qs
        print(json.dumps({'fold': fold, 'elapsed': time.perf_counter() - tick,
                          'baseline': accuracy(bs, y[held])}), flush=True)
    cv = [{'config': cfg, 'oof': accuracy(s, y)} for cfg, s in zip(configs, oof)]
    winner = max(range(len(cv)), key=lambda i: cv[i]['oof']['hard_correct'])
    # Winner and all CV results are persisted BEFORE opening validation.
    (OUT / 'cv_selection.json').write_text(json.dumps({'winner_index': winner,
        'winner': configs[winner], 'results': cv}, indent=2))
    np.savez(OUT / 'oof_diagnostics.npz', scores=oof, labels=y)
    print(json.dumps({'selected': configs[winner], 'cv': cv}, indent=2), flush=True)
    va_bank, va_arrays = load_inputs('validation', DATA)
    vy = labels(va_bank, 'validation')[0].reshape(-1, 4).astype(bool)
    vx = build_matrix(va_bank, va_arrays).reshape(len(va_bank), 4, -1)
    baseline = fit_rank(x, y)
    bs = rank_scores(vx, baseline)
    with np.load('runs/model_bundle.npz') as frozen:
        fs = (((vx.reshape(-1, vx.shape[-1]) - frozen['mu']) / frozen['sd']) @ frozen['w']).reshape(-1, 4)
    results = {'frozen_baseline': accuracy(fs, vy), 'refit_baseline': accuracy(bs, vy),
               'selected_config': configs[winner]}
    # Report the three different hypotheses as declared ablations, not an extra grid.
    # QDA validation is evaluated only if selected by training CV.
    selected_scores = bs
    for mode in ('relative', 'direction', 'augmented'):
        xt, scale = contextual(x, mode); xv, _ = contextual(vx, mode, scale)
        model = fit_rank(xt, y)
        scores = rank_scores(xv, model)
        results[mode] = accuracy(scores, vy)
        np.savez(OUT / f'{mode}_bundle.npz', **model, scale=scale, mode=np.array(mode))
        if configs[winner]['kind'] == mode:
            selected_scores = scores
    if configs[winner]['kind'] == 'qda':
        cfg = configs[winner]
        qm = qda_fit(x, y, cfg['dim'], cfg['shrink'])
        # Normalize with training-input scores, not validation statistics.
        train_bs = rank_scores(x, baseline)
        train_qs = qda_scores(x, qm)
        bscale = np.sqrt(np.mean((train_bs - train_bs.mean(1, keepdims=True)) ** 2))
        qscale = np.sqrt(np.mean((train_qs - train_qs.mean(1, keepdims=True)) ** 2))
        selected_scores = bs / bscale + cfg['alpha'] * qda_scores(vx, qm) / qscale
        np.savez(OUT / 'qda_bundle.npz', **qm, **{f'base_{k}': v for k, v in baseline.items()},
                 bscale=bscale, qscale=qscale, alpha=cfg['alpha'])
    results['selected'] = accuracy(selected_scores, vy)
    good_old = vy[np.arange(len(vy)), fs.argmax(1)]
    good_new = vy[np.arange(len(vy)), selected_scores.argmax(1)]
    results['paired'] = {'rescued': int((~good_old & good_new).sum()),
                         'regressed': int((good_old & ~good_new).sum())}
    results['seconds'] = time.perf_counter() - tick
    results['versions'] = {'numpy': np.__version__}
    (OUT / 'validation_results.json').write_text(json.dumps(results, indent=2))
    np.savez(OUT / 'validation_diagnostics.npz', frozen_scores=fs, selected_scores=selected_scores, labels=vy)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
