"""Round three: answer-aware message passing and diagonal residual energies.

Training-only grouped CV selects one setting per family before validation loads.
No test inputs or references are accessed. Run from code/.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from common import DATA, load_inputs, parse_answer
from features import build_matrix
from train_final import labels
from experiment_context import accuracy
from experiment_bilinear import projection, expand, score

OUT = Path('runs/relation_experiment')


def extra_features(bank, x, kind):
    e = x[..., 46:]
    r = e - e.mean(1, keepdims=True)
    if kind == 'energy':
        return r * r
    if kind != 'messages':
        raise ValueError(kind)
    same = np.zeros((len(bank), 4, 4))
    for n, row in enumerate(bank):
        a = [parse_answer(c['text']) for c in row['candidates']]
        for j in range(4):
            for k in range(4):
                same[n, j, k] = float(j != k and a[j] is not None and a[j] == a[k])
    same /= np.maximum(same.sum(2, keepdims=True), 1)
    message = np.einsum('njk,nkd->njd', same, r)
    return np.concatenate([message, message * r], axis=-1)


def fit(x, y, penalty):
    mu = x.mean((0, 1)); sd = x.std((0, 1)) + 1e-9
    z = ((x - mu) / sd).reshape(-1, x.shape[-1])
    reg = np.full(x.shape[-1], penalty)
    reg[:1070] = .02
    reg[1070:1438] = 1.
    def objective(w):
        s = (z @ w).reshape(-1, 4)
        al = logsumexp(s, axis=1, keepdims=True)
        good = logsumexp(np.where(y, s, -np.inf), axis=1, keepdims=True)
        ds = np.exp(s-al) - np.where(y, np.exp(s-good), 0)
        return ((al-good).mean() + .5 * np.dot(reg*w, w),
                z.T @ ds.ravel() / len(y) + reg*w)
    opt = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                   options={'maxiter': 350, 'ftol': 1e-11, 'gtol': 1e-7})
    return dict(w=opt.x, mu=mu, sd=sd), dict(success=bool(opt.success),
        iterations=int(opt.nit), message=str(opt.message), gradient_max=float(abs(opt.jac).max()))


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    configs = [dict(kind='control', penalty=1.)] + [dict(kind=k, penalty=p)
        for k in ('messages', 'energy') for p in (.1, 1., 10.)]
    protocol = dict(seed=7506, folds=3, configs=configs,
        selection='One winner per family on train OOF hard correct; earliest tie.',
        promotion='Validation hard must exceed frozen 368/419; test cannot select models.',
        baseline='Same contextual features, L-BFGS instead of original Adam; matched control.',
        split_access=['train', 'validation'])
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    b, a = load_inputs('train', DATA)
    x = build_matrix(b, a).reshape(len(b), 4, -1)
    y = labels(b, 'train')[0].reshape(-1, 4).astype(bool)
    extras = {k: extra_features(b, x, k) for k in ('messages', 'energy')}
    oof = np.zeros((len(configs), len(b), 4)); convergence = []
    for fold, (tr, held) in enumerate(StratifiedKFold(3, shuffle=True, random_state=7506).split(x, y.sum(1))):
        proj = projection(x[tr], 8)
        base = expand(x, proj, 'hand_gate', 8)
        for i, c in enumerate(configs):
            z = base if c['kind']=='control' else np.concatenate([base, extras[c['kind']]], axis=-1)
            m, diagnostic = fit(z[tr], y[tr], c['penalty'])
            oof[i, held] = score(z[held], m)
            convergence.append(dict(fold=fold, config=c, **diagnostic))
            print(json.dumps(dict(fold=fold, config=c, held=accuracy(oof[i, held], y[held]),
                                  seconds=time.perf_counter()-start, **diagnostic)), flush=True)
    results = [dict(config=c, oof=accuracy(s, y)) for c, s in zip(configs, oof)]
    winners = [0] + [max([i for i,c in enumerate(configs) if c['kind']==k],
                        key=lambda i:results[i]['oof']['hard_correct']) for k in ('messages','energy')]
    selection = dict(winner_indices=winners, results=results, convergence=convergence)
    (OUT/'cv_selection.json').write_text(json.dumps(selection, indent=2))
    np.savez(OUT/'oof.npz', scores=oof, labels=y)
    print(json.dumps(selection), flush=True)
    # Fit all chosen models before opening validation.
    proj = projection(x, 8); base = expand(x, proj, 'hand_gate', 8)
    models = {}
    for i in winners:
        c = configs[i]; k = c['kind']
        z = base if k=='control' else np.concatenate([base, extras[k]], axis=-1)
        models[k], diagnostic = fit(z, y, c['penalty'])
        np.savez(OUT/f'{k}_bundle.npz', **models[k], **proj, kind=np.array(k))
    vb, va = load_inputs('validation', DATA)
    vx = build_matrix(vb, va).reshape(len(vb), 4, -1)
    vy = labels(vb, 'validation')[0].reshape(-1, 4).astype(bool)
    vbase = expand(vx, proj, 'hand_gate', 8)
    from method_bilinear import scores as frozen_scores
    frozen = frozen_scores(vb, va, Path('runs/final_candidate/selected_bundle.npz'))
    oldgood = vy[np.arange(len(vy)), frozen.argmax(1)]
    report = dict(frozen=accuracy(frozen, vy), results=[], seconds=0,
                  test_evaluated=False)
    for i in winners:
        c=configs[i]; k=c['kind']
        z = vbase if k=='control' else np.concatenate([vbase, extra_features(vb, vx, k)], axis=-1)
        s = score(z, models[k]); newgood = vy[np.arange(len(vy)), s.argmax(1)]
        report['results'].append(dict(config=c, validation=accuracy(s, vy),
            rescued=int((newgood & ~oldgood).sum()), regressed=int((~newgood & oldgood).sum())))
        np.savez(OUT/f'{k}_validation.npz', scores=s, labels=vy)
    report['seconds']=time.perf_counter()-start
    (OUT/'validation_results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
