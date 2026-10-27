"""Experimental contextual selector. NumPy only; no references or fitting.

Pass a bundle produced by experiment_context.py. This is separate from the
frozen submission.py and runs/model_bundle.npz.
"""
import numpy as np
from features import build_matrix


def scores(bank, arrays, checkpoint):
    if not bank:
        return np.empty((0, 4))
    x = build_matrix(bank, arrays).reshape(len(bank), 4, -1)
    with np.load(checkpoint, allow_pickle=False) as model:
        if 'mode' in model:
            mode = str(model['mode'])
            e = x[..., 46:]
            r = e - e.mean(1, keepdims=True)
            if mode == 'direction':
                r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-8)
            elif mode in ('relative', 'augmented'):
                r /= np.sqrt(np.mean(r * r, axis=1, keepdims=True) + model['scale'] ** 2)
            else:
                raise ValueError('Unknown representation mode')
            z = np.concatenate([x if mode == 'augmented' else x[..., :46], r], axis=-1)
            return ((z - model['mu']) / model['sd']) @ model['w']
        e = x[..., 46:]
        z = (e - e.mean(1, keepdims=True)) @ model['p']
        likelihoods = []
        for m, q, c in zip(model['means'], model['precisions'], model['constants']):
            d = z - m
            likelihoods.append(-.5 * np.sum((d @ q) * d, axis=-1) + c)
        bs = ((x - model['base_mu']) / model['base_sd']) @ model['base_w']
        return bs / model['bscale'] + model['alpha'] * (likelihoods[1] - likelihoods[0]) / model['qscale']


def predict(bank, arrays, checkpoint):
    return [int(choice) for choice in scores(bank, arrays, checkpoint).argmax(axis=1)]
