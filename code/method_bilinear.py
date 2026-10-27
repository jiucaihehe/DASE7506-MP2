"""Question-conditioned linear preferences; inference requires NumPy only."""
import numpy as np
from features import build_matrix


def scores(bank, arrays, checkpoint):
    if not bank:
        return np.empty((0, 4))
    x = build_matrix(bank, arrays).reshape(len(bank), 4, -1)
    with np.load(checkpoint, allow_pickle=False) as model:
        kind = str(model['kind']); dim = int(model['dim'])
        if kind != 'baseline':
            e = x[..., 46:]; c = e.mean(1)
            context = (c - model['cm']) @ model['cp'][:, -dim:]
            if kind == 'bilinear':
                residual = (e - c[:, None, :]) @ model['rp'][:, -dim:]
            elif kind == 'hand_gate':
                residual = x[..., :46]
            else:
                raise ValueError('Unknown contextual interaction')
            interaction = (context[:, None, :, None] * residual[:, :, None, :]).reshape(len(x), 4, -1)
            x = np.concatenate([x, interaction], axis=-1)
        return ((x - model['mu']) / model['sd']) @ model['w']


def predict(bank, arrays, checkpoint):
    return [int(c) for c in scores(bank, arrays, checkpoint).argmax(1)]
