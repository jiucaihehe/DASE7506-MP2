"""Submission: group-softmax linear ranker over hand-crafted + embedding features.

Inference is numpy-only: build per-candidate features (see features.py),
standardize with the saved train statistics, score with a linear model, and
pick the argmax candidate per question group. No reference answers are read.

Checkpoint: runs/model_bundle.npz with float32 arrays {w, b, mu, sd}.
"""
import numpy as np

from features import build_matrix


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order."""
    bundle = np.load(checkpoint)
    w, b, mu, sd = bundle['w'], float(bundle['b']), bundle['mu'], bundle['sd']
    X = build_matrix(bank, arrays, include_embedding=True)
    scores = ((X - mu) / sd) @ w + b
    choices = scores.reshape(len(bank), 4).argmax(axis=1)
    return [int(c) for c in choices]
