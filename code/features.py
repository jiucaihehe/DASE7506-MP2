"""Feature extraction for candidate-correctness modeling.

Per-candidate features, all computable at inference without references:
  - text statistics (length, equations, boxed/#### markers, parse success)
  - parsed-answer agreement within the group
  - embedding geometry relative to the group (cosine to centroid / peers,
    norm, distance from group mean)
  - candidate position
Raw 1024-d embeddings are optionally appended for the full model.
"""
import re

import numpy as np

from common import parse_answer

_EQ_RE = re.compile(r'[=+]|\$\$|\\frac|\\times|\\cdot')
_DIGIT_RE = re.compile(r'\d')


def text_stats(text):
    n = len(text)
    return [
        np.log1p(n),
        np.log1p(len(_DIGIT_RE.findall(text))),
        np.log1p(len(_EQ_RE.findall(text))),
        float('\\boxed' in text),
        float('####' in text),
        float(bool(re.search(r'(?i)final answer|answer:', text))),
        np.log1p(text.count('\n')),
    ]


_WORD_RE = re.compile(r'[a-z]+')
_NUM_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?')
_HEDGE_RE = re.compile(r'(?i)\b(wait|hmm|actually|maybe|not sure|confus|oops|mistake|reconsider)\b')
_EQVAL_RE = re.compile(r'=\s*\$?\\?\s*(?:\\text\{[^}]*\}\s*)?\$?\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)')
_BOXED_RE = re.compile(r'\\boxed\{([^{}]+)\}')
_HASH_RE = re.compile(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)')


def _frac(s):
    from fractions import Fraction
    try:
        return str(Fraction(s.replace(',', '').replace('$', '').strip()))
    except (ValueError, ZeroDivisionError):
        return None


def consistency_feats(text, parsed_answer):
    """Internal-consistency signals: does the final answer agree with the
    last computed value in the trace, and do boxed/#### markers conflict?"""
    eq_vals = [_frac(m) for m in _EQVAL_RE.findall(text)]
    eq_vals = [v for v in eq_vals if v is not None]
    n_eq = len(eq_vals)
    consist = 0.0
    if parsed_answer is not None and n_eq:
        consist = 1.0 if eq_vals[-1] == parsed_answer else -1.0
    boxed = [_frac(m) for m in _BOXED_RE.findall(text)]
    boxed = [v for v in boxed if v is not None]
    hashes = [_frac(m) for m in _HASH_RE.findall(text)]
    hashes = [v for v in hashes if v is not None]
    conflict = 0.0
    if boxed and hashes and boxed[-1] != hashes[-1]:
        conflict = 1.0
    return [np.log1p(n_eq), consist, conflict]


def question_context(question):
    nums = _NUM_RE.findall(question)
    vals = set()
    for s in nums:
        try:
            vals.add(float(s.replace(',', '')))
        except ValueError:
            pass
    return {
        'qlen': np.log1p(len(question)),
        'qnums': np.log1p(len(nums)),
        'qvals': vals,
        'qwords': set(_WORD_RE.findall(question.lower())),
    }


def group_features(texts, embeddings, question=''):
    """Return (4, F) hand-crafted feature matrix for one question group."""
    qc = question_context(question)
    emb = embeddings.astype(np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = emb / norms
    centroid = unit.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)

    answers = [parse_answer(t) for t in texts]
    feats = []
    for j in range(4):
        sims = [float(unit[j] @ unit[k]) for k in range(4) if k != j]
        agree = sum(1 for k in range(4) if k != j and answers[k] is not None
                    and answers[k] == answers[j])
        parsed = answers[j] is not None
        val_feats = [0.0, 0.0, 0.0, 0.0]
        if parsed:
            try:
                v = float(answers[j].split('/')[0]) / (float(answers[j].split('/')[1]) if '/' in answers[j] else 1.0)
                val_feats = [np.log1p(abs(v)), float(float(v).is_integer()), float(abs(v) < 1e-9),
                             float(v in qc['qvals'])]
            except (ValueError, ZeroDivisionError):
                pass
        cwords = set(_WORD_RE.findall(texts[j].lower()))
        overlap = len(cwords & qc['qwords']) / max(len(qc['qwords']), 1)
        f = text_stats(texts[j]) + [
            float(parsed),
            float(agree),
            float(answers[j] is not None and answers.count(answers[j]) == 4),
            float(np.linalg.norm(emb[j])),
            float(unit[j] @ centroid),
            float(np.mean(sims)),
            float(np.max(sims)),
            float(j) / 3.0,
            float(len(_HEDGE_RE.findall(texts[j]))),
            overlap,
            qc['qlen'],
            qc['qnums'],
        ] + val_feats
        feats.append(f)
    F = np.asarray(feats, dtype=np.float64)
    # group-normalized copies of the per-candidate varying features
    # (exclude pure group-level columns: qlen/qnums are constant within group;
    #  group-normalizing a constant column yields 0, keep them anyway = 0)
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True)
    Z = (F - mu) / (sd + 1e-9)
    return np.hstack([F, Z])


HAND_DIM = len(group_features(['0'] * 4, np.zeros((4, 1024), dtype=np.float32))[0])


def build_matrix(bank, arrays, include_embedding=True):
    """Stack features for all candidates: returns (N*4, D)."""
    n = len(bank)
    rows = []
    for i in range(n):
        texts = [c['text'] for c in bank[i]['candidates']]
        f = group_features(texts, arrays['embeddings'][i], bank[i]['question'])
        if include_embedding:
            f = np.hstack([f, arrays['embeddings'][i].astype(np.float64)])
        rows.append(f)
    return np.vstack(rows)
