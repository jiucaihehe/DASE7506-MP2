"""Dependency-free TF-IDF vectorizer (word unigrams + bigrams).

Same code path is used for training and inference so vectorization is
guaranteed consistent. Vocabulary and IDF are learned from the train split
candidate texts only and stored in the checkpoint as learned statistics.
"""
import json
import re
from collections import Counter

import numpy as np

_TOK_RE = re.compile(r'[a-z0-9]+')


def tokenize(text):
    toks = _TOK_RE.findall(text.lower())
    return toks + [f'{a}__{b}' for a, b in zip(toks, toks[1:])]


def build_vocab(texts, max_features=4000, min_df=5):
    df = Counter()
    for t in texts:
        df.update(set(tokenize(t)))
    common = [(tok, c) for tok, c in df.items() if c >= min_df]
    common.sort(key=lambda kv: (-kv[1], kv[0]))
    vocab = {tok: i for i, (tok, _) in enumerate(common[:max_features])}
    return vocab


def fit_idf(texts, vocab):
    df = np.zeros(len(vocab))
    for t in texts:
        for tok in set(tokenize(t)):
            i = vocab.get(tok)
            if i is not None:
                df[i] += 1
    n = len(texts)
    return np.log((n + 1) / (df + 1)) + 1.0


def transform(texts, vocab, idf):
    X = np.zeros((len(texts), len(vocab)))
    for r, t in enumerate(texts):
        counts = Counter(tokenize(t))
        if not counts:
            continue
        inv = 1.0 / sum(counts.values())
        for tok, c in counts.items():
            i = vocab.get(tok)
            if i is not None:
                X[r, i] = c * inv * idf[i]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def save_vocab(path, vocab):
    Path = __import__('pathlib').Path
    Path(path).write_text(json.dumps(vocab))


def load_vocab(path):
    Path = __import__('pathlib').Path
    return {k: int(v) for k, v in json.loads(Path(path).read_text()).items()}
