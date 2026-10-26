"""Uncertainty calibration + failure analysis on validation (for the report)."""
import json
import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import build_matrix

va_bank, va_arrays = load_inputs('validation', DATA)
gold = {r['id']: parse_answer(r['answer']) for r in read_jsonl(DATA / 'gsm8k_validation.jsonl')}
y = np.zeros(len(va_bank) * 4)
for i, row in enumerate(va_bank):
    for j, c in enumerate(row['candidates']):
        a = parse_answer(c['text'])
        y[4 * i + j] = float(a is not None and a == gold[row['id']])

bundle = np.load('runs/model_bundle.npz')
w, b, mu, sd = bundle['w'], float(bundle['b']), bundle['mu'], bundle['sd']
X = build_matrix(va_bank, va_arrays, True)
s = (((X - mu) / sd) @ w + b).reshape(-1, 4)
s = s - s.max(axis=1, keepdims=True)
p = np.exp(s); p /= p.sum(axis=1, keepdims=True)
conf = p.max(axis=1)
choice = p.argmax(axis=1)
hit = np.array([y[4 * i + choice[i]] for i in range(len(va_bank))])

# calibration: bucket by confidence
bins = [(0.25, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)]
print('confidence bucket -> n, accuracy')
for lo, hi in bins:
    m = (conf >= lo) & (conf < hi)
    if m.sum():
        print(f'  [{lo:.2f},{hi:.2f}): n={int(m.sum()):4d}  acc={100 * hit[m].mean():.1f}%')
print('spearman conf vs correct:', end=' ')
from scipy.stats import spearmanr
print(round(float(spearmanr(conf, hit).statistic), 3))

# margin analysis: p1 - p2
ps = np.sort(p, axis=1)
margin = ps[:, -1] - ps[:, -2]
for lo, hi in [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 1.0)]:
    m = (margin >= lo) & (margin < hi)
    if m.sum():
        print(f'  margin [{lo:.2f},{hi:.2f}): n={int(m.sum()):4d}  acc={100 * hit[m].mean():.1f}%')

# failure examples on hard groups
hard = np.array([0 < y[4 * i:4 * i + 4].sum() < 4 for i in range(len(va_bank))])
fails = np.where(hard & (hit == 0))[0]
print(f'\nhard-group failures: {len(fails)} / {int(hard.sum())}')
for i in fails[:3]:
    row = va_bank[i]
    print('=' * 80)
    print('Q:', row['question'][:200])
    print('gold:', gold[row['id']])
    for j, c in enumerate(row['candidates']):
        a = parse_answer(c['text'])
        mark = 'PICK' if j == choice[i] else ('OK' if y[4 * i + j] else 'wrong')
        print(f'  cand{j} [{mark}] p={p[i, j]:.2f} ans={a} len={len(c["text"])}')

print('\n=== HARD GROUPS ONLY ===')
for lo, hi in bins:
    m = (conf >= lo) & (conf < hi) & hard
    if m.sum():
        print(f'  [{lo:.2f},{hi:.2f}): n={int(m.sum()):4d}  acc={100 * hit[m].mean():.1f}%')
print('spearman (hard):', round(float(spearmanr(conf[hard], hit[hard]).statistic), 3))
ps_h = ps[hard]; margin_h = ps_h[:, -1] - ps_h[:, -2]; hit_h = hit[hard]
for lo, hi in [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 1.0)]:
    m = (margin_h >= lo) & (margin_h < hi)
    if m.sum():
        print(f'  margin [{lo:.2f},{hi:.2f}): n={int(m.sum()):4d}  acc={100 * hit_h[m].mean():.1f}%')
# how often does the model pick a minority answer over an agreeing majority, and fail?
from collections import Counter
minority_picks = fail_when_minority = 0
for i in np.where(hard)[0]:
    answers = [parse_answer(c['text']) for c in va_bank[i]['candidates']]
    counts = Counter(a for a in answers if a is not None)
    if len(counts) >= 2:
        chosen = answers[i] if False else parse_answer(va_bank[i]['candidates'][choice[i]]['text'])
        if chosen is not None and counts[chosen] < max(counts.values()):
            minority_picks += 1
            fail_when_minority += int(hit[i] == 0)
print(f'minority-answer picks on hard groups: {minority_picks}, of which wrong: {fail_when_minority}')
