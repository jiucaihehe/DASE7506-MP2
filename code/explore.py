"""Exploration: per-candidate correctness stats and simple baselines."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl


def split_stats(split):
    bank, arrays = load_inputs(split, DATA)
    gold = {row['id']: parse_answer(row['answer']) for row in read_jsonl(DATA / f'gsm8k_{split}.jsonl')}
    n = len(bank)
    correct = np.zeros((n, 4), dtype=bool)
    answers = []
    n_unparsed = 0
    for i, row in enumerate(bank):
        cand_ans = [parse_answer(c['text']) for c in row['candidates']]
        answers.append(cand_ans)
        n_unparsed += sum(a is None for a in cand_ans)
        for j in range(4):
            correct[i, j] = cand_ans[j] is not None and cand_ans[j] == gold[row['id']]
    per_pos = correct.mean(axis=0)
    n_correct_per_q = correct.sum(axis=1)
    print(f'=== {split}: {n} questions ===')
    print('P(candidate correct | position):', np.round(per_pos, 4))
    print('P(any candidate correct):', round(float(correct.any(axis=1).mean()), 4))
    print('dist of #correct per question:', Counter(n_correct_per_q.tolist()))
    print('unparseable candidate answers:', n_unparsed, '/', n * 4)
    # candidate-level accuracy
    print('overall candidate correctness:', round(float(correct.mean()), 4))

    # Baseline 1: always pick candidate 0
    print('baseline first-candidate:', round(float(correct[:, 0].mean() * 100), 2))

    # Baseline 2: majority vote on parsed answers (ties -> first occurrence)
    maj_choices, rng_choices = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        counts = Counter(a for a in answers[i] if a is not None)
        if counts:
            top = counts.most_common()
            best_ans = top[0][0]
            # pick first candidate whose answer equals best
            choice = next(j for j, a in enumerate(answers[i]) if a == best_ans)
        else:
            choice = 0
        maj_choices.append(choice)
        valid = [j for j in range(4) if answers[i][j] is not None]
        rng_choices.append(int(rng.choice(valid)))
    def acc(choices):
        return round(float(np.mean([correct[i, c] for i, c in enumerate(choices)])) * 100, 2)
    print('baseline majority-vote:', acc(maj_choices))
    print('baseline random-among-parsed:', acc(rng_choices))

    # oracle: any correct
    print('oracle (pick a correct one if exists):', round(float(correct.any(axis=1).mean() * 100), 2))
    return bank, arrays, correct, answers


if __name__ == '__main__':
    for split in ('train', 'validation', 'test'):
        split_stats(split)
