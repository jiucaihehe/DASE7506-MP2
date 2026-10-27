"""Train-only arithmetic program extraction and group-disjoint transfer selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import json
import shutil
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from common import DATA, load_inputs, read_jsonl
from features import build_matrix
from train_final import labels
from experiment_context import accuracy
from program_transfer import extract_program, fit_retriever, suggested_answers, transfer_scores, save_retriever

OUT = Path('runs/program_experiment')


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    bank, arrays = load_inputs('train', DATA)
    refs = read_jsonl(DATA / 'gsm8k_train.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    y = labels(bank, 'train')[0].reshape(-1, 4).astype(bool)
    programs = [extract_program(r['question'], r['answer']) for r in refs]
    print(json.dumps({'programs_extracted': sum(p is not None for p in programs), 'train_groups': len(bank)}), flush=True)
    base_path = Path('runs/bilinear_experiment/oof_diagnostics.npz')
    if not base_path.exists():
        raise RuntimeError('First finish experiment_bilinear.py to generate matching original-Adam baseline OOF scores')
    with np.load(base_path) as f:
        assert np.array_equal(f['labels'], y)
        base_scores = f['scores'][0]
    thresholds = [.35, .45, .55, .65, .75, .85, 1.01]
    (OUT / 'protocol.json').write_text(json.dumps({'seed': 7506, 'folds': 3,
        'thresholds': thresholds, 'selection': 'max OOF hard correct; larger threshold breaks ties',
        'supervision': 'training questions and reference arithmetic programs only'}, indent=2))
    answers = [None] * len(bank); sims = np.zeros(len(bank))
    for fold, (tr, held) in enumerate(StratifiedKFold(3, shuffle=True, random_state=7506).split(y, y.sum(1))):
        rows = [bank[i] for i in tr]
        vectorizer, matrix, learned = fit_retriever(rows, [programs[i] for i in tr])
        predicted, similarities = suggested_answers([bank[i] for i in held], vectorizer, matrix, learned)
        for i, value, sim in zip(held, predicted, similarities):
            answers[i] = value; sims[i] = sim
        print(json.dumps({'fold': fold, 'programs': len(learned), 'seconds': time.perf_counter()-start}), flush=True)
    results = [dict(threshold=t, oof=accuracy(transfer_scores(bank, base_scores, answers, sims, t), y)) for t in thresholds]
    winner = max(results, key=lambda r: (r['oof']['hard_correct'], r['threshold']))
    selection = {'selected': winner, 'baseline': accuracy(base_scores, y), 'results': results}
    (OUT / 'cv_selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2), flush=True)
    np.savez(OUT / 'oof_diagnostics.npz', similarities=sims, suggestions=np.array([a or '' for a in answers]), labels=y)
    vectorizer, matrix, learned = fit_retriever(bank, programs)
    save_retriever(OUT / 'programs.npz', vectorizer, matrix, learned, winner['threshold'])
    shutil.copyfile('runs/model_bundle.npz', OUT / 'base_bundle.npz')
    vb, va = load_inputs('validation', DATA)
    vy = labels(vb, 'validation')[0].reshape(-1, 4).astype(bool)
    x = build_matrix(vb, va).reshape(len(vb), 4, -1)
    with np.load('runs/model_bundle.npz') as f:
        old = ((x - f['mu']) / f['sd']) @ f['w']
    proposed, confidence = suggested_answers(vb, vectorizer, matrix, learned)
    new = transfer_scores(vb, old, proposed, confidence, winner['threshold'])
    a = vy[np.arange(len(vy)), old.argmax(1)]
    b = vy[np.arange(len(vy)), new.argmax(1)]
    result = dict(selected_threshold=winner['threshold'], baseline=accuracy(old, vy), selected=accuracy(new, vy),
                  rescued=int((~a & b).sum()), regressed=int((a & ~b).sum()),
                  changed=int((old.argmax(1) != new.argmax(1)).sum()), seconds=time.perf_counter()-start,
                  asset_bytes=sum(p.stat().st_size for p in OUT.glob('*.npz') if p.name in ('programs.npz', 'base_bundle.npz')))
    (OUT / 'validation_results.json').write_text(json.dumps(result, indent=2))
    np.savez(OUT / 'validation_diagnostics.npz', frozen_scores=old, selected_scores=new, labels=vy,
             similarities=confidence, suggestions=np.array([a or '' for a in proposed]))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
