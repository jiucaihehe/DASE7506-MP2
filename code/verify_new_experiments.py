"""Fresh-process VALIDATION benchmarks and input-contract checks. Never runs test."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '4'
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from common import DATA, load_inputs

OUT = Path('runs/new_experiment_verification')
EXPECTED = {'submission.py': '3d85573c28baae14b02a6a9f7fd396f04e40791f98380ecedf43672f5ae8e013',
            'runs/model_bundle.npz': 'd1cdc3a3c209980acc43a70c9d428be105f2c2782a608dbbf6833b5c227f08da'}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    integrity = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in EXPECTED}
    assert integrity['runs/model_bundle.npz'] == EXPECTED['runs/model_bundle.npz'], 'Frozen checkpoint changed'
    canonical_source = hashlib.sha256(Path('submission.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    assert canonical_source == 'bce8c4daff2113ee63c0ddc32cc9578523955b73f3d4062943fca4bf9624a68e', 'Frozen source changed'
    configs = [
        ('context', 'method_context.py', 'runs/context_experiment/qda_bundle.npz'),
        ('bilinear', 'method_bilinear.py', 'runs/bilinear_experiment/selected_bundle.npz'),
        ('programs', 'program_transfer.py', 'runs/program_experiment'),
    ]
    results = {}
    for name, method, checkpoint in configs:
        pred = OUT / f'{name}_predictions.json'; score = OUT / f'{name}_score.json'
        tick = time.perf_counter()
        subprocess.run([sys.executable, 'predict.py', '--method', method, '--checkpoint', checkpoint,
                        '--split', 'validation', '--output', str(pred)], check=True)
        prediction_wall = time.perf_counter() - tick
        tick = time.perf_counter()
        subprocess.run([sys.executable, 'score_predictions.py', '--split', 'validation',
                        '--predictions', str(pred), '--output', str(score)], check=True)
        scoring_wall = time.perf_counter() - tick
        if name == 'programs':
            assets = [Path(checkpoint) / 'programs.npz', Path(checkpoint) / 'base_bundle.npz']
        else:
            assets = [Path(checkpoint)]
        result = json.loads(score.read_text())
        result.update(prediction_wall_seconds=prediction_wall, scoring_wall_seconds=scoring_wall,
                      total_wall_seconds=prediction_wall + scoring_wall,
                      asset_bytes=sum(p.stat().st_size for p in assets),
                      asset_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets})
        assert result['asset_bytes'] <= 16 * 2**20
        results[name] = result
    # Same frozen assets must work on arbitrary bank subsets, IDs and question order.
    bank, arrays = load_inputs('validation', DATA)
    order = [7, 2, 7]
    changed = copy.deepcopy([bank[i] for i in order])
    for i, row in enumerate(changed): row['id'] = f'new-unseen-id-{i}'
    from importlib import import_module
    contracts = {}
    for name, method, checkpoint in configs:
        module = import_module(Path(method).stem)
        expected = json.loads((OUT / f'{name}_predictions.json').read_text())['selected']
        got = module.predict(changed, {'embeddings': arrays['embeddings'][order]}, Path(checkpoint))
        assert got == [expected[i] for i in order], (name, got)
        assert module.predict([], {'embeddings': arrays['embeddings'][:0]}, Path(checkpoint)) == []
        contracts[name] = 'passed: subset, reorder, duplicate question, new IDs, empty bank'
    final = dict(integrity=integrity, validation_resources=results, input_contracts=contracts,
                 threads=4, test_evaluated=False,
                 note='Windows local validation timings; not a test-bank or reference-Xeon certification.')
    (OUT / 'verification.json').write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == '__main__': main()
