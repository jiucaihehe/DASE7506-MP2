"""Instructor entrypoint: reproduce ONLY the frozen 83.94% contextual model.

Runs fresh prediction and the unchanged official independent scorer.
Never discovers models, fits parameters, or loads the historical checkpoint.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
EXPECTED_SHA='cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('train','validation','test'),default='test')
    args=parser.parse_args()
    assets=ROOT/'runs/final_candidate'
    config=json.loads((assets/'config.json').read_text(encoding='utf-8'))
    actual=hashlib.sha256((assets/'selected_bundle.npz').read_bytes()).hexdigest()
    if config['family']!='bilinear' or actual!=EXPECTED_SHA or config['checkpoint_sha256']!=EXPECTED_SHA:
        raise ValueError('This is not the published final contextual checkpoint. No fallback is permitted.')
    # Reproduction outputs are separate from frozen score and checkpoint files.
    output=ROOT/'runs/instructor_reproduction';output.mkdir(parents=True,exist_ok=True)
    pred=output/f'{args.split}_predictions.json';score=output/f'{args.split}_score.json'
    prediction_command=[sys.executable,str(ROOT/'predict.py'),'--method',str(ROOT/'submission_candidate.py'),
        '--checkpoint',str(assets),'--split',args.split,'--output',str(pred)]
    scoring_command=[sys.executable,str(ROOT/'score_predictions.py'),'--split',args.split,
        '--predictions',str(pred),'--output',str(score)]
    print('FINAL METHOD: submission_candidate.py -> method_bilinear.py',flush=True)
    print('FINAL ASSETS: runs/final_candidate/{config.json,selected_bundle.npz}',flush=True)
    print('CHECKPOINT SHA-256: '+actual,flush=True)
    tick=time.perf_counter()
    subprocess.run(prediction_command,cwd=ROOT,check=True)
    prediction_wall=time.perf_counter()-tick
    tick=time.perf_counter()
    subprocess.run(scoring_command,cwd=ROOT,check=True)
    scoring_wall=time.perf_counter()-tick
    official=json.loads(score.read_text(encoding='utf-8'))
    record=dict(method='submission_candidate.py',implementation='method_bilinear.py',
        checkpoint='runs/final_candidate/selected_bundle.npz',checkpoint_sha256=actual,
        historical_model_used=False,training_performed=False,threads=4,official_score=official,
        prediction_wall_seconds=prediction_wall,scoring_wall_seconds=scoring_wall,
        total_wall_seconds=prediction_wall+scoring_wall,
        prediction_command=prediction_command,scoring_command=scoring_command)
    (output/f'{args.split}_reproduction.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
