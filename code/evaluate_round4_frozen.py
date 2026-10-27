"""Report two already-frozen candidates; never fit, tune, or select on test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('runs/round4_frozen_test'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    os.chdir(root)
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        env[k]='4'
    kinds=['topic_dro','reference_plan']
    prior=json.loads(Path('runs/five_new_experiment/verification.json').read_text())
    expected={r['kind']:r['sha256'] for r in prior['records']}
    models={k:Path(f'runs/five_new_experiment/{k}_bundle.npz') for k in kinds}
    for k,p in models.items():
        assert digest(p)==expected[k],f'Previously verified checkpoint changed: {k}'
    sources=['method_five_new.py','features.py','common.py','predict.py','score_predictions.py']
    freeze=dict(candidates=kinds,checkpoint_sha256={k:digest(p) for k,p in models.items()},
        source_sha256={p:digest(Path(p)) for p in sources},
        submitted_checkpoint_sha256=digest(Path('runs/final_candidate/selected_bundle.npz')),
        protocol='Both candidates fixed before test; both prediction files complete before either scoring. Report both; no test-based model selection or tuning.',
        threads=4,python=sys.version,executable=sys.executable)
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2))
    records={}
    # Complete both predictions before either independent scoring process.
    for k in kinds:
        command=[sys.executable,'predict.py','--method','method_five_new.py','--checkpoint',
                 str(models[k]),'--split','test','--output',str(out/f'{k}_predictions.json')]
        start=time.perf_counter()
        subprocess.run(command,env=env,check=True)
        records[k]=dict(prediction_seconds=time.perf_counter()-start,
                        asset_bytes=models[k].stat().st_size,predict_command=command)
    for k in kinds:
        command=[sys.executable,'score_predictions.py','--split','test','--predictions',
                 str(out/f'{k}_predictions.json'),'--output',str(out/f'{k}_score.json')]
        start=time.perf_counter()
        subprocess.run(command,env=env,check=True)
        records[k]['scoring_seconds']=time.perf_counter()-start
        records[k]['total_seconds']=records[k]['prediction_seconds']+records[k]['scoring_seconds']
        records[k]['official_score']=json.loads((out/f'{k}_score.json').read_text())
        records[k]['score_command']=command
        assert records[k]['total_seconds']<30
        assert records[k]['asset_bytes']<=16*1024*1024
        assert digest(models[k])==freeze['checkpoint_sha256'][k]
    assert all(digest(Path(p))==h for p,h in freeze['source_sha256'].items())
    assert digest(Path('runs/final_candidate/selected_bundle.npz'))==freeze['submitted_checkpoint_sha256']
    result=dict(records=records,current_submission_unchanged=True,
                no_training_or_parameter_changes=True,test_based_selection_performed=False)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
