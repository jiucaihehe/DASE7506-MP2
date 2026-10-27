"""Fresh-process official evaluation of the frozen candidate, with all wall time.

Explicit split required. Never fits or selects a method using the test result.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='4'
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--split',choices=('validation','test'),required=True)
    args=parser.parse_args(); root=Path('runs/final_candidate')
    freeze=json.loads(Path('runs/candidate_freeze.json').read_text())
    config=json.loads((root/'config.json').read_text())
    assert config==freeze['config'], 'Frozen config changed'
    assert hashlib.sha256((root/'selected_bundle.npz').read_bytes()).hexdigest()==config['checkpoint_sha256']
    for source,sha in freeze['source_sha256'].items():
        raw=Path(source).read_bytes()
        exact=hashlib.sha256(raw).hexdigest()==sha
        normalized=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()
        assert exact or normalized==freeze.get('source_sha256_lf',{}).get(source),source
    output=Path('runs/final_evaluation'); output.mkdir(exist_ok=True,parents=True)
    result_path=output/f'{args.split}_resources.json'
    if args.split=='test' and result_path.exists():
        raise FileExistsError('Test already evaluated. Read the saved result instead of rerunning.')
    pred=output/f'{args.split}_predictions.json'; score=output/f'{args.split}_score.json'
    start=time.perf_counter()
    subprocess.run([sys.executable,'predict.py','--method','submission_candidate.py',
        '--checkpoint',str(root),'--split',args.split,'--output',str(pred)],check=True)
    prediction=time.perf_counter()-start
    start=time.perf_counter()
    subprocess.run([sys.executable,'score_predictions.py','--split',args.split,
        '--predictions',str(pred),'--output',str(score)],check=True)
    scoring=time.perf_counter()-start
    result=dict(split=args.split,official_score=json.loads(score.read_text()),
        prediction_wall_seconds=prediction,scoring_wall_seconds=scoring,
        total_wall_seconds=prediction+scoring,threads=4,
        local_cpu='Intel Core Ultra 9 275HX',platform=sys.platform,
        inference_asset_bytes=sum(p.stat().st_size for p in root.iterdir()),
        checkpoint_sha256=config['checkpoint_sha256'],
        reference_machine_certified=False,test_used_for_selection=False)
    result_path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
