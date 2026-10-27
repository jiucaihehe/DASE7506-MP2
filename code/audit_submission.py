"""Audit frozen assets in an isolated reference-free inference directory.

Reproduces a fixed method only. No model fitting or selection is performed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='4'
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/submission_audit'

GUARD=r'''
import sys,json,runpy
from pathlib import Path
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        raise RuntimeError('Network forbidden in offline inference audit')
    if event=='import' and args[0].split('.')[0] in {'scipy','sklearn','torch','transformers','train_final','experiment_bilinear','experiment_context'}:
        raise RuntimeError('Training/model dependency forbidden: '+args[0])
    if event=='open':
        try: p=Path(args[0])
        except TypeError: return
        if p.name in {'gsm8k_train.jsonl','gsm8k_validation.jsonl','gsm8k_test.jsonl'} and p.parent.name!='candidates':
            raise RuntimeError('Reference read forbidden during inference')
sys.addaudithook(audit)
sys.argv=['predict.py','--method','submission_candidate.py','--checkpoint','assets','--split','test','--output','predictions.json']
runpy.run_path('predict.py',run_name='__main__')
from common import load_inputs,DATA
from submission_candidate import predict
import copy
bank,arrays=load_inputs('test',DATA)
expected=json.loads(Path('predictions.json').read_text())['selected']
order=[9,3,9,0]
changed=copy.deepcopy([bank[i] for i in order])
for i,row in enumerate(changed): row['id']='unseen-audit-id-'+str(i)
got=predict(changed,{'embeddings':arrays['embeddings'][order]},Path('assets'))
assert got==[expected[i] for i in order]
assert predict([],{'embeddings':arrays['embeddings'][:0]},Path('assets'))==[]
from threadpoolctl import threadpool_info
pool=threadpool_info()
assert all(x['num_threads']<=4 for x in pool),pool
Path('inference_guard.json').write_text(json.dumps(dict(reference_free=True,network_blocked=True,
 training_imports_blocked=True,input_contracts='passed',threadpools=pool),indent=2))
'''


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((ROOT/'runs/candidate_freeze.json').read_text())
    assert json.loads((ROOT/'runs/final_candidate/config.json').read_text())==freeze['config']
    for path,sha in freeze['source_sha256'].items():
        raw=(ROOT/path).read_bytes()
        canonical=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()
        assert digest(ROOT/path)==sha or canonical==freeze.get('source_sha256_lf',{}).get(path),path
    assert digest(ROOT/'runs/final_candidate/selected_bundle.npz')==freeze['config']['checkpoint_sha256']
    with zipfile.ZipFile(ROOT/'runs/final_candidate/selected_bundle.npz') as z:
        uncompressed=sum(i.file_size for i in z.infolist())
    assets=sum(p.stat().st_size for p in (ROOT/'runs/final_candidate').iterdir())
    assert assets<=16*2**20 and uncompressed<=16*2**20
    with tempfile.TemporaryDirectory(prefix='dase7506-frozen-audit-') as temp:
        target=Path(temp).resolve()
        for name in ('common.py','features.py','method_bilinear.py','submission_candidate.py','predict.py','score_predictions.py'):
            # Simulate a Unix clone: line endings must not affect frozen inference.
            (target/name).write_bytes((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))
        shutil.copytree(ROOT/'runs/final_candidate',target/'assets')
        (target/'data/candidates').mkdir(parents=True)
        shutil.copyfile(ROOT/'data/manifest.json',target/'data/manifest.json')
        for suffix in ('jsonl','npz'):
            shutil.copyfile(ROOT/f'data/candidates/gsm8k_test.{suffix}',target/f'data/candidates/gsm8k_test.{suffix}')
        assert not (target/'data/gsm8k_test.jsonl').exists()
        (target/'guard.py').write_text(GUARD,encoding='utf-8')
        tick=time.perf_counter()
        subprocess.run([sys.executable,'guard.py'],cwd=target,check=True,capture_output=True,text=True)
        guard_wall=time.perf_counter()-tick
        pred=json.loads((target/'predictions.json').read_text())
        old=json.loads((ROOT/'runs/final_evaluation/test_predictions.json').read_text())
        assert pred['ids']==old['ids'] and pred['selected']==old['selected']
        guard=json.loads((target/'inference_guard.json').read_text())
        # References enter the directory only for the separate official scorer.
        shutil.copyfile(ROOT/'data/gsm8k_test.jsonl',target/'data/gsm8k_test.jsonl')
        tick=time.perf_counter()
        subprocess.run([sys.executable,'score_predictions.py','--split','test','--predictions','predictions.json',
                        '--output','score.json'],cwd=target,check=True,capture_output=True,text=True)
        score_wall=time.perf_counter()-tick
        score=json.loads((target/'score.json').read_text())
        assert score['correct']==392 and score['n']==467
        shutil.copyfile(target/'score.json',OUT/'reproduced_score.json')
        result=dict(status='passed',official_score=score,all_467_choices_match_frozen=True,
            isolated_source_line_endings='LF (portable-clone simulation)',
            inference_guard=guard,guarded_prediction_plus_contract_checks_wall_seconds=guard_wall,
            independent_scoring_wall_seconds=score_wall,inference_asset_bytes=assets,
            npz_member_uncompressed_bytes=uncompressed,checkpoint_sha256=freeze['config']['checkpoint_sha256'],
            no_model_changes=True,test_used_for_tuning=False,
            note='Guarded audit includes extra input-contract predictions; original fresh evaluation timing remains separately recorded.')
        (OUT/'reproduction_audit.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))


if __name__=='__main__': main()
