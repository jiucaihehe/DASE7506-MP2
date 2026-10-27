"""Verify a public immutable commit and reproduce it in a clean NumPy-only venv.

No training or method selection; only fixed-checkpoint reproduction.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'code/runs/submission_audit'
PUBLIC_FILES=['README.md','code/submission_candidate.py','code/method_bilinear.py',
    'code/runs/final_candidate/config.json','code/runs/final_candidate/selected_bundle.npz',
    'code/runs/final_evaluation/test_score.json','report/REPORT_V2.md',
    'output/pdf/REPORT_V2.pdf','output/checkpoint/context_candidate.zip']


def download(url):
    last=None
    for attempt in range(3):
        try:
            # No credentials, cookies or authenticated connector are used.
            with urllib.request.urlopen(url,timeout=20) as r:
                return r.status,r.read()
        except Exception as e:
            last=e
            if attempt<2: time.sleep(1)
    raise last


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--python',type=Path,required=True)
    args=p.parse_args();checkout=args.checkout.resolve();python=args.python.resolve()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    assert sha==args.commit,(sha,args.commit)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=checkout,text=True).strip()
    base=f'https://raw.githubusercontent.com/jiucaihehe/DASE7506-MP2/{sha}/'
    def check(name):
        status,b=download(base+name)
        assert b==(checkout/name).read_bytes(),name
        if name.endswith('context_candidate.zip'):
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                assert set(z.namelist())=={'config.json','selected_bundle.npz'}
                for key in z.namelist():
                    assert z.read(key)==(checkout/'code/runs/final_candidate'/key).read_bytes()
        return name,dict(http_status=status,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),matches_clean_clone=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        access=dict(ex.map(check,PUBLIC_FILES))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'publication_access.json').write_text(json.dumps(dict(verified_commit=sha,
        unauthenticated_access=True,files=access),indent=2),encoding='utf-8')
    print('All public artifact URLs downloaded and verified.',flush=True)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        env[key]='4'
    env['PYTHONIOENCODING']='utf-8'
    code=checkout/'code'
    packages=subprocess.check_output([str(python),'-m','pip','freeze'],cwd=code,env=env,text=True).splitlines()
    assert packages==['numpy==2.5.4'],packages
    no_training=subprocess.check_output([str(python),'-c',
        "import importlib.util,json,sys; print(json.dumps({'python':sys.version,'scipy':importlib.util.find_spec('scipy') is not None,'sklearn':importlib.util.find_spec('sklearn') is not None}))"],cwd=code,env=env,text=True)
    dependencies=json.loads(no_training);assert not dependencies['scipy'] and not dependencies['sklearn']
    data=subprocess.run([str(python),'check_data.py'],cwd=code,env=env,text=True,capture_output=True,check=True)
    data_result=json.loads(data.stdout);assert data_result['status']=='passed'
    # Keep audit outputs outside the checkout, so its tracked working tree remains clean.
    pred=OUT/'clean_clone_predictions.json';score=OUT/'clean_clone_score.json'
    tick=time.perf_counter()
    prediction=subprocess.run([str(python),'predict.py','--method','submission_candidate.py',
        '--checkpoint','runs/final_candidate','--split','test','--output',str(pred)],cwd=code,env=env,
        text=True,capture_output=True,check=True)
    prediction_wall=time.perf_counter()-tick
    tick=time.perf_counter()
    subprocess.run([str(python),'score_predictions.py','--split','test','--predictions',str(pred),
        '--output',str(score)],cwd=code,env=env,text=True,capture_output=True,check=True)
    scoring_wall=time.perf_counter()-tick
    actual=json.loads(score.read_text());expected=json.loads((code/'runs/final_evaluation/test_score.json').read_text())
    assert actual['correct']==expected['correct']==392 and actual['n']==467
    assert actual['accuracy']==expected['accuracy']==83.94004282655246
    a=json.loads(pred.read_text());b=json.loads((code/'runs/final_evaluation/test_predictions.json').read_text())
    assert a['ids']==b['ids'] and a['selected']==b['selected']
    assets=code/'runs/final_candidate';asset_bytes=sum(p.stat().st_size for p in assets.iterdir())
    assert asset_bytes==307183 and prediction_wall+scoring_wall<30
    freeze=json.loads((code/'runs/candidate_freeze.json').read_text())
    for name,h in freeze['source_sha256_lf'].items():
        assert hashlib.sha256((code/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h
    checkpoint_sha=hashlib.sha256((assets/'selected_bundle.npz').read_bytes()).hexdigest()
    assert checkpoint_sha==freeze['config']['checkpoint_sha256']
    # Verify source-level links resolve within this exact published tree.
    import re
    broken=[]
    for name in ('README.md',):
        for target in re.findall(r'\]\(([^)]+)\)',(checkout/name).read_text(encoding='utf-8')):
            if not target.startswith(('http:','https:','#')) and not (checkout/target).exists():broken.append(target)
    assert not broken,broken
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=checkout,text=True).strip()
    result=dict(status='passed',verified_commit=sha,clean_remote_clone=str(checkout),
        installed_packages=packages,dependency_isolation=dependencies,data_check=data_result,
        official_score=actual,all_467_choices_match_frozen=True,inference_asset_bytes=asset_bytes,
        checkpoint_sha256=checkpoint_sha,prediction_wall_seconds=prediction_wall,
        scoring_wall_seconds=scoring_wall,total_wall_seconds=prediction_wall+scoring_wall,threads=4,
        public_unauthenticated_links_passed=len(access),readme_local_links='passed',
        frozen_sources='all LF-normalized digests match',clean_checkout_after_verification=True,
        no_retraining=True,test_used_for_selection=False,reference_machine_certified=False)
    (OUT/'clean_clone_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
