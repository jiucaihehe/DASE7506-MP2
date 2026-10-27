"""Official validation-only reproduction and contract checks for five new families."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from common import DATA,load_inputs
from method_five_new import predict


def main():
    root=Path(__file__).resolve().parent
    os.chdir(root)
    out=Path('runs/five_new_experiment'); records=[]
    report=json.loads((out/'validation_results.json').read_text())
    b,a=load_inputs('validation',DATA)
    for item in report['results']:
        kind=item['config']['kind']
        if kind=='control':continue
        model=out/f'{kind}_bundle.npz'
        pred=out/f'{kind}_predictions.json'; result=out/f'{kind}_official_score.json'
        t=time.perf_counter()
        subprocess.run([sys.executable,'predict.py','--method','method_five_new.py',
            '--checkpoint',str(model),'--split','validation','--output',str(pred)],check=True)
        predict_seconds=time.perf_counter()-t;t=time.perf_counter()
        subprocess.run([sys.executable,'score_predictions.py','--split','validation',
            '--predictions',str(pred),'--output',str(result)],check=True)
        score_seconds=time.perf_counter()-t
        official=json.loads(result.read_text())
        assert official['correct']==item['selected']['all_correct']
        chosen=predict(b,a,model)
        with np.load(out/f'{kind}_validation.npz') as f:
            assert np.array_equal(chosen,f['scores'].argmax(1))
        idx=[7,2,7,0];new=copy.deepcopy([b[i] for i in idx])
        for i,row in enumerate(new):row['id']=f'unseen-round-four-{i}'
        small={'embeddings':a['embeddings'][idx]}
        assert predict(new,small,model)==[chosen[i] for i in idx]
        assert predict([],{'embeddings':np.empty((0,4,1024))},model)==[]
        assert model.stat().st_size<=16*1024*1024
        assert predict_seconds+score_seconds<30
        records.append(dict(kind=kind,official_score=official,predict_seconds=predict_seconds,
            score_seconds=score_seconds,total_seconds=predict_seconds+score_seconds,
            asset_bytes=model.stat().st_size,sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            diagnostic_choices_match=True,new_ids_reordering_repetition_empty_pass=True))
    (out/'verification.json').write_text(json.dumps(dict(split='validation',threads=4,
        python=sys.version,executable=sys.executable,numpy_version=np.__version__,
        test_evaluated=False,records=records),indent=2))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
