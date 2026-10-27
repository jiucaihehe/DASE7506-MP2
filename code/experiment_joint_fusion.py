"""Select complementary model fusion on training OOF only, then validation."""
import json
from pathlib import Path
import numpy as np
from experiment_context import accuracy
from scipy.special import softmax


def main():
    out=Path('runs/joint_fusion'); out.mkdir(parents=True,exist_ok=True)
    weights=[0.,.1,.25,.5,.75,1.]
    (out/'protocol.json').write_text(json.dumps(dict(weights=weights,
        selection='Max train OOF hard correct; earliest weight wins ties.',
        fusion='Convex combination of contextual softmax and joint correctness marginals.',
        validation='One selected fusion only; no test access.'),indent=2))
    selection=json.loads(Path('runs/bilinear_experiment/cv_selection.json').read_text())
    joint=json.loads(Path('runs/joint_experiment/cv_selection.json').read_text())
    ji=[r['lam'] for r in joint['results']].index(joint['winner'])
    with np.load('runs/bilinear_experiment/oof_diagnostics.npz') as f:
        a=softmax(f['scores'][selection['winner_index']],axis=1); y=f['labels']
    with np.load('runs/joint_experiment/oof.npz') as f:
        b=f['scores'][ji]; assert np.array_equal(y,f['labels'])
    metrics=[dict(weight=w,train_oof=accuracy((1-w)*a+w*b,y)) for w in weights]
    wi=max(range(len(weights)),key=lambda i:metrics[i]['train_oof']['hard_correct'])
    w=weights[wi]
    (out/'cv_selection.json').write_text(json.dumps(dict(weight=w,results=metrics),indent=2))
    with np.load('runs/bilinear_experiment/validation_diagnostics.npz') as f:
        av=softmax(f['selected_scores'],axis=1); vy=f['labels']
    with np.load('runs/joint_experiment/validation.npz') as f:
        bv=f['scores']; assert np.array_equal(vy,f['labels'])
    result=dict(weight=w,selected=accuracy((1-w)*av+w*bv,vy),
                frozen=accuracy(av,vy),test_evaluated=False)
    (out/'validation_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(cv=metrics,validation=result),indent=2))


if __name__=='__main__': main()
