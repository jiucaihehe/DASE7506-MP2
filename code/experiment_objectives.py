"""Train-only CV of convex set-supervised ranking objectives. Never reads test."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from common import DATA, load_inputs
from features import build_matrix
from train_final import labels
from experiment_context import accuracy

OUT = Path('runs/objective_experiment')


def fit(x, y, kind, lam):
    mu = x.mean((0,1)); sd = x.std((0,1)) + 1e-9
    z = ((x-mu)/sd).reshape(-1,x.shape[-1]); ng=len(y)
    hard = (y.sum(1)>0)&(y.sum(1)<4)
    if kind == 'pairwise':
        differences=[]; weights=[]
        zz=z.reshape(ng,4,-1)
        for i in np.flatnonzero(hard):
            a=zz[i,y[i]]; b=zz[i,~y[i]]
            differences.append((a[:,None,:]-b[None,:,:]).reshape(-1,zz.shape[-1]))
            weights.extend([1/(len(a)*len(b)*ng)]*(len(a)*len(b)))
        differences=np.concatenate(differences); weights=np.asarray(weights)
        def objective(w):
            margin=differences@w
            return (np.sum(weights*np.logaddexp(0,-margin))+.5*lam*(w@w),
                    differences.T@(-weights*expit(-margin))+lam*w)
    else:
        target=y/y.sum(1,keepdims=True)
        def objective(w):
            s=(z@w).reshape(ng,4)
            lse=logsumexp(s,axis=1,keepdims=True)
            p=np.exp(s-lse)
            if kind == 'uniform':
                loss=((lse[:,0]-(s*target).sum(1))*hard).mean()
                ds=(p-target)*hard[:,None]
            elif kind == 'set':
                good=logsumexp(np.where(y,s,-np.inf),axis=1,keepdims=True)
                loss=(lse-good).mean()
                ds=p-np.where(y,np.exp(s-good),0)
            else: raise ValueError(kind)
            return loss+.5*lam*(w@w),z.T@ds.ravel()/ng+lam*w
    result=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',
        options={'maxiter':300,'gtol':1e-7,'ftol':1e-11})
    return dict(mu=mu,sd=sd,w=result.x)


def scores(x,m): return ((x-m['mu'])/m['sd'])@m['w']


def main():
    OUT.mkdir(parents=True,exist_ok=True); tick=time.perf_counter()
    b,a=load_inputs('train',DATA); x=build_matrix(b,a).reshape(len(b),4,-1)
    y=labels(b,'train')[0].reshape(-1,4).astype(bool)
    configs=[dict(kind=k,lam=l) for k in ('set','uniform','pairwise') for l in (.005,.02,.08)]
    (OUT/'protocol.json').write_text(json.dumps({'seed':7506,'folds':3,'configs':configs},indent=2))
    oof=np.zeros((len(configs),len(b),4))
    for fold,(tr,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=7506).split(x,y.sum(1))):
        for i,c in enumerate(configs):
            oof[i,held]=scores(x[held],fit(x[tr],y[tr],**c))
        print(json.dumps(dict(fold=fold,seconds=time.perf_counter()-tick,
            hard=[accuracy(s[held],y[held])['hard_correct'] for s in oof])),flush=True)
    cv=[dict(config=c,oof=accuracy(s,y)) for c,s in zip(configs,oof)]
    winner=max(range(len(cv)),key=lambda i:cv[i]['oof']['hard_correct'])
    (OUT/'cv_selection.json').write_text(json.dumps(dict(winner=configs[winner],results=cv),indent=2))
    np.savez(OUT/'oof.npz',scores=oof,labels=y)
    print(json.dumps(dict(winner=configs[winner],results=cv),indent=2),flush=True)
    # CV selects exactly one configuration before validation is loaded.
    model=fit(x,y,**configs[winner]); np.savez(OUT/'selected_bundle.npz',**model)
    vb,va=load_inputs('validation',DATA); vx=build_matrix(vb,va).reshape(len(vb),4,-1)
    vy=labels(vb,'validation')[0].reshape(-1,4).astype(bool)
    s=scores(vx,model)
    np.savez(OUT/'validation.npz',scores=s,labels=vy)
    result=dict(config=configs[winner],selected=accuracy(s,vy),seconds=time.perf_counter()-tick)
    (OUT/'validation_results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
