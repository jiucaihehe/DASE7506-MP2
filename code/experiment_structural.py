"""Train-only CV for role-normalized relation ngrams. No test evaluation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='4'
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from common import DATA,load_inputs
from features import build_matrix
from train_final import labels
from experiment_context import accuracy
from structural_text import fit_vectorizer,documents

OUT=Path('runs/structural_experiment')


def fit(x,t,y,kind,penalty):
    mu=x.mean((0,1)); sd=x.std((0,1))+1e-9
    z=((x-mu)/sd).reshape(-1,x.shape[-1]); n=len(y)
    regs=np.full(z.shape[1]+t.shape[1],.02); regs[z.shape[1]:]=penalty
    target=y/y.sum(1,keepdims=True); hard=(y.sum(1)<4)
    def obj(w):
        s=(z@w[:1070]+t@w[1070:]).reshape(n,4)
        a=logsumexp(s,axis=1,keepdims=True); p=np.exp(s-a)
        if kind=='set':
            b=logsumexp(np.where(y,s,-np.inf),axis=1,keepdims=True)
            loss=(a-b).mean(); ds=p-np.where(y,np.exp(s-b),0)
        else:
            loss=((a[:,0]-(target*s).sum(1))*hard).mean(); ds=(p-target)*hard[:,None]
        g=np.concatenate([z.T@ds.ravel(),np.asarray(t.T@ds.ravel()).ravel()])/n+regs*w
        return loss+.5*np.sum(regs*w*w),g
    r=minimize(obj,np.zeros(len(regs)),jac=True,method='L-BFGS-B',
        options={'maxiter':250,'ftol':1e-10,'gtol':1e-7})
    return dict(w=r.x,mu=mu,sd=sd)


def score(x,t,m):
    return ((((x-m['mu'])/m['sd']).reshape(-1,1070)@m['w'][:1070])+t@m['w'][1070:]).reshape(len(x),4)


def main():
    OUT.mkdir(parents=True,exist_ok=True); tick=time.perf_counter()
    b,a=load_inputs('train',DATA); x=build_matrix(b,a).reshape(len(b),4,-1)
    y=labels(b,'train')[0].reshape(-1,4).astype(bool)
    configs=[dict(kind=k,penalty=p) for k in ('set','uniform') for p in (.05,.5,5.)]
    (OUT/'protocol.json').write_text(json.dumps(dict(seed=7506,folds=3,configs=configs),indent=2))
    oof=np.zeros((len(configs),len(b),4))
    for fold,(tr,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=7506).split(x,y.sum(1))):
        vec,t,scale=fit_vectorizer([b[i] for i in tr])
        tv=vec.transform(documents([b[i] for i in held])).multiply(1/scale).tocsr()
        for i,c in enumerate(configs): oof[i,held]=score(x[held],tv,fit(x[tr],t,y[tr],**c))
        print(json.dumps(dict(fold=fold,seconds=time.perf_counter()-tick,
            hard=[accuracy(s[held],y[held])['hard_correct'] for s in oof])),flush=True)
    results=[dict(config=c,oof=accuracy(s,y)) for c,s in zip(configs,oof)]
    winner=max(range(len(configs)),key=lambda i:results[i]['oof']['hard_correct']); cfg=configs[winner]
    (OUT/'cv_selection.json').write_text(json.dumps(dict(winner=cfg,results=results),indent=2))
    np.savez(OUT/'oof.npz',scores=oof,labels=y)
    print(json.dumps(dict(winner=cfg,results=results),indent=2),flush=True)
    vec,t,scale=fit_vectorizer(b); m=fit(x,t,y,**cfg)
    np.savez(OUT/'selected_bundle.npz',**m,text_scale=scale,idf=vec.idf_,
        vocabulary=np.bytes_(json.dumps({k:int(v) for k,v in vec.vocabulary_.items()}).encode()))
    vb,va=load_inputs('validation',DATA); vx=build_matrix(vb,va).reshape(len(vb),4,-1)
    vy=labels(vb,'validation')[0].reshape(-1,4).astype(bool)
    tv=vec.transform(documents(vb)).multiply(1/scale)
    s=score(vx,tv,m); np.savez(OUT/'validation.npz',scores=s,labels=vy)
    result=dict(config=cfg,selected=accuracy(s,vy),seconds=time.perf_counter()-tick)
    (OUT/'validation_results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
