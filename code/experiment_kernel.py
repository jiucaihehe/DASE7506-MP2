"""Small conditional RBF basis, selected with train-only group CV."""
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
from experiment_bilinear import projection

OUT=Path('runs/kernel_experiment')


def latent(x,m,kind):
    e=x[...,46:]; c=e.mean(1)
    context=(c-m['cm'])@m['cp'][:,-8:]
    context=np.broadcast_to(context[:,None,:],(len(x),4,8))
    if kind=='semantic':
        r=(e-c[:,None,:])@m['rp']
    elif kind=='hand': r=(x[...,:46]-m['hm'])/m['hs']
    else: raise ValueError(kind)
    # Balance the context and candidate portions instead of letting dimension decide.
    z=np.concatenate([r/np.sqrt(r.shape[-1]),context/np.sqrt(8)],axis=-1)
    return z.reshape(-1,z.shape[-1])


def kernel(z,anchors,width):
    d=np.maximum((z*z).sum(1)[:,None]+(anchors*anchors).sum(1)[None,:]-2*z@anchors.T,0)
    return np.exp(-d/(2*width))


def make_transform(x,kind):
    m=projection(x,32)
    m['hm']=x[...,:46].mean((0,1)); m['hs']=x[...,:46].std((0,1))+1e-8
    z=latent(x,m,kind)
    rng=np.random.default_rng(7506)
    m['anchors']=z[rng.choice(len(z),256,replace=False)]
    return m


def expand(x,m,kind,width):
    k=kernel(latent(x,m,kind),m['anchors'],width).reshape(len(x),4,-1)
    return np.concatenate([x,k],axis=-1)


def fit(x,y,penalty):
    mu=x.mean((0,1)); sd=x.std((0,1))+1e-8
    z=((x-mu)/sd).reshape(-1,x.shape[-1]); n=len(y)
    penalties=np.full(z.shape[-1],.02); penalties[1070:]=penalty
    def obj(w):
        s=(z@w).reshape(n,4)
        a=logsumexp(s,axis=1,keepdims=True)
        b=logsumexp(np.where(y,s,-np.inf),axis=1,keepdims=True)
        ds=np.exp(s-a)-np.where(y,np.exp(s-b),0)
        return (a-b).mean()+.5*np.sum(penalties*w*w),z.T@ds.ravel()/n+penalties*w
    r=minimize(obj,np.zeros(z.shape[-1]),jac=True,method='L-BFGS-B',
        options={'maxiter':250,'ftol':1e-10,'gtol':1e-7})
    return dict(w=r.x,mu=mu,sd=sd)


def score(x,m): return ((x-m['mu'])/m['sd'])@m['w']


def main():
    OUT.mkdir(parents=True,exist_ok=True); tick=time.perf_counter()
    b,a=load_inputs('train',DATA); x=build_matrix(b,a).reshape(len(b),4,-1)
    y=labels(b,'train')[0].reshape(-1,4).astype(bool)
    configs=[dict(kind=k,width=v,penalty=p) for k in ('semantic','hand') for v in (.5,2.) for p in (.2,1.)]
    (OUT/'protocol.json').write_text(json.dumps(dict(seed=7506,folds=3,configs=configs),indent=2))
    oof=np.zeros((len(configs),len(b),4))
    for fold,(tr,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=7506).split(x,y.sum(1))):
        transforms={k:make_transform(x[tr],k) for k in ('semantic','hand')}
        for i,c in enumerate(configs):
            t=transforms[c['kind']]
            xt=expand(x[tr],t,c['kind'],c['width']); xv=expand(x[held],t,c['kind'],c['width'])
            m=fit(xt,y[tr],c['penalty']); oof[i,held]=score(xv,m)
        print(json.dumps(dict(fold=fold,seconds=time.perf_counter()-tick,
            hard=[accuracy(s[held],y[held])['hard_correct'] for s in oof])),flush=True)
    results=[dict(config=c,oof=accuracy(s,y)) for c,s in zip(configs,oof)]
    winner=max(range(len(configs)),key=lambda i:results[i]['oof']['hard_correct']); cfg=configs[winner]
    (OUT/'cv_selection.json').write_text(json.dumps(dict(winner=cfg,results=results),indent=2))
    np.savez(OUT/'oof.npz',scores=oof,labels=y)
    print(json.dumps(dict(winner=cfg,results=results),indent=2),flush=True)
    t=make_transform(x,cfg['kind']); m=fit(expand(x,t,cfg['kind'],cfg['width']),y,cfg['penalty'])
    np.savez(OUT/'selected_bundle.npz',**t,**m,kind=np.array(cfg['kind']),width=cfg['width'])
    vb,va=load_inputs('validation',DATA); vx=build_matrix(vb,va).reshape(len(vb),4,-1)
    vy=labels(vb,'validation')[0].reshape(-1,4).astype(bool)
    s=score(expand(vx,t,cfg['kind'],cfg['width']),m)
    np.savez(OUT/'validation.npz',scores=s,labels=vy)
    result=dict(config=cfg,selected=accuracy(s,vy),seconds=time.perf_counter()-tick)
    (OUT/'validation_results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
