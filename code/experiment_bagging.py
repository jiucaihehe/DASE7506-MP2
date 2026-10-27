"""Fixed Bayesian-bootstrap ensemble. All members fit train, no per-seed selection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='4'
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from common import DATA,load_inputs
from features import build_matrix
from train_final import labels
from experiment_context import accuracy
from experiment_bilinear import projection,expand

OUT=Path('runs/bagging_experiment')


def fit(z,y,weights):
    w=np.zeros(z.shape[1]); m=w.copy(); v=w.copy()
    regs=np.full(len(w),.02); regs[1070:]=1.
    weights=weights/weights.mean()
    for step in range(1,1501):
        s=(z@w).reshape(-1,4); s-=s.max(1,keepdims=True)
        p=np.exp(s); p/=p.sum(1,keepdims=True)
        ds=(p-y*p/((y*p).sum(1,keepdims=True)+1e-12))*weights[:,None]
        g=z.T@ds.ravel()/len(y)+regs*w
        m=.9*m+.1*g; v=.999*v+.001*g*g
        w-=.02*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    return w


def softmax(s):
    p=np.exp(s-s.max(1,keepdims=True)); return p/p.sum(1,keepdims=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True); tick=time.perf_counter()
    protocol=dict(seed=7506,members=8,weights='independent Exponential(1) group weights',
                  kind='hand_gate',dim=8,penalty=1.,epochs=1500,
                  selection='all eight members equally weighted, no seed selection')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    b,a=load_inputs('train',DATA); x=build_matrix(b,a).reshape(len(b),4,-1)
    y=labels(b,'train')[0].reshape(-1,4).astype(bool)
    proj=projection(x,16); x=expand(x,proj,'hand_gate',8)
    mu=x.mean((0,1)); sd=x.std((0,1))+1e-9; z=((x-mu)/sd).reshape(-1,x.shape[-1])
    rng=np.random.default_rng(7506); ws=[]
    for i in range(8):
        ws.append(fit(z,y,rng.exponential(1,len(y))))
        print(json.dumps(dict(member=i,seconds=time.perf_counter()-tick)),flush=True)
    ws=np.asarray(ws)
    np.savez(OUT/'selected_bundle.npz',**proj,mu=mu,sd=sd,ws=ws,kind=np.array('hand_gate'),dim=8)
    vb,va=load_inputs('validation',DATA); vx=build_matrix(vb,va).reshape(len(vb),4,-1)
    vy=labels(vb,'validation')[0].reshape(-1,4).astype(bool)
    vx=expand(vx,proj,'hand_gate',8)
    s=((vx-mu)/sd)@ws.T
    probabilities=np.mean(np.stack([softmax(s[:,:,i]) for i in range(8)]),axis=0)
    np.savez(OUT/'validation.npz',scores=probabilities,labels=vy)
    result=dict(selected=accuracy(probabilities,vy),seconds=time.perf_counter()-tick)
    (OUT/'validation_results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
