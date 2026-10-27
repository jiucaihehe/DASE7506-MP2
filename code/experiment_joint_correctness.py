"""Joint conditional likelihood over nonempty correctness subsets; no test access."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from common import DATA, load_inputs, parse_answer
from features import build_matrix
from train_final import labels
from experiment_context import accuracy

OUT=Path('runs/joint_experiment')
STATES=((np.arange(1,16)[:,None] >> np.arange(4)) & 1).astype(float)
COUNTS=np.eye(4)[STATES.sum(1).astype(int)-1]


def legal_states(bank):
    legal=np.ones((len(bank),15), dtype=bool)
    for n,row in enumerate(bank):
        ans=[parse_answer(c['text']) for c in row['candidates']]
        for j in range(4):
            if ans[j] is None:
                legal[n] &= STATES[:,j]==0
            for k in range(j):
                if ans[j] is not None and ans[j]==ans[k]:
                    legal[n] &= STATES[:,j]==STATES[:,k]
    # New banks can have no parseable answer. A uniform posterior then ties at zero.
    legal[~legal.any(1)] = True
    return legal


def posterior(unary, bias, legal):
    logits=unary @ STATES.T + COUNTS @ bias
    logits=np.where(legal, logits, -np.inf)
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def fit(x,y,legal,lam):
    mu=x.mean((0,1)); sd=x.std((0,1))+1e-9
    z=((x-mu)/sd).reshape(-1,x.shape[-1]); d=z.shape[1]
    target=np.argmax((STATES[None,:,:]==y[:,None,:]).all(2),axis=1)
    assert legal[np.arange(len(y)),target].all()
    def objective(w):
        unary=(z@w[:d]).reshape(-1,4)
        logits=unary@STATES.T + COUNTS@w[d:]
        logits=np.where(legal,logits,-np.inf)
        lse=logsumexp(logits,axis=1,keepdims=True)
        p=np.exp(logits-lse)
        loss=(lse[:,0]-logits[np.arange(len(y)),target]).mean()
        du=p@STATES-y
        db=(p@COUNTS-COUNTS[target]).mean(0)
        reg=np.r_[np.full(d,lam),np.full(4,.001)]
        grad=np.r_[z.T@du.ravel()/len(y),db]+reg*w
        return loss+.5*np.dot(w,reg*w),grad
    opt=minimize(objective,np.zeros(d+4),jac=True,method='L-BFGS-B',
        options={'maxiter':350,'ftol':1e-11,'gtol':1e-7})
    return dict(w=opt.x[:d],bias=opt.x[d:],mu=mu,sd=sd),dict(
        success=bool(opt.success),iterations=int(opt.nit),gradient_max=float(abs(opt.jac).max()))


def scores(x,m,legal):
    unary=((x-m['mu'])/m['sd'])@m['w']
    return posterior(unary,m['bias'],legal)@STATES


def main():
    tick=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    lambdas=[.005,.02,.08]
    (OUT/'protocol.json').write_text(json.dumps(dict(seed=7506,folds=3,lambdas=lambdas,
        selection='Train OOF hard correct; earliest tie; one validation evaluation.',
        model='Joint likelihood of all correct/incorrect labels, constrained by parsed-answer equality.',
        promotion='Must beat frozen validation hard 368/419; no test selection.'),indent=2))
    b,a=load_inputs('train',DATA); x=build_matrix(b,a).reshape(len(b),4,-1)
    y=labels(b,'train')[0].reshape(-1,4).astype(bool); legal=legal_states(b)
    oof=np.zeros((len(lambdas),len(b),4)); diagnostics=[]
    for fold,(tr,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=7506).split(x,y.sum(1))):
        for i,lam in enumerate(lambdas):
            m,diag=fit(x[tr],y[tr],legal[tr],lam)
            oof[i,held]=scores(x[held],m,legal[held])
            diagnostics.append(dict(fold=fold,lam=lam,**diag))
            print(json.dumps(dict(fold=fold,lam=lam,held=accuracy(oof[i,held],y[held]),
                                 seconds=time.perf_counter()-tick,**diag)),flush=True)
    cv=[dict(lam=l,metrics=accuracy(s,y)) for l,s in zip(lambdas,oof)]
    winner=max(range(len(lambdas)),key=lambda i:cv[i]['metrics']['hard_correct'])
    (OUT/'cv_selection.json').write_text(json.dumps(dict(winner=lambdas[winner],results=cv,
                                                      diagnostics=diagnostics),indent=2))
    np.savez(OUT/'oof.npz',scores=oof,labels=y)
    model,diag=fit(x,y,legal,lambdas[winner]); np.savez(OUT/'selected_bundle.npz',**model)
    vb,va=load_inputs('validation',DATA); vx=build_matrix(vb,va).reshape(len(vb),4,-1)
    vy=labels(vb,'validation')[0].reshape(-1,4).astype(bool)
    s=scores(vx,model,legal_states(vb))
    from method_bilinear import scores as frozen_scores
    fs=frozen_scores(vb,va,Path('runs/final_candidate/selected_bundle.npz'))
    fg=vy[np.arange(len(vy)),fs.argmax(1)]; ng=vy[np.arange(len(vy)),s.argmax(1)]
    result=dict(lam=lambdas[winner],selected=accuracy(s,vy),frozen=accuracy(fs,vy),
                rescued=int((ng&~fg).sum()),regressed=int((~ng&fg).sum()),
                seconds=time.perf_counter()-tick,test_evaluated=False,fit=diag)
    np.savez(OUT/'validation.npz',scores=s,labels=vy)
    (OUT/'validation_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
