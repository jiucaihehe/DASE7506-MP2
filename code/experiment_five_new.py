"""Five new model families. Only train/validation, including all auxiliary fits.

Run from code/. Inference helpers need NumPy only; training uses SciPy/sklearn.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import hashlib
import json
import re
import sys
import time
import zlib
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, KFold
from threadpoolctl import threadpool_limits
from common import DATA, load_inputs, read_jsonl
from features import build_matrix
from train_final import labels
from experiment_context import accuracy
from experiment_bilinear import projection, expand, score

OUT=Path('runs/five_new_experiment')
FAMILIES=['splines','topic_dro','markov','reference_plan','nuisance_projection']


def deny_test(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]).replace('\\','/').lower()
        if name.endswith(('gsm8k_test.jsonl','gsm8k_test.npz')):
            raise PermissionError('This development run forbids test-bank access')


def tokens(text):
    return re.findall(r'[a-z]+|<n>|[+*/=\-]',re.sub(r'\d+(?:\.\d+)?','<n>',text.lower()))


def hash_index(s,n):
    return zlib.crc32(s.encode('utf-8'))%n


def sequence(text):
    words=tokens(text)
    return np.array([127]+[hash_index(w,127) for w in words]+[127],dtype=int)


def qmatrix(bank):
    q=np.zeros((len(bank),512))
    for i,row in enumerate(bank):
        words=tokens(row['question'])
        for s in words+[' '.join(words[j:j+2]) for j in range(len(words)-1)]:
            q[i,hash_index(s,512)]+=1
    q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-9)
    return np.c_[np.ones(len(q)),q]


def operations(text):
    text=text.replace('\\times','*').replace('\\cdot','*').replace('\\div','/')
    # Count written operator occurrences, not arithmetic correctness or answer values.
    return np.log1p([len(re.findall(r'\d\s*'+re.escape(op)+r'\s*\d',text))
                     for op in ['+','-','*','/']]+[text.count('=')])


def teacher_fit(kind,bank,y,targets):
    if kind=='reference_plan':
        q=qmatrix(bank); reg=np.eye(q.shape[1])*10.; reg[0,0]=.001
        return dict(teacher_w=np.linalg.solve(q.T@q+reg,q.T@targets))
    counts=np.ones((2,128,128)) # fixed Laplace smoothing
    for i,row in enumerate(bank):
        for j,c in enumerate(row['candidates']):
            seq=sequence(c['text'])
            np.add.at(counts[int(y[i,j])],(seq[:-1],seq[1:]),1.)
    return dict(log_transition=np.log(counts/counts.sum(2,keepdims=True)))


def teacher_features(kind,bank,m):
    if kind=='reference_plan':
        expect=np.maximum(qmatrix(bank)@m['teacher_w'],0)
        actual=np.array([[operations(c['text']) for c in row['candidates']] for row in bank])
        d=actual-expect[:,None,:]
        return np.concatenate([d,abs(d),actual*expect[:,None,:]],axis=-1)
    rows=[]
    for row in bank:
        group=[]
        for c in row['candidates']:
            seq=sequence(c['text']); logp=m['log_transition'][:,seq[:-1],seq[1:]]
            ratio=logp[1]-logp[0]; mid=max(1,len(ratio)//2)
            group.append([ratio.mean(),ratio.std(),np.quantile(ratio,.1),np.quantile(ratio,.9),
                          ratio[:mid].mean(),ratio[mid:].mean() if mid<len(ratio) else ratio.mean(),
                          logp[0].mean(),logp[1].mean()])
        rows.append(group)
    return np.asarray(rows)


def teacher_crossfit(kind,bank,y,targets):
    result=None
    for tr,held in KFold(2,shuffle=True,random_state=7510).split(bank):
        m=teacher_fit(kind,[bank[i] for i in tr],y[tr],targets[tr])
        z=teacher_features(kind,[bank[i] for i in held],m)
        if result is None: result=np.zeros((len(bank),4,z.shape[-1]))
        result[held]=z
    return result,teacher_fit(kind,bank,y,targets)


def fit_rank(x,y,penalty=1.,groups=None,temperature=None):
    mu=x.mean((0,1)); sd=x.std((0,1))+1e-9
    z=((x-mu)/sd).reshape(-1,x.shape[-1]); n=len(y)
    reg=np.full(x.shape[-1],penalty); reg[:1070]=.02; reg[1070:1438]=1.
    if groups is not None:
        membership=np.eye(8)[groups].T
        membership/=np.maximum(membership.sum(1,keepdims=True),1)
    def obj(w):
        s=(z@w).reshape(n,4)
        al=logsumexp(s,axis=1,keepdims=True)
        good=logsumexp(np.where(y,s,-np.inf),axis=1,keepdims=True)
        losses=(al-good)[:,0]
        ds=np.exp(s-al)-np.where(y,np.exp(s-good),0)
        if groups is None:
            weights=np.full(n,1/n); loss=losses.mean()
        else:
            risks=membership@losses
            weights=softmax(risks/temperature)@membership
            loss=temperature*(logsumexp(risks/temperature)-np.log(8))
        return loss+.5*np.dot(reg*w,w), z.T@(ds*weights[:,None]).ravel()+reg*w
    opt=minimize(obj,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',
                 options={'maxiter':350,'ftol':1e-11,'gtol':1e-7})
    return dict(mu=mu,sd=sd,w=opt.x),dict(success=bool(opt.success),iterations=int(opt.nit),
        gradient_max=float(abs(opt.jac).max()),message=str(opt.message))


def prepare(kind,x,bank,y,targets,proj):
    base=expand(x,proj,'hand_gate',8)
    aux={}; groups=None
    if kind=='control' or kind=='topic_dro':
        if kind=='topic_dro':
            context=(x[...,46:].mean(1)-proj['cm'])@proj['cp']
            groups=KMeans(n_clusters=8,n_init=10,random_state=7506).fit_predict(context)
        return base,aux,groups
    if kind=='splines':
        knots=np.quantile(x[...,:46].reshape(-1,46),[.2,.4,.6,.8],axis=0).T
        aux={'knots':knots}
        extra=np.maximum(x[...,:46,None]-knots,0).reshape(len(x),4,-1)
    elif kind in ('markov','reference_plan'):
        extra,aux=teacher_crossfit(kind,bank,y,targets)
    elif kind=='nuisance_projection':
        e=x[...,46:]; c=e.mean(1); centered=c-c.mean(0)
        residual=(e-c[:,None,:]).reshape(-1,1024)
        within=residual.T@residual/len(residual)
        between=centered.T@centered/len(c)
        ridge=np.trace(between)/1024*.1
        _,p=eigh(within,between+ridge*np.eye(1024),subset_by_index=[960,1023])
        aux={'nuisance_p':p}
        extra=(e-c[:,None,:])@p
    else: raise ValueError(kind)
    return np.concatenate([base,extra],axis=-1),aux,groups


def transform(kind,x,bank,proj,aux):
    base=expand(x,proj,'hand_gate',8)
    if kind in ('control','topic_dro'):return base
    if kind=='splines':
        extra=np.maximum(x[...,:46,None]-aux['knots'],0).reshape(len(x),4,-1)
    elif kind in ('markov','reference_plan'):
        extra=teacher_features(kind,bank,aux)
    else:
        e=x[...,46:];extra=(e-e.mean(1,keepdims=True))@aux['nuisance_p']
    return np.concatenate([base,extra],axis=-1)


def main():
    sys.addaudithook(deny_test)
    tick=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    configs=[dict(kind='control',value=1.)]+[dict(kind=k,value=v) for k in FAMILIES
        for v in ((.1,.5) if k=='topic_dro' else (.2,2.))]
    protocol=dict(seed=7506,folds=3,auxiliary_folds=2,configs=configs,
        selection='One winner per family by train OOF hard correct; earliest breaks ties.',
        validation='Exactly one selected configuration per family plus matched control.',
        test_access='Denied by Python audit hook for both test-bank files.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    b,a=load_inputs('train',DATA);x=build_matrix(b,a).reshape(len(b),4,-1)
    y=labels(b,'train')[0].reshape(-1,4).astype(bool)
    refs={r['id']:r['answer'] for r in read_jsonl(DATA/'gsm8k_train.jsonl')}
    targets=np.array([operations(refs[row['id']]) for row in b])
    oof=np.zeros((len(configs),len(b),4));diagnostics=[]
    for fold,(tr,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=7506).split(x,y.sum(1))):
        proj=projection(x[tr],8);tb=[b[i] for i in tr];hb=[b[i] for i in held]
        for kind in ['control']+FAMILIES:
            xt,aux,groups=prepare(kind,x[tr],tb,y[tr],targets[tr],proj)
            xv=transform(kind,x[held],hb,proj,aux)
            for i,cfg in enumerate(configs):
                if cfg['kind']!=kind:continue
                m,diag=fit_rank(xt,y[tr],cfg['value'],groups,
                                cfg['value'] if kind=='topic_dro' else None)
                oof[i,held]=score(xv,m)
                diagnostics.append(dict(fold=fold,config=cfg,**diag))
                print(json.dumps(dict(fold=fold,config=cfg,held=accuracy(oof[i,held],y[held]),
                    seconds=time.perf_counter()-tick,**diag)),flush=True)
    results=[dict(config=c,metrics=accuracy(s,y)) for c,s in zip(configs,oof)]
    winners=[0]+[max([i for i,c in enumerate(configs) if c['kind']==kind],
                    key=lambda i:results[i]['metrics']['hard_correct']) for kind in FAMILIES]
    (OUT/'cv_selection.json').write_text(json.dumps(dict(winner_indices=winners,results=results,
                                                        diagnostics=diagnostics),indent=2))
    np.savez(OUT/'oof.npz',scores=oof,labels=y)
    proj=projection(x,8);models={}
    for i in winners:
        cfg=configs[i];kind=cfg['kind']
        xt,aux,groups=prepare(kind,x,b,y,targets,proj)
        m,diag=fit_rank(xt,y,cfg['value'],groups,cfg['value'] if kind=='topic_dro' else None)
        models[kind]=(m,aux,diag)
        np.savez(OUT/f'{kind}_bundle.npz',**m,**aux,**proj,kind=np.array(kind),dim=8)
        print(json.dumps(dict(full_fit=cfg,seconds=time.perf_counter()-tick,**diag)),flush=True)
    # All choices and full fits are complete before validation is opened.
    vb,va=load_inputs('validation',DATA);vx=build_matrix(vb,va).reshape(len(vb),4,-1)
    vy=labels(vb,'validation')[0].reshape(-1,4).astype(bool)
    from method_bilinear import scores as frozen_scores
    fs=frozen_scores(vb,va,Path('runs/final_candidate/selected_bundle.npz'))
    fg=vy[np.arange(len(vy)),fs.argmax(1)]
    report=dict(frozen=accuracy(fs,vy),results=[],test_evaluated=False)
    for i in winners:
        cfg=configs[i];kind=cfg['kind'];m,aux,diag=models[kind]
        s=score(transform(kind,vx,vb,proj,aux),m);ng=vy[np.arange(len(vy)),s.argmax(1)]
        report['results'].append(dict(config=cfg,selected=accuracy(s,vy),
            rescued=int((ng&~fg).sum()),regressed=int((~ng&fg).sum()),fit=diag,
            asset_bytes=(OUT/f'{kind}_bundle.npz').stat().st_size))
        np.savez(OUT/f'{kind}_validation.npz',scores=s,labels=vy)
    report['seconds']=time.perf_counter()-tick
    (OUT/'validation_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
