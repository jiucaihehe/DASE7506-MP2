"""Frozen NumPy-only inference for round-four development families.

Pure feature functions match experiment_five_new.py. No fitting or reference IO.
"""
import re
import zlib
from pathlib import Path
import numpy as np
from features import build_matrix


def expand(x, proj, kind, dim):
    e=x[...,46:]
    context=(e.mean(1)-proj['cm'])@proj['cp'][:,-dim:]
    products=(context[:,None,:,None]*x[...,:46][:,:,None,:]).reshape(len(x),4,-1)
    return np.concatenate([x,products],axis=-1)


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


def scores(bank, arrays, checkpoint):
    if not bank:
        return np.empty((0,4))
    x=build_matrix(bank,arrays).reshape(len(bank),4,-1)
    with np.load(Path(checkpoint),allow_pickle=False) as m:
        kind=str(m['kind'])
        if kind not in ('control','splines','topic_dro','markov','reference_plan','nuisance_projection'):
            raise ValueError('Unknown round-four model')
        z=transform(kind,x,bank,m,m)
        return ((z-m['mu'])/m['sd'])@m['w']


def predict(bank, arrays, checkpoint):
    return [int(i) for i in scores(bank,arrays,checkpoint).argmax(1)]
