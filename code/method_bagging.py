"""Frozen posterior-predictive contextual ensemble, NumPy only."""
import numpy as np
from features import build_matrix


def predict(bank,arrays,checkpoint):
    if not bank: return []
    x=build_matrix(bank,arrays).reshape(len(bank),4,-1)
    with np.load(checkpoint,allow_pickle=False) as m:
        e=x[...,46:]; context=(e.mean(1)-m['cm'])@m['cp'][:,-int(m['dim']):]
        products=(context[:,None,:,None]*x[:,:,None,:46]).reshape(len(x),4,-1)
        x=np.concatenate([x,products],axis=-1)
        s=((x-m['mu'])/m['sd'])@m['ws'].T
    s-=s.max(1,keepdims=True); p=np.exp(s); p/=p.sum(1,keepdims=True)
    return [int(c) for c in p.mean(2).argmax(1)]
