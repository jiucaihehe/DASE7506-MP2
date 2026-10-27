"""Frozen conditional RBF basis, NumPy-only prediction."""
import numpy as np
from features import build_matrix


def predict(bank,arrays,checkpoint):
    if not bank: return []
    x=build_matrix(bank,arrays).reshape(len(bank),4,-1)
    with np.load(checkpoint,allow_pickle=False) as m:
        e=x[...,46:]; c=e.mean(1)
        context=(c-m['cm'])@m['cp'][:,-8:]
        context=np.broadcast_to(context[:,None,:],(len(x),4,8))
        r=((e-c[:,None,:])@m['rp']) if str(m['kind'])=='semantic' else (x[...,:46]-m['hm'])/m['hs']
        z=np.concatenate([r/np.sqrt(r.shape[-1]),context/np.sqrt(8)],axis=-1).reshape(len(x)*4,-1)
        anchors=m['anchors']
        d=np.maximum((z*z).sum(1)[:,None]+(anchors*anchors).sum(1)[None,:]-2*z@anchors.T,0)
        k=np.exp(-d/(2*float(m['width']))).reshape(len(x),4,-1)
        features=np.concatenate([x,k],axis=-1)
        scores=((features-m['mu'])/m['sd'])@m['w']
    return [int(c) for c in scores.argmax(1)]
