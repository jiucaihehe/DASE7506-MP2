"""Experimental frozen entrypoint. Model selection and fitting never run here."""
import json
from pathlib import Path


def predict(bank,arrays,checkpoint):
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))
    bundle=checkpoint/'selected_bundle.npz'
    family=config['family']
    if family=='bilinear':
        from method_bilinear import predict as implementation
    elif family=='bagging':
        from method_bagging import predict as implementation
    elif family=='kernel':
        from method_kernel import predict as implementation
    elif family=='structural':
        from structural_text import predict as implementation
    elif family in ('linear','original'):
        import numpy as np
        from features import build_matrix
        if not bank: return []
        with np.load(bundle,allow_pickle=False) as m:
            x=build_matrix(bank,arrays).reshape(len(bank),4,-1)
            scores=((x-m['mu'])/m['sd'])@m['w']
        return [int(c) for c in scores.argmax(1)]
    else: raise ValueError('Unsupported frozen family')
    return implementation(bank,arrays,bundle)
