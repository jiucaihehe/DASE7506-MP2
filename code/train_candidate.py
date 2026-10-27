"""Refit the already chosen contextual model on TRAIN ONLY, without selection.

Writes a separate checkpoint and refuses to overwrite existing output.
The committed/saved frozen assets can be used directly without running training.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='4'
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from common import DATA,load_inputs
from features import build_matrix
from train_final import labels
from experiment_bilinear import projection,expand,fit_adam


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    bank,arrays=load_inputs('train',DATA)
    x=build_matrix(bank,arrays).reshape(len(bank),4,-1)
    y=labels(bank,'train')[0].reshape(-1,4).astype(bool)
    proj=projection(x,16)
    model=fit_adam(expand(x,proj,'hand_gate',8),y,extra_penalty=1.,epochs=1500)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,**model,**proj,kind=np.array('hand_gate'),dim=8)
    print(f'Train-only fixed-configuration checkpoint: {args.output}')


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
