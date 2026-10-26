"""Round: add TF-IDF text features to the group-softmax model."""
import json
import numpy as np

from common import DATA, load_inputs
from train import get_data
from train2 import acc_hard, group_mask_hard
from train6 import fit_fullbatch
import textvec

tr_bank, tr_arrays = load_inputs('train', DATA)
va_bank, va_arrays = load_inputs('validation', DATA)
tr_texts = [c['text'] for row in tr_bank for c in row['candidates']]
va_texts = [c['text'] for row in va_bank for c in row['candidates']]

Xtr, Xva, ytr, yva, tr_answers, va_answers = get_data()
va_hard = group_mask_hard(yva)
tr_hard = group_mask_hard(ytr)
ng_tr = len(ytr) // 4

for maxf in (2000, 4000, 8000):
    vocab = textvec.build_vocab(tr_texts, max_features=maxf)
    idf = textvec.fit_idf(tr_texts, vocab)
    Ttr = textvec.transform(tr_texts, vocab, idf)
    Tva = textvec.transform(va_texts, vocab, idf)
    Xtr2 = np.hstack([Xtr, Ttr])
    Xva2 = np.hstack([Xva, Tva])
    for lam in (0.02, 0.05):
        w, b, mu, sd = fit_fullbatch(Xtr2, ytr, ng_tr, lam=lam, lr=0.02, epochs=1500)
        str_ = ((Xtr2 - mu) / sd) @ w + b
        sva = ((Xva2 - mu) / sd) @ w + b
        print(json.dumps({'tag': f'tfidf{maxf} lam={lam}',
                          'train_hard': acc_hard(str_, tr_answers, ytr, 'argmax', tr_hard),
                          'val_all': acc_hard(sva, va_answers, yva),
                          'val_hard': acc_hard(sva, va_answers, yva, 'argmax', va_hard)}), flush=True)
