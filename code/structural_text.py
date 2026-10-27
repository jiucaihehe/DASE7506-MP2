"""Delexicalized mathematical relation features and frozen NumPy/SciPy inference."""
import json
import re
from fractions import Fraction
import numpy as np
from common import parse_answer
from features import build_matrix
from program_transfer import question_numbers

RELATIONS=set('total altogether sum remain left rest each per every twice triple half quarter third percent percentage more less fewer than times difference add plus subtract minus multiply divide equal ratio average at least most increase decrease after before first second last initial final original now then extra another combined together including excluding except need required cost price spend spent pay paid earn profit loss discount tax interest dollars cents minute hour day week month year second meter kilometer mile foot feet inch pound ounce gallon liter cup distance speed rate area volume length width height square cube buy bought sell sold give gave receive take took return share split equally enough whole complete round integer'.split())
TOKEN=re.compile(r'\d[\d,]*(?:\.\d+)?(?:/\d+)?|[a-z]+|[+*/=<>%\-]|[.!?;\n]',re.I)


def analyze(doc):
    q,text=doc.split('\n@@@\n',1)
    vals=question_numbers(q); roles={}
    for i,v in reversed(list(enumerate(vals))): roles[v]=f'q{i}'
    text=text.lower().replace('\\times','*').replace('\\cdot','*').replace('\\div','/')
    seq=[]
    for token in TOKEN.findall(text):
        if token[0].isdigit():
            try:
                v=Fraction(token.replace(',',''))
                out=roles.get(v, 'c'+str(v) if v in (0,1,2,7,12,24,60,100) else 'value')
            except ValueError: out='value'
        elif token in '+*/=<>%-': out=token
        elif token in '.!?;\n': out='stop'
        else:
            out=token[:-1] if token.endswith('s') and token[:-1] in RELATIONS else token
            if out not in RELATIONS: out='other'
        if out in ('other','stop') and seq and seq[-1]==out: continue
        seq.append(out)
    grams=[t for t in seq if t not in ('other','stop')]
    for n in (2,3):
        grams += ['|'.join(seq[i:i+n]) for i in range(len(seq)-n+1) if any(t not in ('other','stop') for t in seq[i:i+n])]
    return grams


def documents(bank):
    return [r['question']+'\n@@@\n'+c['text'] for r in bank for c in r['candidates']]


def fit_vectorizer(bank):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec=TfidfVectorizer(analyzer=analyze,max_features=1800,min_df=15,sublinear_tf=True,dtype=np.float64)
    t=vec.fit_transform(documents(bank))
    scale=np.maximum(np.sqrt(np.asarray(t.power(2).mean(0)).ravel()),.03)
    return vec,t.multiply(1/scale).tocsr(),scale


def predict(bank,arrays,checkpoint):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not bank: return []
    with np.load(checkpoint,allow_pickle=False) as m:
        vocab=json.loads(m['vocabulary'].item().decode())
        vec=TfidfVectorizer(analyzer=analyze,sublinear_tf=True,vocabulary=vocab)
        vec.fit(['x\n@@@\nx']); vec.idf_=m['idf']
        t=vec.transform(documents(bank)).multiply(1/m['text_scale'])
        x=build_matrix(bank,arrays)
        s=((x-m['mu'])/m['sd'])@m['w'][:1070]+t@m['w'][1070:]
    return [int(c) for c in s.reshape(len(bank),4).argmax(1)]
