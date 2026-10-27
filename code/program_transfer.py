"""Executable arithmetic-template retrieval, learned from TRAIN reasoning only.

Inference uses saved question TF-IDF vectors and variable-only arithmetic programs.
No question IDs, reference strings, final-answer lookup, or external models.
"""
import ast
import json
import re
from fractions import Fraction
import numpy as np
from scipy.sparse import csr_matrix
from common import parse_answer

WORDS = dict(zero=0, one=1, two=2, three=3, four=4, five=5, six=6,
             seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12,
             dozen=12, hundred=100, thousand=1000)
NUM = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?(?:/\d+)?(?!\w)')
WORDNUM = re.compile(r'\b(' + '|'.join(WORDS) + r')\b', re.I)
SAFE_CONSTANTS = {Fraction(x) for x in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 24, 30, 60, 100, 365, 1000)} | {Fraction('0.5'), Fraction('0.01')}


def question_numbers(question):
    text = WORDNUM.sub(lambda m: str(WORDS[m.group().lower()]), question)
    values = [Fraction(m.group().replace(',', '')) for m in NUM.finditer(text)]
    # Preserve occurrence order; duplicates remain separate slots.
    return values


def normalized_question(question):
    text = WORDNUM.sub(lambda m: str(WORDS[m.group().lower()]), question.lower())
    return NUM.sub(' quantity ', text)


def run_program(program, values):
    if program[0] == 'arity':
        if len(values) != program[1]: raise ValueError('Quantity counts differ')
        return run_program(program[2], values)
    if program[0] == 'v':
        return values[program[1]]
    if program[0] == 'c':
        return Fraction(program[1])
    op, left, right = program
    a, b = run_program(left, values), run_program(right, values)
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '*': return a * b
    if op == '/': return a / b
    raise ValueError('Unknown arithmetic operator')


def extract_program(question, answer):
    values = question_numbers(question)
    if not values or len(values) > 24:
        return None
    known = {}
    for idx, value in reversed(list(enumerate(values))):
        known[value] = ['v', idx]
    used = False
    def translate(node):
        nonlocal used
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = Fraction(str(node.value))
            if value in known:
                used = True
                return known[value]
            if value in SAFE_CONSTANTS:
                return ['c', str(value)]
            raise ValueError('Unexplained literal')
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return ['-', ['c', '0'], translate(node.operand)]
        operators = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return [operators[type(node.op)], translate(node.left), translate(node.right)]
        raise ValueError('Non-arithmetic syntax')
    result = None
    for marker in re.findall(r'<<([^<>]+)>>', answer):
        if '=' not in marker: continue
        expr, right = marker.rsplit('=', 1)
        expr = expr.replace(',', '').replace('×', '*').replace('÷', '/')
        try:
            program = translate(ast.parse(expr, mode='eval').body)
            value = run_program(program, values)
            if value != Fraction(right.replace(',', '').strip()):
                continue
            if len(json.dumps(program)) > 2500:
                continue
            known[value] = program
            result = program
        except (ValueError, SyntaxError, ZeroDivisionError, IndexError, OverflowError):
            continue
    gold = parse_answer(answer)
    if result is None or not used or gold is None:
        return None
    try:
        if run_program(result, values) != Fraction(gold): return None
    except (ZeroDivisionError, ValueError, IndexError):
        return None
    # A template must actually use an input quantity; never save constant answers.
    if '"v"' not in json.dumps(result): return None
    return ['arity', len(values), result]


def fit_retriever(rows, programs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    keep = [i for i, p in enumerate(programs) if p is not None]
    # Word bigrams preserve rate/setup cues; numbers are replaced by a shared token.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=16000,
        min_df=2, sublinear_tf=True, dtype=np.float32)
    matrix = vectorizer.fit_transform([normalized_question(rows[i]['question']) for i in keep])
    return vectorizer, matrix, [programs[i] for i in keep]


def suggested_answers(rows, vectorizer, matrix, programs):
    query = vectorizer.transform([normalized_question(r['question']) for r in rows])
    similarities = (query @ matrix.T).toarray()
    suggestions, confidences = [], []
    for row, sims in zip(rows, similarities):
        order = np.argsort(-sims)[:5]
        values = question_numbers(row['question'])
        suggested, confidence = None, 0.
        # Nearest executable program only. No gold labels are consulted here.
        for idx in order:
            try:
                value = run_program(programs[idx], values)
                if abs(value) > 10 ** 12: continue
                suggested, confidence = str(value), float(sims[idx])
                break
            except (ValueError, IndexError, ZeroDivisionError, OverflowError):
                continue
        suggestions.append(suggested); confidences.append(confidence)
    return suggestions, np.asarray(confidences)


def transfer_scores(rows, base_scores, answers, similarities, threshold):
    scores = base_scores.copy()
    for i, (row, answer, sim) in enumerate(zip(rows, answers, similarities)):
        if answer is None or sim < threshold: continue
        matches = np.array([parse_answer(c['text']) == answer for c in row['candidates']])
        if matches.any():
            scores[i, matches] += np.ptp(scores[i]) + 1
    return scores


def save_retriever(path, vectorizer, matrix, programs, threshold):
    np.savez(path, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
             shape=matrix.shape, idf=vectorizer.idf_,
             vocabulary=np.bytes_(json.dumps({k: int(v) for k, v in vectorizer.vocabulary_.items()}).encode('utf-8')),
             programs=np.bytes_(json.dumps(programs).encode('utf-8')), threshold=threshold)


def load_retriever(path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    with np.load(path, allow_pickle=False) as f:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
            dtype=np.float32, vocabulary=json.loads(f['vocabulary'].item().decode('utf-8')))
        # Initial fit establishes fitted state; supplied IDF is entirely train-fitted.
        vectorizer.fit(['quantity'])
        vectorizer.idf_ = f['idf']
        matrix = csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape']))
        return vectorizer, matrix, json.loads(f['programs'].item().decode('utf-8')), float(f['threshold'])


def predict(bank, arrays, checkpoint):
    from pathlib import Path
    from features import build_matrix
    if not bank: return []
    checkpoint = Path(checkpoint)
    with np.load(checkpoint / 'base_bundle.npz') as model:
        x = build_matrix(bank, arrays).reshape(len(bank), 4, -1)
        base = ((x - model['mu']) / model['sd']) @ model['w']
    vectorizer, matrix, programs, threshold = load_retriever(checkpoint / 'programs.npz')
    answers, similarities = suggested_answers(bank, vectorizer, matrix, programs)
    scores = transfer_scores(bank, base, answers, similarities, threshold)
    return [int(c) for c in scores.argmax(1)]
