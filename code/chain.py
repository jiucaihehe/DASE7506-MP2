"""Reasoning-chain coherence: is every number in a trace traceable to the
question or to a previous computed result? Untraceable 'magic numbers'
indicate the solver invented values — a structural reasoning signal.
"""
import re
from fractions import Fraction

from arith import _normalize, _EQ_RE, _eval_lhs

_NUM_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def _frac(s):
    try:
        return Fraction(s.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None


def chain_features(question, text):
    """Per-candidate: [magic_ratio, chain_links, n_computed, final_grounded]."""
    q_nums = {v for v in (_frac(m) for m in _NUM_RE.findall(question)) if v is not None}
    t = _normalize(text)

    computed = []  # RHS values in order of appearance
    for m in _EQ_RE.finditer(t):
        lhs_val = _eval_lhs(m.group(1))
        rhs = _frac(m.group('rhs'))
        if rhs is None:
            continue
        lhs_nums = {v for v in (_frac(x) for x in _NUM_RE.findall(m.group(1))) if v is not None}
        computed.append((lhs_val, rhs, lhs_nums))

    rhs_set = [rhs for _, rhs, _ in computed]
    # chain links: this step's LHS mentions an earlier computed value
    links = 0
    for i, (lhs_val, rhs, lhs_nums) in enumerate(computed):
        if any(v in lhs_nums for v in rhs_set[:i]):
            links += 1

    body_nums = [v for v in (_frac(m) for m in _NUM_RE.findall(t)) if v is not None]
    grounded_set = set(q_nums) | set(rhs_set)
    n_magic = sum(1 for v in body_nums if v not in grounded_set)
    magic_ratio = n_magic / len(body_nums) if body_nums else 0.0

    # final answer grounded?
    from common import parse_answer
    ans = parse_answer(text)
    final_grounded = 0.0
    if ans is not None:
        try:
            fa = Fraction(ans)
            final_grounded = float(fa in grounded_set)
        except (ValueError, ZeroDivisionError):
            pass
    return [magic_ratio, float(links), float(len(computed)), final_grounded]
