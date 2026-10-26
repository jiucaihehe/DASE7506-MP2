"""Arithmetic verification of candidate traces.

Extracts 'expr = value' claims from candidate text and actually evaluates the
left-hand side. A candidate whose visible computations are wrong is likely
wrong overall — a verifier signal the trace embeddings may not carry.
"""
import re
from fractions import Fraction

_FRAC_RE = re.compile(r'\\d?frac\{(-?[\d.,]+)\}\{(-?[\d.,]+)\}')
_NUM = r'-?\d[\d,]*(?:\.\d+)?'
_EQ_RE = re.compile(rf'([-\d.,/*+()\s×·÷]{{3,60}}?)\s*=\s*(?P<rhs>{_NUM})(?![\d.])')
_OP_RE = re.compile(r'[+*/×·÷-]')


def _normalize(text):
    t = text
    t = _FRAC_RE.sub(r'(\1/\2)', t)
    t = t.replace('\\times', '*').replace('×', '*').replace('\\cdot', '*').replace('·', '*')
    t = t.replace('\\div', '/').replace('÷', '/')
    t = t.replace('$', ' ').replace('\\left', '').replace('\\right', '')
    t = t.replace('\\!', '').replace('\\,', ' ').replace('%', '')
    return t


def _eval_lhs(expr):
    expr = expr.replace(',', '').strip()
    if not _OP_RE.search(expr):
        return None
    if re.search(r'[^\d./*+()\s-]', expr):
        return None
    # reject things that are just ranges or dates like '9-12'
    try:
        return Fraction(eval(expr, {'__builtins__': {}}, {}))
    except Exception:
        return None


def check_text(text):
    """Return (n_checked, n_wrong): counts of verified/violated computations."""
    t = _normalize(text)
    n_checked = n_wrong = 0
    for m in _EQ_RE.finditer(t):
        lhs, rhs_s = m.group(1), m.group('rhs')
        val = _eval_lhs(lhs)
        if val is None:
            continue
        try:
            rhs = Fraction(rhs_s.replace(',', ''))
        except (ValueError, ZeroDivisionError):
            continue
        n_checked += 1
        diff = abs(val - rhs)
        tol = max(Fraction(1, 90), abs(rhs) / 200)  # ~2-decimal rounding or 0.5% relative
        if diff > tol:
            n_wrong += 1
    return n_checked, n_wrong


def arith_features(texts):
    """(4, 4) per candidate: log n_checked, log n_wrong, frac_wrong, has_wrong."""
    out = []
    for t in texts:
        c, w = check_text(t)
        out.append([float(c), float(w),
                    w / c if c else 0.0,
                    float(w > 0)])
    return out
