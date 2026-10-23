"""Unit checks for the pure helpers of scripts/gen_negatives.py (no GPU, no model).

    cd verl && python -m pytest -q ../r1x4_tests/test_gen_negatives_helpers.py
"""
import os
import sys

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _helpers():
    src = open(os.path.join(KIT, 'scripts', 'gen_negatives.py')).read().split('def main():')[0]
    argv, sys.argv = sys.argv, ['gen_negatives.py', '--model', 'qwen3-0.6b', '--out', '/tmp/neg_test']
    ns = {'__file__': os.path.join(KIT, 'scripts', 'gen_negatives.py')}
    try:
        exec(src, ns)
    finally:
        sys.argv = argv
    return ns


H = _helpers()
T = r'<think>try \boxed{5} hmm no</think> so the answer is \boxed{\frac{1}{2}} and finally \boxed{7}'


def test_boxed_all_balanced_braces():
    assert H['boxed_all'](T) == ['5', r'\frac{1}{2}', '7']


def test_leak_only_counts_earlier_boxes():
    assert H['leaked'](T, '5')          # had the gold answer earlier, then changed it
    assert not H['leaked'](T, '7')      # gold only in the final box: that is a correct answer, not a leak
    assert not H['leaked'](T, '3')


def test_rep4():
    assert H['rep4']([1, 2, 3, 4] * 20) > 0.9
    assert H['rep4'](list(range(50))) == 0.0
