"""Checks for scripts/trace_filters.py (pure Python, no GPU).

    cd verl && python -m pytest -q ../r1x4_tests/test_trace_filters.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from trace_filters import cut_error_flags, error_flags, has_cjk  # noqa: E402


def test_cut_removes_only_flagged_sentences():
    t = ('So x = 3. Wait, that is a mistake. Let me redo it: x = 4.\n'
         '\n'
         'I think I made an error here.\n'
         'Therefore the answer is \\boxed{4}.')
    new, n = cut_error_flags(t)
    assert n == 2
    assert 'mistake' not in new and 'made an error' not in new
    assert 'So x = 3.' in new and 'Let me redo it: x = 4.' in new and '\\boxed{4}' in new
    assert error_flags(new) == []


def test_flag_next_to_the_answer_is_kept_and_still_detected():
    t = 'Earlier step was wrong, so the answer is \\boxed{5}.'
    new, n = cut_error_flags(t)
    assert n == 0 and new == t and error_flags(new)


def test_clean_text_unchanged():
    t = 'Let $a = 2$. Then $a^2 = 4$.\n\n$$\\boxed{4}$$'
    assert cut_error_flags(t) == (t, 0)


def test_cjk():
    assert has_cjk('the latter equals total minus前者.')
    assert not has_cjk('plain English with $\\frac{1}{2}$')
    assert has_cjk('<think>英文</think> answer', 'all') and not has_cjk('<think>英文</think> answer', 'final')
