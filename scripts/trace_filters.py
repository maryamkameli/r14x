"""Text-quality checks for stored traces (positives and negatives), shared by gen_negatives.py / rewrite_traces.py.

Ivan (2026-10-25): stored traces must not
  - leak that they are wrong: phrases that flag an error ("here we mistakenly", "this is incorrect", "I made a mistake")
    would let the policy learn the label from the wording instead of from the math;
  - switch language: small Qwen models drift into Chinese mid-reasoning ("wo jinwei ..."), see e.g. the language-mixing
    issue of R1-style models.
Checks are on raw text; `scope` decides which part of a thinking trace is checked ('all' or 'final' = after </think>).
"""
import re

# CJK Unified Ideographs (+Ext A, compatibility), CJK punctuation, fullwidth forms, Hiragana/Katakana, Hangul
CJK = re.compile(r'[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\uac00-\ud7af]')

# phrases that state or admit an error. Kept specific: bare "wait"/"let me check" are normal reasoning and not flagged.
ERROR_FLAG = re.compile(r"""(?ix)
    \bmistak(?:e|es|en|enly)\b | \berroneous(?:ly)?\b | \bin\s+error\b | \bmade\s+an?\s+error\b | \ban?\s+error\s+(?:in|here|occurred)\b
  | \bincorrect(?:ly)?\b | \bnot\s+correct\b | \bwrong(?:ly)?\b | \bmiscalculat\w* | \bmiscount\w* | \bmisread\w* | \bmisappl\w*
  | \bflaw(?:ed)?\b | \bfallac\w* | \boops\b | \bthat'?s\s+not\s+right\b | \bthis\s+is\s+not\s+right\b | \bnot\s+the\s+right\s+answer\b
  | \bdeliberate(?:ly)?\b | \bintentional(?:ly)?\b | \bon\s+purpose\b | \bfaulty\b | \binvalid\s+(?:step|assumption|reasoning)\b
""")


def part(text, scope):
    if scope == 'final' and '</think>' in text:
        return text.split('</think>', 1)[1]
    return text


def has_cjk(text, scope='all'):
    return bool(CJK.search(part(text, scope)))


def error_flags(text, scope='all', max_hits=5):
    """Matched error-flag phrases (lowercased, up to max_hits) in the chosen part of the text."""
    return [m.group(0).lower() for m in ERROR_FLAG.finditer(part(text, scope))][:max_hits]


SENT_SPLIT = re.compile(r'(?<=[.!?])[ \t]+(?=[A-Z*(\\$])')  # sentence boundary inside a line


def cut_error_flags(text):
    """Remove every sentence that contains an error-flag phrase (Ivan 2026-10-25: cut 'wait, that's a mistake'
    instead of dropping the trace). Works line by line; within a line, sentence by sentence; whitespace and line
    structure are kept, emptied lines are dropped. A flagged sentence that also holds a \\boxed{} answer is NOT cut
    (the answer would be lost): such a trace keeps its flag and is rejected later by error_flags().
    Returns (new_text, n_sentences_cut)."""
    out, cut = [], 0
    for line in text.split('\n'):
        if not ERROR_FLAG.search(line):
            out.append(line)
            continue
        keep = []
        for s in SENT_SPLIT.split(line):
            if ERROR_FLAG.search(s) and '\\boxed' not in s:
                cut += 1
            else:
                keep.append(s)
        if keep:
            out.append(' '.join(keep))
    new = re.sub(r'\n{3,}', '\n\n', '\n'.join(out))
    return new, cut


def quality(text):
    """Dict of flags stored with every trace: cjk / error flags for the whole text and for the final part."""
    return dict(cjk_any=has_cjk(text, 'all'), cjk_final=has_cjk(text, 'final'),
                errflag_any=error_flags(text, 'all'), errflag_final=error_flags(text, 'final'))
