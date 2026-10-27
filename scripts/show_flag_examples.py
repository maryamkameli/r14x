#!/usr/bin/env python3
"""Show what the error-flag cut removed: examples from CORRECT rewrites, and the traces dropped because the flagged
sentence also held the \\boxed answer. Also counts empty think blocks.

  python scripts/show_flag_examples.py pairs_right_wrong_qwen3-0.6b_26904.parquet
"""
import collections, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq  # noqa: E402
from trace_filters import ERROR_FLAG, SENT_SPLIT, cut_error_flags, error_flags  # noqa: E402

t = pq.read_table(sys.argv[1], columns=['luffy_index', 'right_text', 'wrong_text']).to_pylist()
EMPTY = re.compile(r'^\s*<think>\s*</think>\s*')
print('empty think block at start: right', sum(bool(EMPTY.match(r['right_text'])) for r in t),
      '| wrong', sum(bool(EMPTY.match(r['wrong_text'])) for r in t), '| any <think> at all: right',
      sum('<think>' in r['right_text'] for r in t), 'wrong', sum('<think>' in r['wrong_text'] for r in t), 'of', len(t))

# which phrases trigger in CORRECT rewrites, and the sentences they sit in
phr, ex = collections.Counter(), collections.defaultdict(list)
for r in t:
    for line in r['right_text'].split('\n'):
        for s in SENT_SPLIT.split(line):
            m = ERROR_FLAG.search(s)
            if m:
                k = m.group(0).lower()
                phr[k] += 1
                if len(ex[k]) < 3:
                    ex[k].append((r['luffy_index'], s.strip()[:260]))
print('\nCORRECT rewrites: flagged sentences by phrase:', phr.most_common(10))
for k, _ in phr.most_common(5):
    for i, s in ex[k]:
        print(f'  [{k}] luffy {i}: {s}')

# traces kept-flagged after the cut (flag in the same sentence as \\boxed)
print('\nDropped because the flagged sentence also contains \\boxed:')
for f in ('right_text', 'wrong_text'):
    for r in t:
        new, _ = cut_error_flags(r[f])
        if error_flags(new):
            s = next(s for line in new.split('\n') for s in SENT_SPLIT.split(line) if ERROR_FLAG.search(s))
            print(f'  {f} luffy {r["luffy_index"]}: {s.strip()[:300]}')
