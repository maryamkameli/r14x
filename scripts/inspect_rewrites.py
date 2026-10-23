#!/usr/bin/env python3
"""Quick look at a rewrite_traces.py output: length stats and a few examples.

  python scripts/inspect_rewrites.py $R1X4_ROOT/data/rewrite/rw_qwen3-0.6b_thinkoff_full_first256.parquet [--show 3]
"""
import argparse, statistics as st
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('--show', type=int, default=3)
a = ap.parse_args()
if a.path.endswith('_natural.parquet'):
    raise SystemExit(0)
r = [x for x in pq.read_table(a.path).to_pylist() if x['n_drawn']]
got = [x for x in r if x['best_tokens'] > 0]
q = lambda v: [sorted(v)[int(p * (len(v) - 1))] for p in (0, .1, .25, .5, .75, .9, 1)]
print(f'{a.path}\nproblems {len(r)}, with a correct rewrite {len(got)}')
if got:
    print('R1 tokens        quantiles 0/10/25/50/75/90/100%:', q([x['orig_tokens'] for x in got]))
    print('best rewrite     quantiles 0/10/25/50/75/90/100%:', q([x['best_tokens'] for x in got]))
    print('all correct samples, median tokens:', st.median(s['n_tokens'] for x in got for s in x['samples'] if s['correct']))
    print('best rewrites under 80 tokens:', sum(x['best_tokens'] < 80 for x in got),
          '| single paragraph (no blank line):', sum('\n\n' not in x['best'] for x in got))
    if 'cap' in got[0]:
        print('cap (natural correct median) quantiles:', q([x['cap'] for x in got]),
              '| kept rewrite / cap median', round(st.median(x['best_tokens'] / x['cap'] for x in got), 3),
              '| max', round(max(x['best_tokens'] / x['cap'] for x in got), 3))
    for x in got[:a.show] + [min(got, key=lambda x: x['best_tokens'])]:
        cap = f', cap {x["cap"]} ({x["cap_source"]})' if 'cap' in x else ''
        print(f'\n===== row {x["row"]}: R1 {x["orig_tokens"]} -> {x["best_tokens"]} tokens{cap}, '
              f'{x["n_correct"]}/{x["n_drawn"]} kept =====')
        print(x['best'][:1500])
