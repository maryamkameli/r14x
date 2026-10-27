#!/usr/bin/env python3
"""Build the R1x4 training parquet from Qwen3-0.6B rewrite pairs (scripts/rewrite_traces.py output).

One row per problem with a valid correct AND a valid wrong rewrite (wrong_ref preferred, wrong_cut otherwise):
  correct_text = the `right` rewrite, wrong_texts = [the wrong rewrite]
Both are stored as the generator produced them under Qwen3's non-thinking chat template, i.e. prefixed with the empty
think block '<think>\\n\\n</think>\\n\\n' that the template put before the generated text.
Rows keep the source problem's prompt (LUFFY chat), reward_model, data_source, luffy_index and visit (sorted by visit).

  python data/build_rewrite_pairs.py --rewrites rewrite_qwen3-0.6b_extended_first30k.parquet \\
      --source extended.parquet --out rewrite_pairs_qwen3-0.6b.parquet
"""
import argparse, collections, json
import pyarrow as pa, pyarrow.parquet as pq

EMPTY_THINK = '<think>\n\n</think>\n\n'
ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument('--rewrites', required=True)
ap.add_argument('--source', required=True, help='R1x4 parquet the rewrites were made from (prompt, reward_model, ...)')
ap.add_argument('--out', required=True)
a = ap.parse_args()

ok = collections.defaultdict(dict)
for r in pq.read_table(a.rewrites, columns=['luffy_index', 'task', 'text', 'ok']).to_pylist():
    if r['ok']:
        ok[r['luffy_index']][r['task']] = r['text']
src = {r['luffy_index']: r for r in pq.read_table(a.source, columns=['luffy_index', 'visit', 'prompt', 'reward_model', 'data_source']).to_pylist()}
rows, kinds = [], collections.Counter()
for i, t in ok.items():
    wtask = 'wrong_ref' if 'wrong_ref' in t else 'wrong_cut' if 'wrong_cut' in t else None
    if 'right' not in t or wtask is None:
        continue
    s = src[i]
    kinds[wtask] += 1
    rows.append({'data_source': s['data_source'], 'prompt': s['prompt'], 'reward_model': s['reward_model'],
                 'correct_text': EMPTY_THINK + t['right'], 'wrong_texts': [EMPTY_THINK + t[wtask]],
                 'wrong_meta': [json.dumps({'model': 'Qwen/Qwen3-0.6B', 'kind': 'rewrite_' + wtask})],
                 'luffy_index': i, 'visit': s['visit']})
rows.sort(key=lambda r: r['visit'])
pq.write_table(pa.Table.from_pylist(rows), a.out, compression='zstd', row_group_size=1000)
print(f'{len(rows)} rows -> {a.out}; wrong trace kind {dict(kinds)}')
