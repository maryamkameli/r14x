#!/usr/bin/env python3
"""Remove a leading <think></think> block from correct_text and every wrong_texts entry when it is empty (whitespace
only; Ivan 2026-10-26). Non-empty think blocks are left alone. Same columns and row order as the input.

  python scripts/strip_empty_think.py extended.parquet extended_nothinkprefix.parquet
"""
import collections, json, re, sys

import pyarrow as pa, pyarrow.parquet as pq

EMPTY = re.compile(r'\s*<think>\s*</think>\s*')


def strip(text, c, key):
    m = EMPTY.match(text)
    c[key] += bool(m)
    return text[m.end():] if m else text


src, dst = sys.argv[1], sys.argv[2]
p = pq.ParquetFile(src)
c = collections.Counter()
w = None
for g in range(p.metadata.num_row_groups):
    t = p.read_row_group(g)
    rows = t.to_pylist()
    for r in rows:
        r['correct_text'] = strip(r['correct_text'], c, 'correct_stripped')
        r['wrong_texts'] = [strip(x, c, 'wrong_stripped') for x in r['wrong_texts']]
        c['rows'] += 1
        c['wrong_texts'] += len(r['wrong_texts'])
    out = pa.Table.from_pylist(rows, schema=t.schema)
    w = w or pq.ParquetWriter(dst, t.schema, compression='zstd')
    w.write_table(out)
w.close()
print(json.dumps(dict(c)))
