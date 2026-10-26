#!/usr/bin/env python3
"""Scan stored traces for CJK text and error-flag phrases (scripts/trace_filters.py) and print how many survive.

  python scripts/scan_trace_quality.py $R1X4_ROOT/data/neg/neg_*_first256.jsonl          # negatives (jsonl, field text)
  python scripts/scan_trace_quality.py $R1X4_ROOT/data/rewrite/rw_*_capped_first256.parquet  # rewrites (field best)
"""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_filters import quality  # noqa: E402


def load(path):
    """-> {field: [(problem id, text)]}. jsonl negatives (text), our rewrites (best), Ivan's pairs (right/wrong_text)."""
    if path.endswith('.jsonl'):
        return {'text': [(r['luffy_index'], r['text']) for r in map(json.loads, open(path))]}
    import pyarrow.parquet as pq
    cols = pq.read_schema(path).names
    if 'right_text' in cols:
        t = pq.read_table(path, columns=['luffy_index', 'right_text', 'wrong_text']).to_pylist()
        return {'right_text': [(r['luffy_index'], r['right_text']) for r in t],
                'wrong_text': [(r['luffy_index'], r['wrong_text']) for r in t]}
    return {'best': [(r['row'], r['best']) for r in pq.read_table(path, columns=['row', 'best']).to_pylist() if r['best']]}


def report(path, rows):
    q = [quality(t) for _, t in rows]
    n = len(rows)
    keep_all = [i for i, x in enumerate(q) if not x['cjk_any'] and not x['errflag_any']]
    keep_final = [i for i, x in enumerate(q) if not x['cjk_any'] and not x['errflag_final']]
    words = collections.Counter(w for x in q for w in set(x['errflag_any']))
    print(f'\n== {path}: {n} traces from {len({p for p, _ in rows})} problems')
    print(f'  CJK anywhere {sum(x["cjk_any"] for x in q)} | CJK in final part {sum(x["cjk_final"] for x in q)}')
    print(f'  error-flag phrase anywhere {sum(bool(x["errflag_any"]) for x in q)} | in final part {sum(bool(x["errflag_final"]) for x in q)}')
    print(f'  survive strict (no CJK, no flag anywhere): {len(keep_all)} traces, {len({rows[i][0] for i in keep_all})} problems')
    print(f'  survive lenient (no CJK, no flag in final part): {len(keep_final)} traces, {len({rows[i][0] for i in keep_final})} problems')
    print('  top flag phrases:', words.most_common(12))
    ex = next((rows[i][1] for i, x in enumerate(q) if x['cjk_any']), None)
    if ex:
        import re
        m = re.search(r'[\u4e00-\u9fff]', ex)
        print('  CJK example:', repr(ex[max(0, m.start() - 120):m.start() + 80]))
    return {p for i, (p, _) in enumerate(rows) if not q[i]['cjk_any'] and not q[i]['errflag_any']}


for path in sys.argv[1:]:
    fields = load(path)
    keep = {f: report(f'{os.path.basename(path)} [{f}]', rows) for f, rows in fields.items()}
    if len(keep) == 2:
        both = keep['right_text'] & keep['wrong_text']
        print(f'  ==> problems where BOTH the right and the wrong text pass strict: {len(both)} of {len(fields["right_text"])}')
