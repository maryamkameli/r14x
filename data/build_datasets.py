#!/usr/bin/env python3
"""Build the five R1x4 training sets (one parquet each), rows sorted by LUFFY visit order.

  luffy21k  the 21k R1x4 set used for the 0.5B no-decline run (its own wrong pool), all visits
  core      first 39,000 visits: neggen core negatives (wrong + no leak + complete + R1 window + clean final part),
            plus the lab machine's pre-existing negatives for the 6,604 visits not regenerated (taken from luffy21k)
  extended  first 39,000 visits: neggen extended negatives (wrong + no leak + complete, any length) + the same existing ones
  r1qwen    only DeepSeek-R1-Distill-Qwen-1.5B negatives (extended criteria, in-window ones first)
  qwen3     only Qwen3 negatives (--qwen3_prefix: qwen3-8b once generated; qwen3-0.6b thinking-mode for now)

Row schema (one R1 text, distinct wrongs; the trainer copies the R1 text into every correct slot):
  prompt (chat list, LUFFY system prompt), data_source, reward_model{ground_truth, style}, correct_text,
  wrong_texts (1..max_wrong distinct), wrong_meta (json per wrong), luffy_index, visit.
Every R1 text must re-grade CORRECT and every wrong must re-grade WRONG under the training grader (reward v4 =
math_verify on the whole response, truncated to max_response_length tokens); failures are dropped and counted.

  python build_datasets.py --sets core,extended,r1qwen,qwen3,luffy21k --out /mnt/data/radke149/r1x4/data
"""
import argparse, collections, json, os, sys
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'verl'))
from verl.r1x4.reward import grade_texts  # noqa: E402  (same grader the trainer uses)

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument('--sets', default='luffy21k,core,extended,r1qwen,qwen3')
ap.add_argument('--out', required=True)
ap.add_argument('--src', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'),
                help='dir with openr1.parquet, visit_order_rows.txt, luffy21k_r1x4.parquet, usable_negatives.jsonl, negatives_extended.jsonl')
ap.add_argument('--first', type=int, default=39000, help='neggen sets cover the first N LUFFY visits')
ap.add_argument('--max_wrong', type=int, default=8, help='distinct wrong traces kept per prompt')
ap.add_argument('--qwen3_prefix', default='qwen3-0.6b', help="model tag prefix for the qwen3 set ('qwen3-8b' once generated)")
ap.add_argument('--tokenizer', default='Qwen/Qwen3-0.6B', help='only for the length cut before re-grading (Qwen2.5/Qwen3 share the vocab)')
ap.add_argument('--max_response_length', type=int, default=8192)
ap.add_argument('--workers', type=int, default=max(4, (len(os.sched_getaffinity(0)) or 8) - 2))
ap.add_argument('--no_verify', action='store_true')
args = ap.parse_args()
S = lambda f: os.path.join(args.src, f)
os.makedirs(args.out, exist_ok=True)

order = [int(l) for l in open(S('visit_order_rows.txt'))]
visit_of = {r: k for k, r in enumerate(order)}
first = set(order[:args.first])
luffy = pq.read_table(S('openr1.parquet'), columns=['data_source', 'prompt', 'target', 'reward_model']).to_pylist()
print(f'openr1 rows {len(luffy)}, visit order {len(order)}', flush=True)

# luffy21k: its own (already re-graded) wrong pool; distinct texts only
w21 = {}
for r in pq.read_table(S('luffy21k_r1x4.parquet'), columns=['wrong_target_lst', 'extra_info']).to_pylist():
    ei = r['extra_info']
    texts = list(dict.fromkeys(r['wrong_target_lst']))
    w21[ei['luffy_index']] = [(t, {'source': s}) for t, s in zip(texts, ei['wrong_sources'] or ['luffy21k'] * len(texts))]


def load_neg(path):
    """luffy_index -> [(text, meta)], natural before forced_close, in-window before out-of-window, R1-distill first."""
    by = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        key = (r['kind'] != 'natural', not r.get('in_window', True), not r['model'].startswith('r1d'), r['n_tokens'])
        by[r['luffy_index']].append((key, r['text'], {k: r.get(k) for k in ('model', 'kind', 'n_tokens', 'core', 'in_window')}))
    return {i: [(t, m) for _, t, m in sorted(v, key=lambda x: x[0])] for i, v in by.items()}


sets = args.sets.split(',')
core = load_neg(S('usable_negatives.jsonl')) if 'core' in sets else {}
ext = load_neg(S('negatives_extended.jsonl')) if set(sets) & {'extended', 'r1qwen', 'qwen3'} else {}
queued = {r for r in order[:args.first] if r in core or r in ext}


def pool_for(name, i):
    if name == 'luffy21k':
        return w21.get(i, [])
    if i not in first:
        return []
    if name in ('core', 'extended'):
        own = (core if name == 'core' else ext).get(i, [])
        return own or w21.get(i, [])  # visits the lab machine already covered: its negatives (luffy21k pool)
    if name == 'r1qwen':
        return [x for x in ext.get(i, []) if x[1]['model'].startswith('r1d')]
    if name == 'qwen3':  # thinking-mode only: '_nothink' and Qwen2.5 tags are excluded
        return [x for x in ext.get(i, []) if x[1]['model'].startswith(args.qwen3_prefix) and 'nothink' not in x[1]['model']]
    raise ValueError(name)


tok = None
if not args.no_verify:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)


def cut(text):  # the response the trainer will see: at most max_response_length tokens
    if len(text) < 2 * args.max_response_length:  # >= 2 chars/token in practice: cannot exceed the cap
        return text
    ids = tok(text, add_special_tokens=False).input_ids
    return text if len(ids) < args.max_response_length else tok.decode(ids[:args.max_response_length - 1])


for name in sets:
    cand = sorted((i for i in (w21 if name == 'luffy21k' else first) if pool_for(name, i)), key=lambda i: visit_of.get(i, 10 ** 9))
    rows, dropped = [], collections.Counter()
    jobs = []  # (row position, 'c' | wrong index, text, gold)
    for i in cand:
        if i not in visit_of:
            dropped['not in LUFFY visit order (prompt filter)'] += 1
            continue
        rec = luffy[i]
        wr = list({t: m for t, m in pool_for(name, i)}.items())[:args.max_wrong]
        rows.append({'data_source': rec['data_source'], 'prompt': rec['prompt'], 'reward_model': rec['reward_model'],
                     'correct_text': rec['target'][0]['content'], 'wrong_texts': [t for t, _ in wr],
                     'wrong_meta': [json.dumps(m) for _, m in wr], 'luffy_index': i, 'visit': visit_of[i]})
    if not args.no_verify:
        for k, r in enumerate(rows):
            g = r['reward_model']['ground_truth']
            jobs.append((k, 'c', cut(r['correct_text']), g))
            jobs += [(k, j, cut(t), g) for j, t in enumerate(r['wrong_texts'])]
        print(f'[{name}] grading {len(jobs)} texts with {args.workers} workers', flush=True)
        ok = grade_texts([j[2] for j in jobs], [j[3] for j in jobs], workers=args.workers)
        bad_c, bad_w = set(), collections.defaultdict(set)
        for (k, j, _, _), o in zip(jobs, ok):
            if j == 'c' and o != 1:
                bad_c.add(k)
            elif j != 'c' and o != 0:
                bad_w[k].add(j)
        kept = []
        for k, r in enumerate(rows):
            if k in bad_c:
                dropped['R1 not graded correct'] += 1
                continue
            keep = [j for j in range(len(r['wrong_texts'])) if j not in bad_w[k]]
            dropped['wrong traces graded correct/ungraded (trace dropped)'] += len(r['wrong_texts']) - len(keep)
            if not keep:
                dropped['no wrong trace left'] += 1
                continue
            r['wrong_texts'] = [r['wrong_texts'][j] for j in keep]
            r['wrong_meta'] = [r['wrong_meta'][j] for j in keep]
            kept.append(r)
        rows = kept
    rows.sort(key=lambda r: r['visit'])
    path = os.path.join(args.out, f'{name}.parquet')
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd', row_group_size=1000)
    man = {'set': name, 'rows': len(rows), 'dropped': dict(dropped), 'first_visits': args.first if name != 'luffy21k' else None,
           'max_wrong': args.max_wrong, 'wrong_distinct_hist': dict(collections.Counter(len(r['wrong_texts']) for r in rows)),
           'qwen3_prefix': args.qwen3_prefix if name == 'qwen3' else None, 'verified': not args.no_verify,
           'grader': 'reward v4: math_verify(parse($gold$), parse(response)), response cut to max_response_length tokens'}
    json.dump(man, open(path + '.manifest.json', 'w'), indent=1)
    print(json.dumps(man), flush=True)
