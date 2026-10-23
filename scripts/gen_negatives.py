#!/usr/bin/env python3
"""Generate our own NEGATIVE (wrong) traces, in the jsonl format data/build_datasets.py reads, to compare with Ivan's.

Each model answers openr1 problems unaided, prompted like the student in training: dataset prompt (LUFFY system prompt
+ problem) with the model's own chat template, T=1 top_p=1 top_k=-1 (--sampling card for model-card values), up to
8192 tokens, --n samples per problem. Every sample is graded with the trainer's grader (verl.r1x4.reward).

A sample is a usable negative when (our reading of the build_datasets.py docstring; Ivan's exact neggen rules are not
in the repo, so these are assumptions to check against his files):
  wrong     grader returns 0 (not 1, not -1 = timeout/ungraded)
  complete  finish_reason == 'stop' and the thinking is closed ('</think>' present for thinking models)
  clean     the final part (after '</think>') contains a \\boxed{} answer
  no leak   no EARLIER \\boxed{} in the trace equals the gold answer (math_verify), i.e. it did not reach the right
            answer and then change it
  core      = all four above AND in_window
  in_window n_tokens within [--win_lo, --win_hi] x the R1 trace length of the same problem (default 0.25-1.5)
Output: <out>.jsonl, one usable negative per line: luffy_index, text, model, kind='natural', n_tokens, core,
in_window, + extra: final_answer, rep4 (1 - distinct-4-gram ratio; high = loops), r1_tokens.
<out>_stats.json: per-model counts (drawn / correct / wrong / complete / clean / no-leak / core) and quality stats.

  python scripts/gen_negatives.py --model qwen3-0.6b --first 256 --n 8 --out $R1X4_ROOT/data/neg/neg_qwen3-0.6b
"""
import argparse, json, os, re, statistics as st, sys

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, 'verl'))
MODELS = {  # tag = the 'model' field (build_datasets.py: r1qwen set = tags starting 'r1d', qwen3 set = '--qwen3_prefix')
    'qwen3-0.6b': dict(path='Qwen/Qwen3-0.6B', think=True, card=dict(temperature=0.6, top_p=0.95, top_k=20)),
    'r1d-qwen-1.5b': dict(path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', think=True, card=dict(temperature=0.6, top_p=0.95, top_k=-1)),
    'qwen2.5-0.5b': dict(path='Qwen/Qwen2.5-0.5B-Instruct', think=False, card=dict(temperature=0.7, top_p=0.8, top_k=20)),
}
ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model', required=True, choices=list(MODELS))
ap.add_argument('--data', default=os.path.join(os.environ.get('R1X4_ROOT', '.'), 'data', 'src', 'openr1.parquet'))
ap.add_argument('--first', type=int, default=256)
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--max_tokens', type=int, default=8192)
ap.add_argument('--sampling', default='train', choices=['train', 'card'])
ap.add_argument('--win_lo', type=float, default=0.25)
ap.add_argument('--win_hi', type=float, default=1.5)
ap.add_argument('--gpu_mem', type=float, default=0.85)
ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--out', required=True, help='path prefix: writes <out>.jsonl and <out>_stats.json')
a = ap.parse_args()
M = MODELS[a.model]
BOX = re.compile(r'\\boxed\s*\{')


def boxed_all(text):
    """Contents of every \\boxed{...} (balanced braces), in order."""
    out = []
    for m in BOX.finditer(text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            out.append(text[m.end():i - 1])
    return out


def rep4(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def leaked_batch(texts, golds, workers=16):
    """For each trace: True if an EARLIER \\boxed{} (not the last one) equals the gold answer (it had the answer and
    changed it). Graded with grade_texts (worker processes + hard timeout): math_verify can hang in C code
    (verl/r1x4/reward.py), so it must never be called in the main process. Timeouts (-1) count as no leak."""
    from verl.r1x4.reward import grade_texts
    jobs = [(k, '\\boxed{' + b + '}', g) for k, (t, g) in enumerate(zip(texts, golds)) for b in boxed_all(t)[:-1]]
    out = [False] * len(texts)
    if jobs:
        for (k, _, _), v in zip(jobs, grade_texts([j[1] for j in jobs], [j[2] for j in jobs], workers=workers)):
            out[k] = out[k] or v == 1
    return out


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.r1x4.reward import grade_texts
    tok = AutoTokenizer.from_pretrained(M['path'])
    t = pq.read_table(a.data, columns=['prompt', 'target', 'reward_model'])
    rows = t.slice(0, min(a.first, t.num_rows) if a.first > 0 else t.num_rows).to_pylist()
    r1_tokens = [len(tok(r['target'][0]['content'], add_special_tokens=False).input_ids) for r in rows]
    texts = [tok.apply_chat_template(r['prompt'], add_generation_prompt=True, tokenize=False) for r in rows]
    llm = LLM(M['path'], dtype='bfloat16', gpu_memory_utilization=a.gpu_mem, seed=a.seed, max_model_len=2048 + a.max_tokens)
    sp = SamplingParams(n=a.n, max_tokens=a.max_tokens, **(dict(temperature=1.0, top_p=1.0, top_k=-1) if a.sampling == 'train' else M['card']))
    print(f'[neg] {a.model}: {len(rows)} problems x {a.n}, sampling {sp.temperature}/{sp.top_p}/{sp.top_k}', flush=True)
    gens = llm.generate(texts, sp, use_tqdm=False)
    flat = [(i, o) for i, g in enumerate(gens) for o in g.outputs]
    golds = [rows[i]['reward_model']['ground_truth'] for i, _ in flat]
    grade = grade_texts([o.text for _, o in flat], golds, workers=16)
    c = dict(drawn=len(flat), correct=0, ungraded=0, wrong=0, complete=0, clean=0, no_leak=0, in_window=0, core=0)
    negs, per_problem, cand = [], {}, []
    for (i, o), g, gold in zip(flat, grade, golds):
        if g == 1:
            c['correct'] += 1
            continue
        if g == -1:
            c['ungraded'] += 1
            continue
        c['wrong'] += 1
        text = o.text
        final = text.split('</think>', 1)[1] if '</think>' in text else ('' if M['think'] else text)
        complete = o.finish_reason == 'stop' and (not M['think'] or '</think>' in text)
        if not complete:
            continue
        c['complete'] += 1
        fb = boxed_all(final)
        if not fb:
            continue
        c['clean'] += 1
        cand.append((i, o, gold, fb))
    leaks = leaked_batch([o.text for _, o, _, _ in cand], [g for _, _, g, _ in cand])
    for (i, o, gold, fb), lk in zip(cand, leaks):
        if lk:
            continue
        text = o.text
        c['no_leak'] += 1
        n_tok = len(o.token_ids)
        win = a.win_lo * r1_tokens[i] <= n_tok <= a.win_hi * r1_tokens[i]
        c['in_window'] += win
        c['core'] += win
        per_problem[i] = per_problem.get(i, 0) + 1
        negs.append(dict(luffy_index=i, text=text, model=a.model, kind='natural', n_tokens=n_tok, core=bool(win),
                         in_window=bool(win), final_answer=fb[-1], rep4=round(rep4(list(o.token_ids)), 4), r1_tokens=r1_tokens[i]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + '.jsonl', 'w') as f:
        for x in negs:
            f.write(json.dumps(x) + '\n')
    q = lambda v: [sorted(v)[int(p * (len(v) - 1))] for p in (0, .25, .5, .75, 1)] if v else []
    stats = dict(model=M['path'], problems=len(rows), n=a.n, sampling=a.sampling, counts=c,
                 problems_with_negative=len(per_problem), problems_with_core_negative=len({x['luffy_index'] for x in negs if x['core']}),
                 neg_tokens_q0_25_50_75_100=q([x['n_tokens'] for x in negs]), r1_tokens_median=st.median(r1_tokens),
                 neg_over_r1_median=round(st.median(x['n_tokens'] / x['r1_tokens'] for x in negs), 3) if negs else None,
                 rep4_median=round(st.median(x['rep4'] for x in negs), 4) if negs else None,
                 rep4_over_0_5=sum(x['rep4'] > 0.5 for x in negs))
    json.dump(stats, open(a.out + '_stats.json', 'w'), indent=1)
    print(json.dumps(stats, indent=1), flush=True)
    print('NEG_DONE', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()

