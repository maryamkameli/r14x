#!/usr/bin/env python3
"""Teacher-conditioned rewrites of the stored R1 traces by a small model (default Qwen3-0.6B, non-thinking mode).

For every row (first --first rows of an R1x4 parquet, LUFFY visit order) the model sees the problem and a reference
solution and writes one response per task in --tasks (1 sample each); --fallback (default wrong_cut) is run only for
rows whose wrong_ref is not a valid wrong rewrite:
  right         reference = R1 trace (`correct_text`); own words, same final answer
  wrong_ref     reference = a stored wrong trace of the same problem; own words, keep its (wrong) final answer
  wrong_cut     reference = first half of the R1 trace; own words, plausible mistake towards a stored wrong answer
  wrong_target  reference = R1 trace; asked to reach a stored wrong answer     (smoke test: ~0 % wrong, 0.6B copies)
  wrong_free    reference = R1 trace; asked not to reach the gold answer        (smoke test: 0 % wrong)
The wrong answer used as target is the boxed answer of the first stored wrong trace whose answer differs from the gold
string. Every task is asked to stay under --words words and is cut at --max_tokens tokens.

Checks (columns, nothing is dropped): finish_reason ('length' = over the token cap), grade under the trainer's grader
(verl.r1x4.reward: 1 correct, 0 wrong, -1 ungraded), ok = stop and graded as the task asks (right: 1, wrong_*: 0).

  CUDA_VISIBLE_DEVICES=0 python scripts/rewrite_traces.py --data extended.parquet --first 30000 \
      --also_done 'rewrite_qwen3-0.6b_extended_first10k.parquet' --shard 0 --num_shards 4 \
      --out rewrite_qwen3-0.6b_extended_first30k.parquet
Output rows: luffy_index, visit, task, target, text, n_tokens, finish_reason, grade, ok   (resumable per chunk)
"""
import argparse, glob, os, sys, time
import pyarrow as pa, pyarrow.parquet as pq

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, 'verl'))
ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument('--model', default='Qwen/Qwen3-0.6B')
ap.add_argument('--data', required=True, help='R1x4 parquet (rows in visit order, with correct_text)')
ap.add_argument('--first', type=int, default=10000)
ap.add_argument('--words', type=int, default=2000, help='length limit stated in the instruction')
ap.add_argument('--max_tokens', type=int, default=4096, help='hard cap on generated tokens')
ap.add_argument('--temperature', type=float, default=0.7)  # Qwen3 non-thinking defaults
ap.add_argument('--top_p', type=float, default=0.8)
ap.add_argument('--top_k', type=int, default=20)
ap.add_argument('--gpu_mem', type=float, default=0.9)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--num_shards', type=int, default=1)
ap.add_argument('--chunk', type=int, default=1024, help='rows per generate() call (one request per row and task)')
ap.add_argument('--tasks', default='right,wrong_ref',
                help='comma list of: right, wrong_ref, wrong_cut, wrong_target, wrong_free')
ap.add_argument('--fallback', default='wrong_cut', help="task run only for rows whose wrong_ref fails ('' = none)")
ap.add_argument('--also_done', default=None, help='glob of earlier output files whose rows count as done')
ap.add_argument('--max_model_len', type=int, default=32768, help='prompt + response; stored wrong traces can exceed 8k')
ap.add_argument('--out', required=True)
a = ap.parse_args()

STYLE = 'Rewrite this solution in your own words, the way you would naturally solve the problem yourself'
TASK = {
    'right': STYLE + '. Follow the reasoning of the reference solution and keep its final answer.',
    # the target is the boxed answer of a stored wrong trace of the same problem (no target: falls back to wrong_free)
    'wrong_target': STYLE + (', but make one plausible mistake along the way so that your final answer is {target} '
                             'instead of {gold}. Do not mention the mistake and do not say that the answer is wrong.'),
    'wrong_free': STYLE + (', but make one plausible mistake along the way so that your final answer is NOT {gold}. '
                           'Do not mention the mistake and do not say that the answer is wrong.'),
    # reference = a stored WRONG trace of the same problem; same instruction as 'right'
    'wrong_ref': STYLE + '. Follow the reasoning of the reference solution and keep its final answer.',
    # reference = first half of the R1 trace (its answer is not visible)
    'wrong_cut': ('The reference solution is unfinished. Solve the problem in your own words, the way you would '
                  'naturally do it, using the start of the reference, but make one plausible mistake so that your '
                  'final answer is {target}. Do not mention the mistake and do not say that the answer is wrong.'),
}
TASKS = a.tasks.split(',')


def last_boxed(s):
    i = s.rfind('\\boxed{')
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {'{': 1, '}': -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1].strip() if depth == 0 else None


def wrong_target(r):
    gold = r['reward_model']['ground_truth'].strip()
    return next((b for b in map(last_boxed, r['wrong_texts']) if b and b != gold), None)


def message(r, task):
    problem = next(m['content'] for m in reversed(r['prompt']) if m['role'] == 'user')
    gold, target, ref = r['reward_model']['ground_truth'], r['target'], r['correct_text']
    if task in ('wrong_target', 'wrong_cut') and target is None:
        task = 'wrong_free'
    if task == 'wrong_ref':
        ref = next((w for w in r['wrong_texts'] if last_boxed(w) == target), r['wrong_texts'][0])
    if task == 'wrong_cut':
        ref = ref[:len(ref) // 2]
    return (f'Problem:\n{problem}\n\nReference solution (written by another model):\n{ref}\n\n---\n\n'
            + TASK[task].format(gold=gold, target=target)
            + f' Put your final answer within \\boxed{{}}. Your whole response must be at most {a.words} words.')


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.r1x4.reward import grade_texts
    cols = ['prompt', 'reward_model', 'correct_text', 'wrong_texts', 'luffy_index', 'visit']
    out = a.out if a.num_shards == 1 else a.out.replace('.parquet', f'_shard{a.shard:02d}of{a.num_shards:02d}.parquet')
    res = pq.read_table(out).to_pylist() if os.path.exists(out) else []
    # rows finished by ANY earlier run (any shard count, or --also_done files) are skipped; the rest is split over shards
    prev = [f for f in glob.glob(a.out.replace('.parquet', '*.parquet')) + (glob.glob(a.also_done) if a.also_done else [])
            if f != out]
    done = {r['luffy_index'] for r in res} | {i for f in prev for i in pq.read_table(f, columns=['luffy_index']).column(0).to_pylist()}
    rows = [r for r in pq.read_table(a.data, columns=cols).to_pylist()[:a.first] if r['luffy_index'] not in done]
    todo = rows[a.shard::a.num_shards]
    for r in todo:
        r['target'] = wrong_target(r)
    tok = AutoTokenizer.from_pretrained(a.model)
    print(f'[rewrite] {a.model} shard {a.shard}/{a.num_shards}: {len(todo)} rows to do ({len(done)} done in any shard)', flush=True)
    llm = LLM(a.model, dtype='bfloat16', gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              enable_prefix_caching=True, seed=1234 + a.shard)
    sp = SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_tokens)
    t_start, n_gen = time.time(), 0

    def run(reqs):
        """Generate + grade one request per (row, task); returns (result rows, generated outputs)."""
        texts = [tok.apply_chat_template([{'role': 'user', 'content': message(r, task)}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False) for r, task in reqs]
        # a few stored wrong traces are degenerate (40k+ tokens): such requests are recorded, not generated
        plen = [len(ids) for ids in tok(texts, add_special_tokens=False).input_ids]
        fit = [k for k, n in enumerate(plen) if n < a.max_model_len]
        gens = dict(zip(fit, llm.generate([texts[k] for k in fit], sp, use_tqdm=False)))
        outs = [gens[k].outputs[0] if k in gens else None for k in range(len(reqs))]
        grades = grade_texts([o.text if o else '' for o in outs], [r['reward_model']['ground_truth'] for r, _ in reqs],
                             workers=16)
        new = []
        for (r, task), o, g in zip(reqs, outs, grades):
            stop = o is not None and o.finish_reason == 'stop'
            new.append({'luffy_index': r['luffy_index'], 'visit': r['visit'], 'task': task,
                        'target': r['target'] if task == 'wrong_target' else None, 'text': o.text if o else '',
                        'n_tokens': len(o.token_ids) if o else 0,
                        'finish_reason': o.finish_reason if o else 'prompt_too_long', 'grade': int(g) if o else -1,
                        'ok': bool(stop and g == (1 if task == 'right' else 0))})
        return new, [o for o in outs if o is not None]

    for c in range(0, len(todo), a.chunk):
        batch, t0 = todo[c:c + a.chunk], time.time()
        new, outs = run([(r, task) for r in batch for task in TASKS])
        if a.fallback:  # second request only for rows whose wrong_ref is not a valid wrong rewrite
            bad = {x['luffy_index'] for x in new if x['task'] == 'wrong_ref' and not x['ok']}
            new2, outs2 = run([(r, a.fallback) for r in batch if r['luffy_index'] in bad])
            new, outs = new + new2, outs + outs2
        res.extend(new)
        pq.write_table(pa.Table.from_pylist(res), out)
        ntok = sum(len(o.token_ids) for o in outs)
        n_gen += ntok
        rate = ' '.join(f'{t} {sum(x["ok"] for x in new if x["task"] == t)}/{sum(x["task"] == t for x in new)}'
                        for t in TASKS + ([a.fallback] if a.fallback else []))
        print(f'  chunk {c // a.chunk}: {len(batch)} rows, {ntok} tokens in {time.time() - t0:.0f} s '
              f'({ntok / (time.time() - t0):.0f} tok/s), cap hits {sum(o.finish_reason == "length" for o in outs)}, '
              f'ok: {rate}', flush=True)
    print(f'REWRITE_DONE {len(res)} rows, {n_gen} tokens generated in {time.time() - t_start:.0f} s', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
