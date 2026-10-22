#!/usr/bin/env python3
"""Ask a small model to rewrite a correct solution in its own words (shorter traces).

Input: LUFFY's openr1.parquet (prompt = LUFFY system prompt + problem, target[0] = DeepSeek-R1 trace '<think>..</think>
solution'). Each rewriter gets the problem + the R1 trace and is asked to restate the solution concisely, ending in
\\boxed{}. Every sample is graded with the trainer's grader (verl.r1x4.reward = math_verify on the whole text).

The two rewriters use DIFFERENT chat templates; each prompt is rendered with the rewriter's own tokenizer:
  qwen3-0.6b   Qwen/Qwen3-0.6B (hybrid thinking).
               --thinking off (default): enable_thinking=False; the template pre-fills '<think>\\n\\n</think>\\n\\n' and the
               model writes the rewrite directly. Sampling T=0.7 top_p=0.8 top_k=20 (Qwen3 non-thinking recipe).
               --thinking on: the model thinks first; only the text after '</think>' is the rewrite (the thinking is
               stored separately). T=0.6 top_p=0.95 top_k=20. A sample that never closes '</think>' has no rewrite.
  qwen2.5-0.5b Qwen/Qwen2.5-0.5B-Instruct (no thinking mode). Its template injects 'You are Qwen, created by Alibaba
               Cloud...' when no system message is given, so an explicit system message is always passed.
               T=0.7 top_p=0.8 top_k=20 repetition_penalty=1.05 (its generation_config).
Both get the same system + user text and stop on <|im_end|> / <|endoftext|>.

  python scripts/rewrite_traces.py --model qwen3-0.6b   --first 256 --n 4
  python scripts/rewrite_traces.py --model qwen2.5-0.5b --first 256 --n 4
  python scripts/rewrite_traces.py --model qwen3-0.6b   --first 2 --dry_run   # print the rendered prompts only

Output (one row per problem, resumable): row (index into openr1.parquet), model, thinking, source, orig_tokens,
n_drawn, n_correct, samples = [{text, thinking, n_tokens, finish, correct}], best = shortest correct rewrite
('' if none), best_tokens.
"""
import argparse, os, re, sys

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, 'verl'))

MODELS = {
    'qwen3-0.6b': dict(path='Qwen/Qwen3-0.6B', family='qwen3'),
    'qwen2.5-0.5b': dict(path='Qwen/Qwen2.5-0.5B-Instruct', family='qwen2.5'),
}
SAMPLING = {  # model-card recommendations
    ('qwen3', 'off'): dict(temperature=0.7, top_p=0.8, top_k=20),
    ('qwen3', 'on'): dict(temperature=0.6, top_p=0.95, top_k=20),
    ('qwen2.5', 'off'): dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05),
}
SYSTEM = 'You are a careful math tutor. You rewrite solutions clearly and concisely, in your own words.'
USER = """Here is a math problem and a correct solution to it.

### Problem
{problem}

### Correct solution
{solution}

### Task
Rewrite the solution above in your own words. Keep every step that is needed to reach the answer, drop repetition,
false starts and self-checks, and do not introduce new methods or a different answer. Be concise.
End with the final answer in \\boxed{{}}."""

ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model', required=True, choices=list(MODELS))
ap.add_argument('--thinking', default='off', choices=['off', 'on'], help='Qwen3 only (Qwen2.5-Instruct has no thinking mode)')
ap.add_argument('--data', default=os.path.join(os.environ.get('R1X4_ROOT', '.'), 'data', 'src', 'openr1.parquet'))
ap.add_argument('--source', default='full', choices=['full', 'solution'],
                help="full: the whole R1 trace (thinking + solution); solution: only the part after '</think>'")
ap.add_argument('--first', type=int, default=256, help='first N rows of the parquet (0 = all)')
ap.add_argument('--n', type=int, default=4, help='samples per problem')
ap.add_argument('--max_tokens', type=int, default=4096, help='rewrite budget (thinking on: thinking + rewrite)')
ap.add_argument('--max_source_tokens', type=int, default=8192, help='R1 traces longer than this are skipped')
ap.add_argument('--gpu_mem', type=float, default=0.85)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--num_shards', type=int, default=1)
ap.add_argument('--chunk', type=int, default=256)
ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--out', default=None)
ap.add_argument('--dry_run', action='store_true')
a = ap.parse_args()
M = MODELS[a.model]
if M['family'] == 'qwen2.5' and a.thinking == 'on':
    sys.exit('Qwen2.5-0.5B-Instruct has no thinking mode: use --thinking off')


def problem_of(prompt):
    return next(m['content'] for m in reversed(prompt) if m['role'] == 'user')


def source_of(target, which):
    t = target[0]['content']
    if which == 'solution' and '</think>' in t:
        return t.split('</think>', 1)[1].strip()
    return t.replace('<think>', '', 1).replace('</think>', '\n', 1).strip()


def render(tok, problem, solution):
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER.format(problem=problem, solution=solution)}]
    kw = {'enable_thinking': a.thinking == 'on'} if M['family'] == 'qwen3' else {}
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)


def split_output(text):
    """(thinking, rewrite). Thinking on: rewrite = text after '</think>' ('' if never closed)."""
    if M['family'] == 'qwen3' and a.thinking == 'on':
        if '</think>' not in text:
            return text.strip(), ''
        th, rw = text.split('</think>', 1)
        return th.replace('<think>', '', 1).strip(), rw.strip()
    return '', re.sub(r'^\s*<think>\s*</think>\s*', '', text).strip()


def empty_row(r, n_drawn=0):
    return dict(row=r['row'], model=M['path'], thinking=a.thinking, source=a.source, orig_tokens=r['orig_tokens'],
                n_drawn=n_drawn, n_correct=0, samples=[], best='', best_tokens=-1)


def main():
    import pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M['path'])
    t = pq.read_table(a.data, columns=['prompt', 'target', 'reward_model'])
    n_rows = t.num_rows if a.first <= 0 else min(a.first, t.num_rows)
    rows = [dict(r, row=i) for i, r in enumerate(t.slice(0, n_rows).to_pylist())][a.shard::a.num_shards]
    if a.dry_run:
        for r in rows[:2]:
            print(f'===== row {r["row"]} ({a.model}, thinking {a.thinking}) =====')
            print(render(tok, problem_of(r['prompt']), source_of(r['target'], a.source)[:600] + ' [...]'))
        print('sampling:', SAMPLING[(M['family'], a.thinking)])
        return
    from vllm import LLM, SamplingParams
    from verl.r1x4.reward import grade_texts
    out = a.out or f'rw_{a.model}_think{a.thinking}_{a.source}.parquet'
    if a.num_shards > 1:
        out = out.replace('.parquet', f'_shard{a.shard:02d}of{a.num_shards:02d}.parquet')
    res = pq.read_table(out).to_pylist() if os.path.exists(out) else []
    done = {r['row'] for r in res}
    todo = [r for r in rows if r['row'] not in done]
    print(f'[rewrite] {a.model} thinking={a.thinking} source={a.source}: {len(todo)} to do, {len(done)} done -> {out}', flush=True)
    llm = LLM(M['path'], dtype='bfloat16', gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=a.seed + a.shard,
              max_model_len=a.max_source_tokens + 1536 + a.max_tokens)
    sp = SamplingParams(n=a.n, max_tokens=a.max_tokens, **SAMPLING[(M['family'], a.thinking)])
    for c in range(0, len(todo), a.chunk):
        batch, texts, keep = todo[c:c + a.chunk], [], []
        for r in batch:
            src = source_of(r['target'], a.source)
            r['orig_tokens'] = len(tok(r['target'][0]['content'], add_special_tokens=False).input_ids)
            if len(tok(src, add_special_tokens=False).input_ids) <= a.max_source_tokens:
                keep.append(r)
                texts.append(render(tok, problem_of(r['prompt']), src))
            else:
                res.append(empty_row(r))
        gens = llm.generate(texts, sp, use_tqdm=False)
        flat = []  # (problem k, sample)
        for k, g in enumerate(gens):
            for o in g.outputs:
                th, rw = split_output(o.text)
                flat.append((k, dict(text=rw, thinking=th, finish=o.finish_reason or '', correct=0,
                                     n_tokens=len(tok(rw, add_special_tokens=False).input_ids) if rw else 0)))
        ok = grade_texts([s['text'] for _, s in flat], [keep[k]['reward_model']['ground_truth'] for k, _ in flat], workers=16)
        per = {}
        for (k, s), v in zip(flat, ok):
            s['correct'] = int(v == 1 and s['finish'] == 'stop' and bool(s['text']))
            per.setdefault(k, []).append(s)
        for k, r in enumerate(keep):
            ss = per.get(k, [])
            good = sorted((s for s in ss if s['correct']), key=lambda s: s['n_tokens'])
            row = empty_row(r, a.n)
            row.update(n_correct=len(good), samples=ss, best=good[0]['text'] if good else '',
                       best_tokens=good[0]['n_tokens'] if good else -1)
            res.append(row)
        pq.write_table(pa.Table.from_pylist(res), out)
        print(f'  chunk {c // a.chunk}: {len(keep)} problems; {sum(x["best_tokens"] > 0 for x in res)}/{len(res)} '
              f'with a correct rewrite so far', flush=True)
    summarize(res)
    print('REWRITE_DONE', flush=True)
    os._exit(0)  # vLLM worker threads can keep the process alive


def summarize(res):
    import statistics as st
    tried = [r for r in res if r['n_drawn']]
    samples = [s for r in tried for s in r['samples']]
    got = [r for r in tried if r['best_tokens'] > 0]
    print(f'problems {len(tried)} | samples {len(samples)}: correct {sum(s["correct"] for s in samples)}, '
          f'finish=length {sum(s["finish"] == "length" for s in samples)}, no rewrite text {sum(not s["text"] for s in samples)}')
    print(f'problems with >=1 correct rewrite: {len(got)}/{len(tried)}')
    if got:
        ratio = [r['best_tokens'] / r['orig_tokens'] for r in got]
        print(f'tokens: R1 median {st.median(r["orig_tokens"] for r in got):.0f} -> shortest correct rewrite median '
              f'{st.median(r["best_tokens"] for r in got):.0f}; ratio median {st.median(ratio):.3f}, mean {st.mean(ratio):.3f}')


if __name__ == '__main__':
    main()

