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

LENGTH CAP (rewrite <= the median length of the SAME model's naturally generated correct answers on this dataset).
Phase 1 ("natural"): the model solves problems by itself, prompted exactly as the student in training: dataset prompt
(LUFFY system prompt + problem), its default chat template (Qwen3: thinking on; --nat_thinking off to change), training
rollout sampling (T=1, top_p=1, top_k=-1; --nat_sampling card for model-card values), max 8192 tokens, graded.
  --cap_mode global (default): random problems from the whole dataset (seeded), --nat_n answers each, until
      --nat_target (50) correct answers exist; cap = their median token length, one cap for the model.
  --cap_mode per_problem: every rewritten problem gets its own median (>= --nat_min_correct correct answers), else the
      median of similar problems (same data_source + R1-length quintile -> data_source -> all).
Phase 2 rewrites with max_tokens = cap (longer outputs are cut and rejected) and states the budget in the prompt; a
rewrite is kept only if correct AND n_tokens <= cap.

Output (one row per problem, resumable): row (index into openr1.parquet), model, thinking, source, orig_tokens,
nat_correct_tokens (lengths of natural correct answers), nat_n, cap, cap_source, n_drawn, n_correct (kept rewrites),
samples = [{text, thinking, n_tokens, finish, correct, within_cap}], best (--pick: longest kept rewrite by default, the
one closest to the model's natural length; 'shortest' tends to pick bare answers), best_tokens.
Natural answers go to <out>_natural.parquet (resumable, reused by later rewrite runs of the same model).
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
false starts and self-checks, and do not introduce new methods or a different answer. Be concise: your rewrite must be
shorter than {words} words.
End with the final answer in \\boxed{{}}."""
TRAIN_SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=-1)  # = actor_rollout_ref.rollout in run.py

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
ap.add_argument('--nat_n', type=int, default=4, help='natural (unaided) answers per problem for the length cap')
ap.add_argument('--nat_target', type=int, default=50, help='stop the natural pass once this many correct answers exist')
ap.add_argument('--nat_problems', type=int, default=64, help='random problems (seeded, whole dataset) per natural chunk')
ap.add_argument('--cap_mode', default='global', choices=['global', 'per_problem'],
                help='global: one cap = median of nat_target correct natural answers; per_problem: own median w/ fallbacks')
ap.add_argument('--nat_max_tokens', type=int, default=8192, help='= max_response_length of training')
ap.add_argument('--nat_thinking', default='default', choices=['default', 'on', 'off'],
                help="Qwen3 natural answers: 'default' = the template default used in training (thinking on)")
ap.add_argument('--nat_sampling', default='train', choices=['train', 'card'])
ap.add_argument('--nat_min_correct', type=int, default=2, help='own correct answers needed for a per-problem cap')
ap.add_argument('--pick', default='longest', choices=['longest', 'shortest'], help='which kept rewrite becomes best')
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


def render_natural(tok, prompt):
    """The training prompt: dataset chat (LUFFY system + problem) with the model's own template."""
    kw = {} if M['family'] != 'qwen3' or a.nat_thinking == 'default' else {'enable_thinking': a.nat_thinking == 'on'}
    return tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False, **kw)


def render(tok, problem, solution, cap=None):
    words = max(10, int(0.6 * cap)) if cap else 100000  # ~0.75 words/token for English+LaTeX, with margin
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': USER.format(problem=problem, solution=solution, words=words)}]
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




def ntok(tok, text):
    return len(tok(text, add_special_tokens=False).input_ids) if text else 0


def natural_pass(llm, tok, rows, path, pa, pq, grade_texts, SamplingParams):
    """Phase 1: the model answers each problem unaided (training prompt + sampling). Resumable parquet:
    row, data_source, nat_n, n_correct, correct_tokens, all_tokens, finish_length."""
    res = pq.read_table(path).to_pylist() if os.path.exists(path) else []
    done = {r['row'] for r in res}
    todo = [r for r in rows if r['row'] not in done]
    samp = TRAIN_SAMPLING if a.nat_sampling == 'train' else SAMPLING[(M['family'], 'on' if M['family'] == 'qwen3' else 'off')]
    sp = SamplingParams(n=a.nat_n, max_tokens=a.nat_max_tokens, **samp)
    print(f'[natural] {a.model} {len(todo)} to do, {len(done)} done, n={a.nat_n}, sampling {samp} -> {path}', flush=True)
    step = a.nat_problems if a.cap_mode == 'global' else a.chunk
    for c in range(0, len(todo), step):
        batch = todo[c:c + step]
        gens = llm.generate([render_natural(tok, r['prompt']) for r in batch], sp, use_tqdm=False)
        flat = [(k, o) for k, g in enumerate(gens) for o in g.outputs]
        ok = grade_texts([o.text for _, o in flat], [batch[k]['reward_model']['ground_truth'] for k, _ in flat], workers=16)
        per = {}
        for (k, o), v in zip(flat, ok):
            per.setdefault(k, []).append((len(o.token_ids), o.finish_reason == 'stop' and v == 1, o.finish_reason == 'length'))
        for k, r in enumerate(batch):
            s = per.get(k, [])
            res.append(dict(row=r['row'], data_source=r['data_source'], nat_n=a.nat_n, n_correct=sum(x[1] for x in s),
                            correct_tokens=[x[0] for x in s if x[1]], all_tokens=[x[0] for x in s],
                            finish_length=sum(x[2] for x in s)))
        pq.write_table(pa.Table.from_pylist(res), path)
        n_ok = sum(len(x['correct_tokens']) for x in res)
        print(f'  natural chunk {c // step}: {len(batch)} problems, {n_ok} correct answers so far', flush=True)
        if a.cap_mode == 'global' and n_ok >= a.nat_target:
            break
    return {r['row']: r for r in res}


def global_cap(nat):
    """ONE cap for the model on this dataset = median length of nat_target correct natural answers (Ivan, 2026-10-22)."""
    import statistics as st
    lens = [x for r in nat.values() for x in r['correct_tokens']]
    if len(lens) < a.nat_target:
        print(f'[cap] WARNING: only {len(lens)} correct natural answers (< {a.nat_target})', flush=True)
    s = sorted(lens)
    print(f'[cap] {a.model}: {len(lens)} correct natural answers from {len(nat)} problems; tokens median {st.median(s):.0f}, '
          f'mean {st.mean(s):.0f}, p25 {s[len(s) // 4]}, p75 {s[3 * len(s) // 4]}', flush=True)
    return int(st.median(s))


def caps_for(rows, nat):
    """row -> (cap, cap_source). Own median if >= nat_min_correct correct; else similar problems (same data_source and
    R1-length quintile -> same data_source -> all)."""
    import statistics as st
    lens = sorted(r['orig_tokens'] for r in rows)
    edges = [lens[int(q * (len(lens) - 1))] for q in (.2, .4, .6, .8)]
    quint = lambda x: sum(x > e for e in edges)
    pools = {}
    for r in rows:
        ct = nat[r['row']]['correct_tokens'] if r['row'] in nat else []
        for key in ((r['data_source'], quint(r['orig_tokens'])), (r['data_source'],), ()):
            pools.setdefault(key, []).extend(ct)
    out = {}
    for r in rows:
        ct = nat[r['row']]['correct_tokens'] if r['row'] in nat else []
        if len(ct) >= a.nat_min_correct:
            out[r['row']] = (int(st.median(ct)), 'own')
            continue
        for key, name in (((r['data_source'], quint(r['orig_tokens'])), 'source+length'), ((r['data_source'],), 'source'),
                          ((), 'all')):
            if len(pools.get(key, [])) >= 5:
                out[r['row']] = (int(st.median(pools[key])), name)
                break
        else:
            out[r['row']] = (a.max_tokens, 'none')
    return out


def empty_row(r, cap=-1, cap_src='', nat=None, n_drawn=0):
    return dict(row=r['row'], model=M['path'], thinking=a.thinking, source=a.source, orig_tokens=r['orig_tokens'],
                nat_correct_tokens=(nat or {}).get('correct_tokens', []), nat_n=(nat or {}).get('nat_n', 0), cap=cap,
                cap_source=cap_src, n_drawn=n_drawn, n_correct=0, samples=[], best='', best_tokens=-1)



def main():
    import pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M['path'])
    t = pq.read_table(a.data, columns=['data_source', 'prompt', 'target', 'reward_model'])
    n_rows = t.num_rows if a.first <= 0 else min(a.first, t.num_rows)
    rows = [dict(r, row=i) for i, r in enumerate(t.slice(0, n_rows).to_pylist())][a.shard::a.num_shards]
    if a.dry_run:
        r = rows[0]
        print(f'===== NATURAL prompt, row {r["row"]} ({a.model}) =====\n{render_natural(tok, r["prompt"])}')
        print(f'===== REWRITE prompt, row {r["row"]} ({a.model}, thinking {a.thinking}, cap 400) =====')
        print(render(tok, problem_of(r['prompt']), source_of(r['target'], a.source)[:400] + ' [...]', cap=400))
        print('rewrite sampling:', SAMPLING[(M['family'], a.thinking)], '| natural sampling:',
              TRAIN_SAMPLING if a.nat_sampling == 'train' else 'card')
        return
    from vllm import LLM, SamplingParams
    from verl.r1x4.reward import grade_texts
    out = a.out or f'rw_{a.model}_think{a.thinking}_{a.source}.parquet'
    if a.num_shards > 1:
        out = out.replace('.parquet', f'_shard{a.shard:02d}of{a.num_shards:02d}.parquet')
    nat_path = out.replace('.parquet', '_natural.parquet')
    for r in rows:
        r['orig_tokens'] = ntok(tok, r['target'][0]['content'])
    llm = LLM(M['path'], dtype='bfloat16', gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=a.seed + a.shard,
              max_model_len=max(a.max_source_tokens + 1536 + a.max_tokens, 2048 + a.nat_max_tokens))
    if a.cap_mode == 'global':
        # natural answers on random problems from the WHOLE dataset (not only the rewritten ones), seeded
        import random
        allrows = [dict(r, row=i) for i, r in enumerate(t.to_pylist())]
        random.Random(a.seed).shuffle(allrows)
        nat = natural_pass(llm, tok, allrows, nat_path, pa, pq, grade_texts, SamplingParams)
        cap = global_cap(nat)
        caps = {r['row']: (cap, f'global median of {sum(len(x["correct_tokens"]) for x in nat.values())}') for r in rows}
    else:
        nat = natural_pass(llm, tok, rows, nat_path, pa, pq, grade_texts, SamplingParams)
        caps = caps_for(rows, nat)

    res = pq.read_table(out).to_pylist() if os.path.exists(out) else []
    done = {r['row'] for r in res}
    todo = [r for r in rows if r['row'] not in done]
    print(f'[rewrite] {a.model} thinking={a.thinking} source={a.source}: {len(todo)} to do, {len(done)} done -> {out}', flush=True)
    samp = SAMPLING[(M['family'], a.thinking)]
    for c in range(0, len(todo), a.chunk):
        batch, texts, sps, keep = todo[c:c + a.chunk], [], [], []
        for r in batch:
            cap, cap_src = caps[r['row']]
            src = source_of(r['target'], a.source)
            if ntok(tok, src) > a.max_source_tokens:
                res.append(empty_row(r, cap, cap_src, nat.get(r['row'])))
                continue
            keep.append(r)
            texts.append(render(tok, problem_of(r['prompt']), src, cap))
            # thinking on: the cap applies to the rewrite (after </think>); allow room for the thinking itself
            budget = cap + 1 if a.thinking == 'off' else min(a.max_tokens, cap + 4096)
            sps.append(SamplingParams(n=a.n, max_tokens=budget, **samp))
        gens = llm.generate(texts, sps, use_tqdm=False)
        flat = []
        for k, g in enumerate(gens):
            cap = caps[keep[k]['row']][0]
            for o in g.outputs:
                th, rw = split_output(o.text)
                n = ntok(tok, rw)
                flat.append((k, dict(text=rw, thinking=th, finish=o.finish_reason or '', correct=0, n_tokens=n,
                                     within_cap=int(bool(rw) and n <= cap))))
        ok = grade_texts([s['text'] for _, s in flat], [keep[k]['reward_model']['ground_truth'] for k, _ in flat], workers=16)
        per = {}
        for (k, s), v in zip(flat, ok):
            s['correct'] = int(v == 1 and s['finish'] == 'stop' and bool(s['text']))
            per.setdefault(k, []).append(s)
        for k, r in enumerate(keep):
            ss = per.get(k, [])
            good = sorted((s for s in ss if s['correct'] and s['within_cap']), key=lambda s: s['n_tokens'],
                          reverse=a.pick == 'longest')
            row = empty_row(r, *caps[r['row']], nat.get(r['row']), a.n)
            row.update(n_correct=len(good), samples=ss, best=good[0]['text'] if good else '',
                       best_tokens=good[0]['n_tokens'] if good else -1)
            res.append(row)
        pq.write_table(pa.Table.from_pylist(res), out)
        print(f'  chunk {c // a.chunk}: {len(keep)} problems; {sum(x["best_tokens"] > 0 for x in res)}/{len(res)} '
              f'with a kept rewrite so far', flush=True)
    summarize(res, nat)
    print('REWRITE_DONE', flush=True)
    os._exit(0)  # vLLM worker threads can keep the process alive


def summarize(res, nat):
    import statistics as st, collections
    nn = list(nat.values())
    print(f'natural: {len(nn)} problems x {a.nat_n}: correct {sum(r["n_correct"] for r in nn)}, '
          f'hit max tokens {sum(r["finish_length"] for r in nn)}; problems with >= {a.nat_min_correct} correct '
          f'{sum(r["n_correct"] >= a.nat_min_correct for r in nn)}; median correct length '
          f'{st.median([x for r in nn for x in r["correct_tokens"]] or [0]):.0f}')
    tried = [r for r in res if r['n_drawn']]
    sm = [s for r in tried for s in r['samples']]
    got = [r for r in tried if r['best_tokens'] > 0]
    print('cap source:', dict(collections.Counter(r['cap_source'] for r in tried)),
          f'| cap median {st.median([r["cap"] for r in tried] or [0]):.0f}')
    print(f'rewrite samples {len(sm)}: correct {sum(s["correct"] for s in sm)}, within cap {sum(s["within_cap"] for s in sm)}, '
          f'correct AND within cap {sum(s["correct"] and s["within_cap"] for s in sm)}, cut at cap {sum(s["finish"] == "length" for s in sm)}')
    print(f'problems with a kept rewrite: {len(got)}/{len(tried)}')
    if got:
        print(f'tokens: R1 median {st.median(r["orig_tokens"] for r in got):.0f} | cap median {st.median(r["cap"] for r in got):.0f} '
              f'| kept rewrite ({a.pick}) median {st.median(r["best_tokens"] for r in got):.0f}; '
              f'max best/cap {max(r["best_tokens"] / r["cap"] for r in got):.3f}')


if __name__ == '__main__':
    main()

