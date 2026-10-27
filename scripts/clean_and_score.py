#!/usr/bin/env python3
"""Clean rewrite pairs (cut error-flag sentences, drop CJK), re-grade, and add teacher-forced token log-probs.

Input: Ivan's pairs parquet (pairs_right_wrong_*.parquet: luffy_index, visit, data_source, prompt, ground_truth,
right_text, wrong_text, ...). For each problem and each of right_text / wrong_text:
  1. wrong_text only: cut every sentence with an error-flag phrase, thinking included (trace_filters.cut_error_flags;
     Ivan 2026-10-25). right_text keeps its self-corrections (Ivan 2026-10-27): nothing is cut from it
  2. reject the text if it still has a flag (wrong_text only, e.g. next to the \\boxed answer) or any CJK character
  3. re-grade with the trainer's grader: right must stay correct (1), wrong must stay wrong (0)
  4. teacher-force the final text under the TRAINING prompt with the generating model (--model) and store the log-prob
     of every response token. Tokenization = R1x4Data.teacher_ids (verl/r1x4/trainer.py): the stored text (an empty
     leading <think></think> block is removed, Ivan 2026-10-26; build_rewrite_pairs.py used to add one), minus
     '<think>\\n' when the prompt already ends with it, + EOS, cut at --max_response_length; the prompt is the dataset
     chat with the model's default chat template. The log-probs therefore line up 1:1 with the trainer's response tokens.
     Computed by vLLM prompt_logprobs (logprob of each given token; no sampling), bf16.
  5. wrong_text only: the same response tokens teacher-forced under the GENERATION prompt (Ivan 2026-10-27: negative
     teacher tokens unlikely under the teacher are dropped in training). The prompt is rebuilt exactly as
     rewrite_traces.py built it (problem + reference from --source, the task's instruction, --words limit, Qwen3 chat
     template with enable_thinking=False, i.e. ending in the empty think block). The log-prob is under the SAMPLING
     distribution of the generator (Ivan 2026-10-27; --gen_temperature/--gen_top_k/--gen_top_p = rewrite_traces.py's
     0.7 / 20 / 0.8), rebuilt like vLLM's sampler from the top-20 raw log-probs: logits / T, top-k, then top-p on the
     renormalised top-k, renormalised again; a token outside that support gets -inf.
     Check: every sampled token is inside the support, so on uncut wrong texts no token should be at -inf.
A problem is kept only if BOTH its right and its wrong text survive.

Output parquet (R1x4 row format of data/build_rewrite_pairs.py + log-probs), sorted by visit:
  data_source, prompt, reward_model, correct_text, wrong_texts=[w], wrong_meta, luffy_index, visit,
  correct_token_ids, correct_token_logps, wrong_token_ids, wrong_token_logps, wrong_teacher_logps   (lists of lists for wrong_*)
<out>_stats.json: counts per step.

  python scripts/clean_and_score.py --pairs pairs_right_wrong_qwen3-0.6b_26904.parquet --out rw_pairs_clean_lp.parquet
  python scripts/clean_and_score.py --pairs ... --first 64 --out /tmp/t.parquet      # quick check
"""
import argparse, collections, json, math, os, re, sys

# vLLM's engine process must be spawned, not forked: the grader's worker processes (fork) run in this process too
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, 'verl'))
sys.path.insert(0, os.path.join(KIT, 'scripts'))
from trace_filters import cut_error_flags, error_flags, has_cjk  # noqa: E402


def strip_empty_think(text):
    """Ivan 2026-10-26: drop a leading <think></think> block if it is empty (whitespace only); non-empty blocks are kept.
    Texts are stored WITHOUT the empty block that build_rewrite_pairs.py used to add."""
    m = re.match(r'\s*<think>(\s*)</think>\s*', text)
    return (text[m.end():], True) if m else (text, False)


ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--pairs', required=True)
ap.add_argument('--model', default='Qwen/Qwen3-0.6B', help='the model that generated the texts (log-probs under it)')
ap.add_argument('--first', type=int, default=0)
ap.add_argument('--max_response_length', type=int, default=8192)
ap.add_argument('--max_prompt_length', type=int, default=1536)
ap.add_argument('--no_logprobs', action='store_true')
ap.add_argument('--chunk', type=int, default=256, help='sequences per generate() call')
ap.add_argument('--gpu_mem', type=float, default=0.5, help='vLLM share (rest is needed for prompt_logprobs tensors)')
ap.add_argument('--step_tokens', type=int, default=2048, help='max tokens per engine step (bounds logprob memory)')
ap.add_argument('--source', default=None, help='extended.parquet the rewrites were made from (for wrong_teacher_logps)')
ap.add_argument('--words', type=int, default=2000, help="rewrite_traces.py --words used for the pairs")
ap.add_argument('--gen_temperature', type=float, default=0.7, help='sampling params of the rewrites (rewrite_traces.py)')
ap.add_argument('--gen_top_k', type=int, default=20)
ap.add_argument('--gen_top_p', type=float, default=0.8)
ap.add_argument('--teacher_chunk', type=int, default=32, help='sequences per generate() for the teacher log-probs '
                '(long prompts x top-20 log-prob objects use a lot of host memory)')
ap.add_argument('--out', required=True)
a = ap.parse_args()


def response_ids(tok, prompt_text, text, eos):
    """= R1x4Data.teacher_ids."""
    if prompt_text.endswith('<think>\n') and text.startswith('<think>\n'):
        text = text[len('<think>\n'):]
    return (tok(text, add_special_tokens=False)['input_ids'] + [eos])[:a.max_response_length]


# --- generation prompt of the wrong rewrites: copied from the kit's scripts/rewrite_traces.py (same text, same logic)
STYLE = 'Rewrite this solution in your own words, the way you would naturally solve the problem yourself'
GEN_TASK = {
    'wrong_ref': STYLE + '. Follow the reasoning of the reference solution and keep its final answer.',
    'wrong_cut': ('The reference solution is unfinished. Solve the problem in your own words, the way you would '
                  'naturally do it, using the start of the reference, but make one plausible mistake so that your '
                  'final answer is {target}. Do not mention the mistake and do not say that the answer is wrong.'),
    'wrong_free': STYLE + (', but make one plausible mistake along the way so that your final answer is NOT {gold}. '
                           'Do not mention the mistake and do not say that the answer is wrong.'),
}


def last_boxed(s):
    i = s.rfind('\\boxed{')
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += {'{': 1, '}': -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1].strip() if depth == 0 else None


def gen_message(src, task):
    """= rewrite_traces.message(r, task) for a row of the source parquet."""
    problem = next(m['content'] for m in reversed(src['prompt']) if m['role'] == 'user')
    gold, ref = src['reward_model']['ground_truth'], src['correct_text']
    target = next((b for b in map(last_boxed, src['wrong_texts']) if b and b != gold.strip()), None)
    if task == 'wrong_cut' and target is None:
        task = 'wrong_free'
    if task == 'wrong_ref':
        ref = next((w for w in src['wrong_texts'] if last_boxed(w) == target), src['wrong_texts'][0])
    if task == 'wrong_cut':
        ref = ref[:len(ref) // 2]
    return (f'Problem:\n{problem}\n\nReference solution (written by another model):\n{ref}\n\n---\n\n'
            + GEN_TASK[task].format(gold=gold, target=target)
            + f' Put your final answer within \\boxed{{}}. Your whole response must be at most {a.words} words.')


def clean(text, cut=True):
    """-> (cleaned text or None if rejected, reason, n_cut). Empty think block removed, no prefix added.
    cut=False (correct texts): no error-flag sentence is cut or rejected, only the CJK check applies."""
    text, _ = strip_empty_think(text)
    if not cut:
        return (None, 'cjk', 0) if has_cjk(text) else (text, 'ok', 0)
    new, n = cut_error_flags(text)
    if error_flags(new):
        return None, 'flag_kept', n
    if has_cjk(new):
        return None, 'cjk', n
    return new, 'ok', n


def main():
    import pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from verl.r1x4.reward import grade_texts
    # start vLLM BEFORE the grader forks its workers (same order as gen_negatives.py / rewrite_traces.py)
    llm = None
    if not a.no_logprobs:
        from vllm import LLM
        # prompt_logprobs materializes a full-vocab (~152k) float32 log-softmax per scheduled token: with the default
        # 8,192-token step that is ~5 GB + copies and OOMed a 44 GB L40S at gpu_mem 0.85 (job 4702867). Small steps and
        # a smaller KV-cache share leave room for it.
        llm = LLM(a.model, dtype='bfloat16', gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=False,
                  max_model_len=40960 if a.source else a.max_prompt_length + a.max_response_length + 8,  # Qwen3 max
                  max_num_batched_tokens=a.step_tokens, enable_chunked_prefill=True)
    t = pq.read_table(a.pairs).to_pylist()
    t = t[:a.first] if a.first > 0 else t
    c = collections.Counter(problems=len(t))
    cand = []  # (row, right_clean, wrong_clean, n_cut_right, n_cut_wrong)
    for r in t:
        rc, rwhy, rn = clean(r['right_text'], cut=False)
        wc, wwhy, wn = clean(r['wrong_text'])
        c[f'right_{rwhy}'] += 1
        c[f'wrong_{wwhy}'] += 1
        c['right_sentences_cut'] += rn
        c['wrong_sentences_cut'] += wn
        c['right_texts_with_cut'] += rn > 0
        c['wrong_texts_with_cut'] += wn > 0
        if rc is not None and wc is not None:
            cand.append((r, rc, wc, rn, wn))
    c['after_text_filters'] = len(cand)
    # re-grade (the cut must not change the label): right stays 1, wrong stays 0 (-1 ungraded = reject)
    texts = [x for _, rc, wc, _, _ in cand for x in (rc, wc)]
    golds = [r['ground_truth'] for r, *_ in cand for _ in (0, 1)]
    g = grade_texts(texts, golds, workers=16)
    keep = []
    for k, x in enumerate(cand):
        gr, gw = g[2 * k], g[2 * k + 1]
        c['regrade_right_not_correct'] += gr != 1
        c['regrade_wrong_not_wrong'] += gw != 0
        if gr == 1 and gw == 0:
            keep.append(x)
    c['after_regrade'] = len(keep)
    print(json.dumps(dict(c)), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.convert_tokens_to_ids('<|im_end|>')
    rows = []
    for r, rc, wc, rn, wn in keep:
        rows.append({'data_source': r['data_source'], 'prompt': r['prompt'],
                     'reward_model': {'ground_truth': r['ground_truth'], 'style': 'rule'},
                     'correct_text': rc, 'wrong_texts': [wc],
                     'wrong_meta': [json.dumps({'model': a.model, 'kind': 'rewrite_' + r.get('wrong_task', 'wrong'),
                                                'sentences_cut': wn})],
                     'luffy_index': r['luffy_index'], 'visit': r['visit'], 'correct_sentences_cut': rn,
                     '_wrong_task': r.get('wrong_task', 'wrong_ref')})
    if llm is not None:
        add_logprobs(llm, rows, tok, eos, c)
        if a.source:
            add_teacher_logprobs(llm, rows, tok, c)
    for r in rows:
        r.pop('_wrong_task')
    rows.sort(key=lambda r: r['visit'])
    pq.write_table(pa.Table.from_pylist(rows), a.out, compression='zstd', row_group_size=1000)
    c['written'] = len(rows)
    json.dump(dict(c), open(a.out.replace('.parquet', '') + '_stats.json', 'w'), indent=1)
    print(json.dumps(dict(c)), flush=True)
    print('CLEAN_DONE', flush=True)
    os._exit(0)


def add_logprobs(llm, rows, tok, eos, c):
    """Teacher-forced log-prob of every response token under the training prompt (vLLM prompt_logprobs)."""
    import statistics as st
    from vllm import SamplingParams, TokensPrompt
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)
    jobs = []  # (row, field, prompt_ids, resp_ids)
    for r in rows:
        ptext = tok.apply_chat_template(r['prompt'], add_generation_prompt=True, tokenize=False)
        pids = tok(ptext, add_special_tokens=False)['input_ids']
        jobs.append((r, 'correct', pids, response_ids(tok, ptext, r['correct_text'], eos)))
        jobs.append((r, 'wrong', pids, response_ids(tok, ptext, r['wrong_texts'][0], eos)))
    for s in range(0, len(jobs), a.chunk):
        part = jobs[s:s + a.chunk]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p + q) for _, _, p, q in part], sp, use_tqdm=False)
        for (r, f, p, q), o in zip(part, outs):
            plp = o.prompt_logprobs  # list aligned with prompt tokens; entry k = {token_id: Logprob} for token k
            lps = [float(plp[len(p) + j][tid].logprob) for j, tid in enumerate(q)]
            if f == 'correct':
                r['correct_token_ids'], r['correct_token_logps'] = q, lps
            else:
                r['wrong_token_ids'], r['wrong_token_logps'] = [q], [lps]
        print(f'  logprobs {min(s + a.chunk, len(jobs))}/{len(jobs)}', flush=True)
    for f in ('correct', 'wrong'):
        m = [st.mean(r[f'{f}_token_logps'] if f == 'correct' else r[f'{f}_token_logps'][0]) for r in rows]
        c[f'{f}_mean_token_logp_median'] = round(st.median(m), 4) if m else None
    c['responses_cut_at_max_len'] = sum(len(r['correct_token_ids']) >= a.max_response_length for r in rows)


def sampling_logprob(top, y):
    """log-prob of token y under vLLM's sampler (temperature -> top-k -> top-p on the renormalised top-k), from the raw
    log-probs `top` {token: logprob} of the k most likely tokens (y may be missing = outside top-k). -inf = outside."""
    if y not in top:
        return -math.inf
    s = sorted(((lp / a.gen_temperature, t) for t, lp in top.items()), reverse=True)[:a.gen_top_k]
    m = s[0][0]
    w = [math.exp(x - m) for x, _ in s]
    z = sum(w)
    q = [x / z for x in w]
    # vLLM: ascending cumsum; drop a token when the mass of it and all less likely ones is <= 1 - top_p (keep the top one)
    keep, tail = [], 0.0
    for qi in reversed(q):
        tail += qi
        keep.append(tail > 1 - a.gen_top_p)
    keep = keep[::-1]
    keep[0] = True
    kept = {t: qi for (_, t), qi, k in zip(s, q, keep) if k}
    return math.log(kept[y] / sum(kept.values())) if y in kept else -math.inf


def add_teacher_logprobs(llm, rows, tok, c):
    """wrong_teacher_logps: log-prob of each stored wrong response token (wrong_token_ids, the trainer's tokens) under
    the generator's sampling distribution given the generation prompt (sampling_logprob). Stats: share of tokens below
    1 % and outside the support (-inf), and the raw (T=1) share below 1 % for comparison."""
    import pyarrow.parquet as pq
    from vllm import SamplingParams, TokensPrompt
    need = {r['luffy_index'] for r in rows}
    src = {}
    pf = pq.ParquetFile(a.source)
    for g in range(pf.metadata.num_row_groups):
        for s in pf.read_row_group(g, columns=['luffy_index', 'prompt', 'reward_model', 'correct_text', 'wrong_texts']).to_pylist():
            if s['luffy_index'] in need:
                src[s['luffy_index']] = s
    assert len(src) == len(need), f'{len(need) - len(src)} problems missing from --source'
    sp = SamplingParams(max_tokens=1, prompt_logprobs=a.gen_top_k, temperature=1.0)
    jobs = []
    for r in rows:
        ptext = tok.apply_chat_template([{'role': 'user', 'content': gen_message(src[r['luffy_index']], r['_wrong_task'])}],
                                        add_generation_prompt=True, tokenize=False, enable_thinking=False)
        jobs.append((r, tok(ptext, add_special_tokens=False)['input_ids'], r['wrong_token_ids'][0]))
    n = low = out = raw_low = n_uncut = out_uncut = 0
    for s in range(0, len(jobs), a.teacher_chunk):
        part = jobs[s:s + a.teacher_chunk]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p + q) for _, p, q in part], sp, use_tqdm=False)
        for (r, p, q), o in zip(part, outs):
            lps = []
            for j, y in enumerate(q):
                ent = o.prompt_logprobs[len(p) + j]
                top = {t: float(e.logprob) for t, e in ent.items() if e.rank <= a.gen_top_k}
                lps.append(sampling_logprob(top, y))
                raw_low += ent[y].logprob < math.log(0.01)
            r['wrong_teacher_logps'] = [lps]
            o_n = sum(x == -math.inf for x in lps)
            n += len(lps)
            low += sum(x < math.log(0.01) for x in lps)
            out += o_n
            if json.loads(r['wrong_meta'][0])['sentences_cut'] == 0:
                n_uncut += len(lps)
                out_uncut += o_n
        del outs
        print(f'  teacher logprobs {min(s + a.teacher_chunk, len(jobs))}/{len(jobs)}', flush=True)
    c['wrong_teacher_tokens'] = n
    c['wrong_teacher_p_below_1pct_frac'] = round(low / max(n, 1), 6)
    c['wrong_teacher_outside_support_frac'] = round(out / max(n, 1), 6)
    c['wrong_teacher_outside_support_frac_uncut_texts'] = round(out_uncut / max(n_uncut, 1), 6)
    c['wrong_teacher_raw_T1_p_below_1pct_frac'] = round(raw_low / max(n, 1), 6)


if __name__ == '__main__':
    main()
