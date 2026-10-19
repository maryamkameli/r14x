#!/usr/bin/env python3
"""Sample correct teacher traces WITH their sampling log-probs mu (variant IV, truncated per-token IS).

vLLM at T=1, top_p=1, top_k=-1, no penalties, logprobs=0: the returned log-prob of each sampled token is exactly log mu.
Kept: finish_reason == 'stop', graded correct by the trainer's grader (verl.r1x4.reward), and the student tokenizer
re-encodes the text to the sampled ids (BPE can sample non-canonical splits). Prompts = the training prompts (dataset
chat + the student's chat template), in LUFFY visit order: only the first ~T_a * batch prompts of a run need mu.

  python scripts/gen_teacher_mu.py --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-1.7B --data core.parquet \
      --first 3200 --n 4 --out mu_qwen3-4b_core.parquet [--shard 0 --num_shards 1 --tp 1]
Output rows: luffy_index, teacher, n_drawn, n_correct, traces = [{token_ids: [...content, stop], token_logps: [... N+1]}]
"""
import argparse, os, sys
import pyarrow as pa, pyarrow.parquet as pq

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, 'verl'))
ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument('--teacher', required=True)
ap.add_argument('--student', required=True, help='tokenizer used by the trainer (round-trip check, chat template)')
ap.add_argument('--data', required=True, help='R1x4 parquet (rows in visit order)')
ap.add_argument('--first', type=int, default=3200, help='first N rows (= prompts of the teacher phase, with margin)')
ap.add_argument('--n', type=int, default=4)
ap.add_argument('--keep', type=int, default=4, help='correct traces kept per prompt')
ap.add_argument('--max_tokens', type=int, default=8191, help='content + stop token must fit max_response_length=8192')
ap.add_argument('--tp', type=int, default=1)
ap.add_argument('--gpu_mem', type=float, default=0.9)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--num_shards', type=int, default=1)
ap.add_argument('--chunk', type=int, default=256)
ap.add_argument('--out', required=True)
a = ap.parse_args()


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.r1x4.reward import grade_texts
    rows = pq.read_table(a.data, columns=['prompt', 'reward_model', 'luffy_index']).to_pylist()[:a.first][a.shard::a.num_shards]
    stok = AutoTokenizer.from_pretrained(a.student)
    llm = LLM(a.teacher, dtype='bfloat16', tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
              max_model_len=1024 + a.max_tokens + 64, enable_prefix_caching=True, seed=1234 + a.shard)
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=a.max_tokens, logprobs=0)
    out = a.out if a.num_shards == 1 else a.out.replace('.parquet', f'_shard{a.shard:02d}of{a.num_shards:02d}.parquet')
    done = {r['luffy_index'] for r in pq.read_table(out).to_pylist()} if os.path.exists(out) else set()
    res = pq.read_table(out).to_pylist() if done else []
    todo = [r for r in rows if r['luffy_index'] not in done]
    print(f'[mu] {a.teacher} shard {a.shard}: {len(todo)} prompts to do ({len(done)} done)', flush=True)
    for c in range(0, len(todo), a.chunk):
        batch = todo[c:c + a.chunk]
        texts = [stok.apply_chat_template(r['prompt'], add_generation_prompt=True, tokenize=False) for r in batch]
        gens = llm.generate(texts, sp, use_tqdm=False)
        cand = []  # (row k, ids, logps, text)
        for k, g in enumerate(gens):
            for o in g.outputs:
                if o.finish_reason != 'stop' or not o.token_ids:
                    continue
                ids = list(o.token_ids)
                lps = [float(d[t].logprob) for t, d in zip(ids, o.logprobs)]
                text = stok.decode(ids[:-1])
                if stok.encode(text, add_special_tokens=False) != ids[:-1]:
                    continue  # non-canonical BPE split: the student could never produce these ids from this text
                cand.append((k, ids, lps, text))
        ok = grade_texts([x[3] for x in cand], [batch[x[0]]['reward_model']['ground_truth'] for x in cand], workers=16)
        per = {}
        for (k, ids, lps, _), o in zip(cand, ok):
            if o == 1:
                per.setdefault(k, []).append({'token_ids': ids, 'token_logps': lps})
        for k, r in enumerate(batch):
            tr = per.get(k, [])
            res.append({'luffy_index': r['luffy_index'], 'teacher': a.teacher, 'n_drawn': a.n, 'n_correct': len(tr), 'traces': tr[:a.keep]})
        pq.write_table(pa.Table.from_pylist(res), out)
        print(f'  chunk {c // a.chunk}: {len(batch)} prompts, {sum(1 for k in range(len(batch)) if per.get(k))} with a correct mu-trace, '
              f'{len(cand)} stop+round-trip of {len(batch) * a.n}', flush=True)
    print('MU_DONE', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
