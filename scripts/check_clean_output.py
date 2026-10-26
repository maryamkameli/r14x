#!/usr/bin/env python3
"""Sanity-check a clean_and_score.py output against the trainer's own tokenization (R1x4Data.teacher_ids).

  python scripts/check_clean_output.py $R1X4_ROOT/data/clean/rw_pairs_qwen3-0.6b_clean_lp_first64.parquet
Checks per row: stored token ids == R1x4Data.teacher_ids(row, text) (same tokenizer/EOS/cut), one log-prob per token,
all log-probs <= 0 and finite, the EOS is the last token; no error-flag phrase and no CJK left; prints one example.
"""
import math, os, sys

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, 'verl'))
sys.path.insert(0, os.path.join(KIT, 'scripts'))
import pyarrow.parquet as pq  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trace_filters import error_flags, has_cjk  # noqa: E402
from verl.r1x4.trainer import R1x4Data  # noqa: E402

path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(sys.argv[2] if len(sys.argv) > 2 else 'Qwen/Qwen3-0.6B')
eos = tok.convert_tokens_to_ids('<|im_end|>')
data = R1x4Data(path, tok, 1536, 8192, eos)  # the trainer's loader (asserts visit order)
bad = {'ids_mismatch': 0, 'len_mismatch': 0, 'logp_invalid': 0, 'no_eos_end': 0, 'flag_left': 0, 'cjk_left': 0}
for i, r in enumerate(data.rows):
    for text, ids, lps in [(r['correct_text'], r['correct_token_ids'], r['correct_token_logps']),
                           (r['wrong_texts'][0], r['wrong_token_ids'][0], r['wrong_token_logps'][0])]:
        ref = data.teacher_ids(i, text).tolist()
        bad['ids_mismatch'] += ids != ref
        bad['len_mismatch'] += len(ids) != len(lps)
        bad['logp_invalid'] += any((not math.isfinite(x)) or x > 1e-6 for x in lps)
        bad['no_eos_end'] += ids[-1] != eos and len(ids) < 8192
        bad['flag_left'] += bool(error_flags(text))
        bad['cjk_left'] += has_cjk(text)
print(f'{path}: {len(data.rows)} rows, visits {data.rows[0]["visit"]}..{data.rows[-1]["visit"]}')
print('problems found:', bad, '-> OK' if not any(bad.values()) else '-> CHECK')
r = data.rows[0]
print('example luffy_index', r['luffy_index'], '| correct tokens', len(r['correct_token_ids']),
      'mean logp %.3f' % (sum(r['correct_token_logps']) / len(r['correct_token_logps'])),
      '| wrong tokens', len(r['wrong_token_ids'][0]), 'mean logp %.3f' % (sum(r['wrong_token_logps'][0]) / len(r['wrong_token_logps'][0])))
print('first 5 tokens / logps:', [tok.decode([t]) for t in r['correct_token_ids'][:5]], [round(x, 3) for x in r['correct_token_logps'][:5]])
