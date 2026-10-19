# Variant IV = p<1% + IS + annealing (NOT BUILT YET: plan agreed in conversation 2026-10-18)

Where the pasted recipe conflicts with the user's earlier instructions, the earlier ones win:
- Group stays R1x4: 4 students + 4 teachers, exactly 4C/4W, plain GRPO over 8 rows (±0.5) while λ>0.
  Teacher-row advantages are multiplied by λ_t (λ0=1 -> identical to R1x4 at step 1).
- Prompts/step 96 (0.6B) / 40 (8B) / 48 (1.7B) / 40 (4B); 8,192-token responses; entropy 1e-3; p<1% drop kept.
- Gate defaults to `always` (all_fail would silently delete every wrong teacher); length norm off by default (flags exist).
- For t > T_a = round(frac·steps), with frac 0.2: no teacher rows, GRPO-4 on students (Dr.GRPO, no std); Adam state cleared once at T_a+1.
  The mini-batch keeps the 8-row size, so per-row weight is unchanged across T_a (loss_row_mean_fix).

## IS
- μ exists only for self-sampled traces. Add `scripts/gen_teacher_mu.py`:
  - vLLM, T=1, top_p=1, top_k=-1, n=8, logprobs=0, max_tokens=8191;
  - training prompt + chat template;
  - keep a trace only if it stopped, is graded correct by `verl.r1x4.reward.grade_texts`, and round-trips through the student tokenizer;
  - store token_ids + token_logps (N+1) in `mu_<teacher>.parquet` keyed by luffy_index.
- Only the first T_a·B prompts in visit order need μ.
- Proposed teachers:

  | student | teacher |
  |---|---|
  | Qwen3-0.6B, Qwen3-1.7B | Qwen3-4B |
  | Qwen3-4B, Qwen3-8B | Qwen3-32B |
  | Qwen2.5-Math-7B | Elliott/LUFFY-Qwen-Math-7B-Zero (same tokenizer) |

- Loss on teacher tokens with μ: `-A · sg(min(c̄, π/μ)) · log π` in fp32 (mix_core_alg token loss, new `teacher_logp` + mask).
  - Rows without μ (wrong teachers, prompts without a μ trace) keep the p/(p+0.1) shaping (R1 fallback).
- Metrics: λ, IS rho_mean / frac_trunc / token_ESS, student-only score/length/entropy/clip rate.
- Gradient diagnostic: cos and norm ratio of g_S vs g_T, on diag steps.
- run.py flags: `--is yes|no --anneal yes|no --anneal_frac --lambda0 --is_cbar --gate --len_ref --mu_pool --diag_every`.
- Recipe §4 CPU tests, then a 4-step smoke (frac 0.5) on MSI.
- Apply to both kits; re-zip.

## Built 2026-10-19: IS + annealing options (all combinable with --p, --ches, --ranking, any --data / --model)
| flag | default (= R1x4 behaviour) | recipe value |
|---|---|---|
| `--is yes` | no | truncated per-token IS, `-A·sg(min(c̄, π/μ))·log π`, on μ rows; other teacher rows keep p/(p+0.1) |
| `--anneal yes` | no | λ_t on teacher rows; λ = 0 → no teacher rows (GRPO-4), AdamW reset once |
| `--anneal_frac` | 0.2 | T_a = frac × steps |
| `--schedule` | cosine | cosine / linear / constant |
| `--hold_steps` | 0 | H steps at λ0 first |
| `--lambda0`, `--lambda_min` | 1, 0 | λ0 from the step-1 `diag/grad_ratio` |
| `--layout` | r1x4 (4 students + 4 teachers, exactly 4C/4W) | `append`: 4 students + `--k_teacher` (2) positive rows |
| `--baseline` | group (mean of all rows) | `student` (mean of the 4 students, Dr.GRPO) |
| `--neg_weight` | 1 | 0 (drop wrong teacher rows) |
| `--len_ref` | 0 (off) | L_ref / len per teacher row |
| `--gate` | always | all_fail |
| `--entropy`, `--entropy_student_only` | 0.001, no | 0, yes |
| `--is_cbar`, `--is_logratio_min` | 1, -20 | 1, -20 |
| `--diag_every`, `--no_reset_adam` | 10, reset on | |

The full recipe arm (A2):
`--p 1 --is yes --anneal yes --layout append --k_teacher 2 --baseline student --neg_weight 0 --len_ref 2048 --entropy 0 --entropy_student_only yes`.
Its LUFFY-shaping twin is the same line with `--is no`.
Not built from the long draft: fp16/GradScaler (bf16 here), the ρ+ damage probe, the overwrite-path variant (§4.8).
Tests: r1x4_tests/test_iv.py, test_loss.py, test_iv_fit.py (`python test_iv_fit.py append` runs the recipe layout with CHES + ranking).
