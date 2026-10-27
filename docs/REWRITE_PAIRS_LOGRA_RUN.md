# Rewrite-pairs run with gold ranking, RL-PLUS, MC skip and LoGRA (Qwen3-0.6B)

## Data

`data/build_rewrite_pairs.py` turns `rewrite_qwen3-0.6b_extended_first30k.parquet` (output of `scripts/rewrite_traces.py`:
stock `Qwen/Qwen3-0.6B`, non-thinking mode, T=0.7, top_p=0.8, top_k=20, max 4096 tokens) into
`rewrite_pairs_qwen3-0.6b.parquet`.

| | |
|---|---|
| problems | 26,904 (of the first 30,000 LUFFY visits of `extended`) with a valid correct and a valid wrong rewrite |
| `correct_text` | valid correct rewrite |
| `wrong_texts=[...]` | valid wrong rewrite |
| wrong_ref | 26,381: rewrite of a stored wrong trace, keeping its wrong answer |
| wrong_cut | 523: built from the first half of the R1 trace |

Both texts are stored with the empty think block `'<think>\n\n</think>\n\n'` that Qwen3's non-thinking template put before
the generated text.

## Features

| run.py flags | Hydra keys | behaviour |
|---|---|---|
| `--p 1 --p_rows neg` | `actor.teacher_token_min_prob=0.01`, `actor.teacher_token_min_prob_rows=neg` | per token: wrong-teacher tokens with pre-update student probability < 1 % lose their gradient (dropped, not renormalised) |
| `--ches yes` | `r1x4.ches.*` | per-token CHES (Razin et al.) on every negative row (wrong teacher rows and negative-advantage students): the 50 % of response tokens whose final hidden state is most cosine-aligned with S⁺ = Σ of the group's positive-trace hidden states are dropped; the prefix shared with the positive is never dropped; ≥ 1 token kept |
| `--ranking yes --rank_score gold --rank_rolling yes --rank_pool 1000 --rank_top 100 --batch 100` | `r1x4.rank.*` | rolling pool of 1,000 prompts topped up in LUFFY visit order. Every step all pooled prompts are teacher-forced and scored by AMPO-style gold-answer probability: correct trace up to its last `\boxed{`, then the ground-truth answer tokens and `}`; score = mean over gold tokens of the student's token probability. Top 100 are trained (descending); the other 900 stay and age (`pool/*`) |
| `--rlplus yes --rlplus_warmup 10 --rlplus_cbar 1.0` | `r1x4.rlplus.*`, `actor.is_cbar` | teacher tokens use the existing truncated-IS loss −A · sg(min(c̄, π_θ/μ̂)) · log π_θ. Steps 1–10: μ̂ = π_old (ratio 1, no IS). From step 11: RL-PLUS multiple importance sampling (arXiv 2508.00222 eq. 4, Bayes estimate of Thm 3.2, π_ω = (π_old + U)/2, U = 1/V): r = 2π_θ/(π_ω + π_old) = π_θ/(0.75 π_old + 0.25/V), clipped at c̄ = 1. Bounded (≤ 4/3 at the update point), small for surprise tokens (π_old ≪ 1/V). Smoke test: 99.9 % of teacher tokens hit the clip (weight 1), the rest < 1 |
| `--mc_skip yes --mc_samples 4 --mc_z 1.0` | `r1x4.mc_skip.*` | prompts whose 4 students are all correct get 4 extra Monte Carlo rollouts (graded only, never trained); p̂ over the 8 samples; the whole group (teacher and student rows) is skipped iff 1−p̂ ≤ z·√(p̂(1−p̂)/8), i.e. with z = 1 only 8/8. Cost: one extra generate call per step on the candidates only (smoke, 16 prompts/step: 28–74 s; train logs show ~21–28 % of prompts all-correct) |
| `--logra yes` (optional, off by default; + `--logra_rank 256 --logra_refresh yes --logra_optimizer rowadam --logra_kl_start 2e-4 --logra_kl_end 2e-5`) | `r1x4.logra.*`; forces actor FSDP `use_orig_params=True` and `rollout.calculate_log_probs=True` | LoGRA (arXiv 2610.06647) replaces `AdamW.step()`. For each attention/MLP projection (196 matrices in Qwen3-0.6B): S = G Aᵀ with a fresh seeded Rademacher A (r = 256, entries ±1/√r, identical on all ranks); RowAdam (per-row second moment, β₂ = 0.95, no first moment, rescaled to ‖S‖_F); direction D = −η U A, η from the reference `target` mode (‖D‖_F = 6e-5 ‖W_targets‖_F). Predicted-KL step control via finite-difference probe (2 rows × 256 positions per rank, KL(π_before ‖ π_after) on full-vocab logits from final hidden state × tied lm_head in fp32), α = min(3, √(δ_eff/q)). KL budget cosine-annealed 2e-4 → 2e-5 over the run; mismatch subtraction δ_eff = max(δ − κ, 0.5δ), κ = EMA(0.3) of max(mean(vLLM logp − trainer logp), 0) over student tokens. Embeddings, norms, lm_head frozen. State saved as `logra_state_rank{r}.pt` next to the actor checkpoint and restored on resume. Smoke: q = 1.1e-3–1.5e-3 at α = 1, α 0.09–0.26, 1.4 s/step |
| `--muon yes` (+ `--muon_momentum 0.95 --muon_nesterov yes --muon_ns_steps 5 --muon_scale moonlight`) | `r1x4.muon.*`; forces actor FSDP `use_orig_params=True` | Muon (Keller Jordan) on the 196 attention/MLP projection matrices: M ← 0.95 M + G; U = G + 0.95 M (Nesterov); O = Newton-Schulz-5(U) (quintic coefficients 3.4445, −4.7750, 2.0315, fp32); W ← W(1 − lr·wd) − lr · 0.2·√max(d, k) · O (Moonlight RMS matching: reuses `--lr` 1e-6 and weight decay 0.01 of AdamW). Embeddings, tied lm_head and norms keep AdamW. Momentum buffers (fp32, identical on all ranks) are saved by rank 0 as `muon_state.pt` in the actor checkpoint and restored on every rank. Mutually exclusive with `--logra`; with neither flag the optimizer is plain AdamW |
| (no `--anneal`), `--entropy 0` | | teacher weight on for the whole run; no entropy bonus |

## RL-PLUS weight with the clip

At the update point π_θ = π_old, so the ratio is r(π_old) = 1 / (0.75 + 0.25/(V π_old)) with V = 151,936 (1/V = 6.6e-6):
r ≥ 1 iff π_old ≥ 1/V, r → 4/3 as π_old → 1, r ≈ 4 V π_old below 1/V.
With c̄ = 1 the weight is min(1, r) (1 for 99.86 % of teacher tokens in the smoke runs); with c̄ ≥ 4/3 it is r itself.

| π_old | r | weight c̄ = 1 | weight c̄ = 4/3 | LUFFY 0.1p/(p+0.1)² (earlier R1x4 runs) |
|---|---|---|---|---|
| 1e-7 | 0.058 | 0.058 | 0.058 | 0.0000 |
| 1e-6 | 0.418 | 0.418 | 0.418 | 0.0000 |
| 6.6e-6 | 1.000 | 1.000 | 1.000 | 0.0001 |
| 1e-4 | 1.305 | 1.000 | 1.305 | 0.0010 |
| 1e-2 | 1.333 | 1.000 | 1.333 | 0.0826 |
| 0.1 | 1.333 | 1.000 | 1.333 | 0.2500 |
| 0.9 | 1.333 | 1.000 | 1.333 | 0.0900 |

## Launch

4 GPUs, 260 steps: the pool consumes 100 new prompts per step, so 1,000 + 100·(260−1) ≈ 26,900 covers the 26,904 prompts
once without an epoch wrap. Other hardware (4×A40, 8×A40, 4×A100 on MSI, resumable Slurm chains, `--hw` memory
profiles): `scripts/slurm/README.md`.

```bash
cd /mnt/data/radke149/r1x4kit_1p7b_4b && source scripts/rtx_env.sh && export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/run.py --model qwen3-0.6b --data $R1X4_ROOT/data/rewrite_pairs_qwen3-0.6b.parquet \
  --p 1 --p_rows neg --ches yes --ranking yes --rank_score gold --rank_rolling yes --rank_pool 1000 --rank_top 100 \
  --batch 100 --rlplus yes --rlplus_warmup 10 --rlplus_cbar 1.0 --mc_skip yes --mc_samples 4 --mc_z 1.0 \
  --muon yes --entropy 0 --steps 260 --gpus 4 --test_freq 20 --save_freq 5 --root $R1X4_ROOT
```

Export `R1X4_DEADLINE=<unix seconds>` to make the trainer save a checkpoint and stop cleanly before a reservation/walltime
ends; restarting the same command (default `--resume auto`) continues from the latest checkpoint, including the
rolling-pool state and the Muon/LoGRA state.

Experiment name: `R1X4-REWRITE_PAIRS_QWEN3-0.6B-QWEN3-0.6B-P1PCT-TOKCHES-GOLDROLLRANK100OF1000-ENT0-RLPLUS10-MCSKIP4-MUON`.

## Verification

2026-10-23, 4× RTX PRO 6000, 16 prompts/step, pool 64 → top 16:

- SMOKE-A: all features except LoGRA, 3 steps (rolling pool top 16 of 64: gold p 1.000 chosen vs 0.816 pool; MC skipped
  1 / 3 / 0 prompts; RL-PLUS weight 1 at step 1, active from step 2 with mean weight 0.9994; p-drop ~1 % of teacher
  tokens, CHES 50 % of negative-row tokens).
- SMOKE-B: all features + LoGRA, 2 steps, checkpoint saved at step 2 (`logra_state_rank0-3.pt` written), then a resume
  run to step 3: pool rounds and the LoGRA mismatch EMA continue from the checkpoint (6.53e-4 = 0.7·6.74e-4 + 0.3·6.04e-4).
- SMOKE-C (final code): all features + LoGRA, 2 steps: 196 target matrices, q_pred_kl 1.5e-3 / 1.4e-3 at α=1, α 0.26 / 0.09.
- SMOKE-D (final code): same without LoGRA, 1 step: AdamW path (actor/lr 1e-6), no logra/* metrics.
- SMOKE-E (Muon, all features): 2 steps + checkpoint + resume to step 3: 196 Muon matrices, update RMS 1.8e-7 (= 0.18 lr),
  relative update 6.4e-6 per step, Muon step 0.7-1.1 s; `muon_state.pt` (1.76 GB) written by rank 0 at step 2 and restored
  on all 4 ranks (momentum norm 0.079 -> 0.143 -> 0.261 across the resume); AdamW state only for embeddings/norms (311 MB/rank).
- Label-mismatch group drops: 0 in every smoke run.
- CPU tests `r1x4_tests/test_run_config.py`, `test_loss.py`, `test_iv.py`, `test_iv_fit.py` pass.

## Metrics

`pool/*`, `rlplus/*`, `mc/*`, `logra/*`, `actor/teacher_lowp_drop_frac`, `actor/ches_drop_frac`, `teacher/token_weight_mean`.

## Known limits

- LoGRA: per-step weight changes (~1e-5 relative) are below bf16 resolution, so the bf16 probe sees rounding flips; q then
  grows ~linearly in α, so α is only approximately calibrated. Watch `logra/q_pred_kl`, `logra/q_zero`.
- RL-PLUS: its exploration-based advantage (1−π)^γ is not implemented (not requested).
- MC skip adds one generate call per step (on all-correct candidates only).
