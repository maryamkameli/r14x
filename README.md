# r1x4kit_1p7b_4b: R1x4 runs + the modified AMPO fork, for the RTX PRO 6000 Blackwell box

Copy of r1x4kit with the students swapped: **Qwen3-1.7B** (48 prompts/step) and **Qwen3-4B** (40 prompts/step) (and LUFFY's **Qwen2.5-Math-7B-16k-think**). Stack: AMPO's verl 0.5.0.dev
(`verl/`) with our R1x4 trainer added (`verl/verl/r1x4/`) and AMPO's own trainer patched (`verl/verl/adaptive_mix_src/`).
`verl_orig_ampo_ro/` is the untouched AMPO tree; `PATCHES.diff` lists every change.

## 1. What a run does (R1x4)
Per prompt and step: 4 live student rollouts + 4 stored teacher rows, filled to **exactly 4 correct / 4 wrong**
(k correct students → 4−k copies of LUFFY's DeepSeek-R1 trace + k distinct stored wrong traces). The group mean is
pinned at 0.5, so every advantage is ±0.5. Teacher rows use the off-policy token loss with the p/(p+0.1) reshape.
Recipe = our LUFFY-tree runs: lr 1e-6, no KL, entropy 1e-3, T=1.0, 8,192-token responses, 48 (1.7B) / 40 (4B) prompts/step (`--batch` overrides), no clip,
one optimizer step per training step, prompts retired after 2 consecutive all-correct visits.

| flag | meaning |
|---|---|
| `--data` | `luffy21k` (the 21k set of the 0.5B no-decline run), `core`, `extended`, `r1qwen` (only DeepSeek-R1-Distill-Qwen-1.5B negatives), `qwen3` (only Qwen3 negatives; built from Qwen3-0.6B now, rebuild with `--qwen3_prefix qwen3-8b` once the 8B generation is done) |
| `--p` | percent. Teacher tokens whose **pre-update** policy probability is below p% lose their gradient (dropped, not renormalized). `--p 1` = p < 1 %. `--p 0` = off |
| `--ches yes/no` | per-token CHES on every negative row: each token's final hidden state h⁻ₜ is scored against S⁺ = Σₜ h⁺ₜ of the group's positive (R1) trace (cosine; `--ches_score ip` for the inner product); the top `--ches_frac` (50 %) of the response tokens are dropped; the shared prefix with y⁺ is never dropped and ≥1 token is kept (the fixed rule from CHES_PER_TOKEN) |
| `--ranking yes/no` | take the next 1,000 prompts in LUFFY's visit order, teacher-force their R1 trace and their wrong traces under the **current** policy, keep the 100 whose traces are most likely (score = mean token log-likelihood of the pos trace + of the most likely neg trace; `--rank_score min/pos/neg`), train on those 100 **in decreasing likelihood**, then the next 1,000, and so on. The wrong slots of a ranked prompt use its most likely wrong traces first. `no` = plain LUFFY visit order |
| `--p_rows all/pos/neg` | which teacher rows the `--p` drop applies to; `neg` = only wrong-teacher rows |
| `--rank_score gold` | AMPO-style score: mean student probability of the gold answer tokens after the correct trace's last `\boxed{` |
| `--rank_rolling yes/no` | rolling pool: unpicked prompts stay and age; pool topped up to `--rank_pool` in visit order each step |
| `--rlplus yes/no`, `--rlplus_warmup N`, `--rlplus_cbar c` | teacher-token IS ratio π/(0.75 π_old + 0.25/V) (RL-PLUS), clipped at c; ratio π/π_old for the first N steps |
| `--mc_skip yes/no`, `--mc_samples k`, `--mc_z z` | all-correct prompts get k extra graded rollouts; whole group skipped iff 1−p̂ ≤ z·√(p̂(1−p̂)/n) |
| `--logra yes/no` (+ `--logra_rank`, `--logra_refresh`, `--logra_optimizer rowadam/sgd`, `--logra_lr`, `--logra_kl_start`, `--logra_kl_end`, `--logra_alpha_max`) | LoGRA low-rank sketched updates with predicted-KL step control instead of AdamW |
| `--muon yes/no` (+ `--muon_momentum`, `--muon_nesterov`, `--muon_ns_steps`, `--muon_scale moonlight/none`) | Muon on attention/MLP projections, AdamW on the rest; exclusive with `--logra` |
| `--hw rtx6000/a40/a100_40/a100_80` | memory profile per GPU type (see `scripts/slurm/README.md`) |

Rewrite-pairs run with these flags: see `docs/REWRITE_PAIRS_LOGRA_RUN.md`.

Variants: **I** `--p 1 --ches yes --ranking yes`, **II** `--p 1 --ches no --ranking yes`, **III** `--p 1 --ches no --ranking no`.

```bash
python scripts/run.py --model qwen3-1.7b --data core --p 1 --ches yes --ranking yes --steps 300
python scripts/run.py --model qwen3-4b   --data extended --p 1 --ches no --ranking no --steps 200
python scripts/run.py ... --dry_run          # print the full Hydra command only
```
Names follow the knobs, e.g. `R1X4-CORE-QWEN3-1.7B-P1PCT-TOKCHES-RANK100OF1000`. Logs: `$R1X4_ROOT/logs/`,
TensorBoard: `$R1X4_ROOT/tensorboard/`, checkpoints: `$R1X4_ROOT/ckpt/<name>/` (only the last one is kept). Validation:
the same 2,019 problems as all earlier runs (AIME24, AMC, MATH-500, Minerva, OlympiadBench), every 10 steps,
`val/avg_score`. Resume is automatic (same name → continues from the last checkpoint; `--resume disable` to start over).

## 1b. Variant IV (p<1% + IS + annealing)
`--is yes --anneal yes` (with `--p 1`). See docs/VARIANT_IV_PLAN.md, AGENT_TASK.md, BEFORE_SLEEP.md. The mu pool comes from
`scripts/gen_teacher_mu.py` (Qwen3-4B teacher, T=1, sampled-token log-probs, round-trip + grader filtered).

## 2. Modified AMPO (`scripts/run_ampo.py`)
Original AMPO: `RLHFDatasetWithTarget` sorts every prompt's teacher traces by a static `quality_score` and truncates
them to `max_available_targets` before training (the "selection at the beginning"); the trainer then ranks the kept
ones with its p(gold | teacher reasoning) score. The fork (`--selection loglik`, default): every teacher trace is
kept (`max_available_targets=-1`, `length_strategy=none`) and **right before each update** all of them are
teacher-forced under the current policy (one batched forward); the most likely ones fill the injected slots.
`--selection prob` keeps all traces with AMPO's own per-step score; `--selection static` is the original control.
Logged: `injection/loglik_chosen_mean` vs `injection/loglik_all_mean`.
```bash
python scripts/run_ampo.py --model qwen3-4b --selection loglik
python scripts/run_ampo.py --model qwen3-4b --selection static     # original AMPO, same everything else
```
Bugs found in AMPO and fixed: `_dump_generations` crashed (missing `sources`); with `use_dynamic_bsz=True` +
`loss_remove_token_mean=True` (both set by `train_ampo.sh`) each row is weighted by how many rows share its micro-batch.
The second fix is opt-in for AMPO (`--fix_loss`) so the control stays faithful, and always on for R1x4.

## 3. RTX box: first login and setup (GPUs 0-3 only, Mon 19 - Fri 23 Oct)
```bash
# laptop, on the UMN VPN
ssh radke149@cs-u-ding6000pro.cs.umn.edu          # change the default password when asked
tmux new -s main                                    # everything below inside tmux (Ctrl-b d detaches)
nvidia-smi                                          # GPUs 0-3 must be idle
mkdir -p /mnt/data/radke149 && cd /mnt/data/radke149
scp radke149@agate.msi.umn.edu:/scratch.global/radke149/exports/r1x4kit_1p7b_4b_*.zip .   # pull FROM MSI (MSI cannot ssh here)
unzip r1x4kit_1p7b_4b_*.zip && cd r1x4kit_1p7b_4b
bash scripts/setup_rtx.sh 2>&1 | tee /mnt/data/radke149/setup.log   # conda env, vLLM 0.9.2 / torch 2.7 cu128, kernel tests
source scripts/rtx_env.sh
mkdir -p $R1X4_ROOT/data && cp data/*.parquet $R1X4_ROOT/data/ && cp -r data/ampo $R1X4_ROOT/data/
python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['Qwen/Qwen3-1.7B','Qwen/Qwen3-4B']]"
bash scripts/run_queue.sh scripts/queue.txt          # runs the queue, one run at a time on all 4 GPUs
```
`setup_rtx.sh` checks that torch has sm_120 kernels, tries flash-attn (wheel, then source build for sm_120) and falls
back to PyTorch SDPA with a pure-torch shim if it cannot run; the choice is saved and picked up by `rtx_env.sh`.
`run_queue.sh` re-reads `queue.txt` before every launch (edit it any time) and waits if someone else is on GPUs 0-3.

**Disk** (`/mnt/data` is shared and nearly full): env ≈ 12 GB (on `/mnt/data2`), models 1.5 + 16 GB, datasets
≈ 1-2 GB, checkpoints per run (model + optimizer): 1.7B ≈ 27 GB, 4B ≈ 64 GB (`--ckpt_contents model,extra` ≈ 7 / 16 GB). Only the last checkpoint is kept. When done: copy `$R1X4_ROOT/{logs,tensorboard}` and the checkpoints you
need off the box (or to `/mnt/data2/radke149` as a backup), then delete them from `/mnt/data`.

## 4. What was tested where
MSI (A100/A40, torch 2.6 / vLLM 0.8.5, the `ampo` env): config composition, the grader, the dataset builder, and a
3-step smoke run of variant I on Qwen3-0.6B (the original kit; this copy differs only in model presets and batch) (generation → teachers → ranking → CHES → update → validation →
checkpoint). Not tested: the Blackwell stack itself (setup_rtx.sh tests it on the box before anything runs) and
Qwen3-1.7B/4B training at full batch.

## 5. Data (`data/`)
`build_datasets.py` builds all five sets from `data/src/` (LUFFY's openr1.parquet, the visit order, the 21k set, and
the neggen deliverables). Rows are sorted by LUFFY visit; every R1 trace re-grades correct and every wrong trace
re-grades wrong under the training grader (reward v4: math_verify on the whole response). Manifests next to each
parquet list the counts and drops.

AMPO (`run_ampo.py`) keeps its own recipe of 128 prompts x 8 rollouts per step; pass `data.train_batch_size=48` (or 40) as an override to change it.
