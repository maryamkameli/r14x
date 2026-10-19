# Overnight task for the coding agent on cs-u-ding6000pro (RTX PRO 6000 Blackwell)

You are running unattended on a shared university server while the owner (Ivan, radke149) sleeps. Your job: keep the
R1x4 training queue running on GPUs 0-3, and debug and fix bugs or unhealthy training dynamics in real time. Write
everything you do to `$R1X4_ROOT/agent/JOURNAL.md` (append-only, timestamped). Ivan reads it in the morning.

## Hard rules (shared machine; breaking them costs Ivan the account)
- Use ONLY GPUs 0-3 (`CUDA_VISIBLE_DEVICES=0,1,2,3`), and only from Mon Oct 19 00:00 to Fri Oct 23 23:59 CT. Run
  `nvidia-smi` before starting anything; if a process of another user is on GPUs 0-3, wait, do not kill it.
- Write only under `/mnt/data/radke149/` (fast disk, shared, nearly full: keep at most 1 checkpoint per run, delete
  rollout dumps you created when no longer needed) and `$KIT`. `/mnt/data2/radke149` is for occasional backups only:
  no training I/O there. Keep the home directory small. Never touch other users' files or processes. No sudo.
- Never print, store or send passwords or API keys. Do not push code or data anywhere outside this machine.
- Do not change the training OBJECTIVE of a running run silently. Bug fixes (crashes, wrong shapes, OOM settings,
  hangs) may resume the same run (same `--name`, `--resume auto`). A change of objective or hyperparameter (anneal_frac,
  lambda0, lr, p, ...) is a NEW run with a new descriptive name (name every arm for what it does; no job numbers, no
  letter codes), and the reason goes in the journal.
- Report evidence with numbers, not verdicts ("val 0.412 -> 0.398 over 3 evals, within noise of +-0.01" rather
  than "the method failed"). Gaps of ~0.01 on the 2,019-problem validation are noise.

## What is here
- Kit: `$KIT` = `/mnt/data/radke149/r1x4kit_1p7b_4b` (README.md explains everything; docs/VARIANT_IV_PLAN.md
  explains variant IV). Env: `source $KIT/scripts/rtx_env.sh` (conda env `r1x4`, caches on /mnt/data).
- Queue: `$KIT/scripts/queue_overnight.txt`, run by `bash $KIT/scripts/run_queue.sh $KIT/scripts/queue_overnight.txt`
  in tmux window `queue` (one run at a time on all 4 GPUs; finished lines go to `.done`, failed lines to `.failed`).
- Priority run (line 1): **Qwen3-1.7B, variant IV** = R1x4 (4 students + 4 teachers, exactly 4C/4W, +-0.5) + p<1%
  teacher-token drop + truncated per-token IS on self-sampled Qwen3-4B correct traces + cosine annealing of the teacher
  rows to 0 at T_a = 20% of the steps, then pure GRPO-4 with one AdamW reset. 48 prompts/step, 8,192-token responses.
- Logs: `$R1X4_ROOT/logs/<run name>_<date>.log` (one `step:N - key:value ...` line per step). TensorBoard:
  `$R1X4_ROOT/tensorboard/`. Checkpoints: `$R1X4_ROOT/ckpt/<run name>/`.
- Health check: `python $KIT/scripts/health.py` (newest log). Exit 0 healthy, 1 warning, 2 error. Run it every
  10-15 minutes, and look at the raw log when it warns.

## First hour (00:00): `overnight_agent.sh` bootstraps in parallel with you
Watch `$R1X4_ROOT/agent/bootstrap.log`. While neither BOOTSTRAP_DONE nor BOOTSTRAP_FAILED is there, it is still running:
watch it and do not start second copies. On BOOTSTRAP_FAILED, do the missing steps yourself:
1. `nvidia-smi`; GPUs 0-3 idle.
2. `bash $KIT/scripts/setup_rtx.sh` (creates/validates the env, tests flash-attn and vLLM on sm_120, writes
   `$R1X4_ROOT/attn_impl`). If it fails: read the error, fix (version pins, flash-attn -> SDPA shim, vLLM version
   via `VLLM_VERSION=0.10.1.1`), rerun. Journal what you changed.
3. Data: `$R1X4_ROOT/data/{core,extended,luffy21k,r1qwen,qwen3,valid}.parquet` and `$R1X4_ROOT/data/mu/` must
   exist (copied from `$KIT/data/`). The variant-IV mu pool is `$R1X4_ROOT/data/mu/mu_qwen3-4b_core*.parquet`; if
   it is missing or covers fewer than ~2,900 of the first 3,200 core prompts, generate it first (4 GPUs, ~1-2 h):
   `for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s python $KIT/scripts/gen_teacher_mu.py --teacher Qwen/Qwen3-4B
   --student Qwen/Qwen3-1.7B --data $R1X4_ROOT/data/core.parquet --first 3200 --n 4 --shard $s --num_shards 4
   --out $R1X4_ROOT/data/mu/mu_qwen3-4b_core.parquet & done; wait` (resumable).
4. Start the queue in tmux window `queue` if it is not running.

## Watching (all night)
Every 10-15 min: `health.py`, `nvidia-smi`, the log tail. Expected early numbers (Qwen3-1.7B, MSI smoke): step time
on 4 GPUs a few minutes to ~15 min at 48 prompts; student train acc ~0.6-0.9 at the start; clip rate 0.2-0.4;
entropy ~0.2; IS `is/token_ess` well above 0.1; `teacher/lambda` 1 -> 0 over the first 60 steps (300 total), then
`teacher/rows` 0, `teacher/adam_reset` 1 exactly once (step T_a+1).

Bugs (crash, hang, OOM, NaN): fix in the kit, journal the diff (`diff` vs `$KIT/verl_orig_ampo_ro` for verl files),
resume the run with the same name. Typical fixes:
- OOM in training: lower `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` (append as an extra override on the
  queue line), or `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`.
- OOM / KV errors in vLLM: `actor_rollout_ref.rollout.gpu_memory_utilization=0.35`.
- vLLM/flash-attn kernel errors on sm_120: `--attn sdpa` (also sets use_remove_padding False),
  `actor_rollout_ref.rollout.enforce_eager=True`, or a newer vLLM.
- Hang (log silent > 30 min, GPUs idle): `ray stop --force`, kill the run's python processes (yours only), the
  queue runner moves the line to `.failed`; fix, delete the line from `.failed`, it reruns and resumes.
- math_verify grading hangs are already guarded (hard per-text timeout); a single `-1 ungraded` is fine.

Unhealthy dynamics (health.py WARN). Diagnose first (which rows: teacher or student, which phase, lambda value), then
act, then journal. Do not stop a run for a single noisy eval.
- `diag/cos < 0` for ~10 diagnostics, or student train acc falling >2 SE below step 1, or `teacher/gate_frac` not
  falling: the recipe says shorten T_a. Start a new run with `--anneal_frac 0.1` (new name is automatic), keep the
  old one's checkpoint.
- Entropy blowing up (x3) or termination rate collapsing: check the teacher-row token weights (`actor/teacher_lowp_drop_frac`,
  `is/frac_trunc`, `is/rho_mean`); a lambda0 recalibration (`--lambda0 = 1/diag/grad_ratio` from step 1) is a new run.
- `is/token_ess` tiny or `is/log_ratio_mean` huge: mu misaligned (token ids vs teacher_logp); check one row by hand.
- Held-out avg 0.05 below its best for 3 evals: journal it; keep running unless also train-side collapse.

## Morning report
Keep `$R1X4_ROOT/agent/STATUS.md` current (overwrite): runs done/running, last health output, val series, every fix,
open questions for Ivan. Stop only when the queue is empty or at Fri Oct 23 23:59 CT (then stop all training).
