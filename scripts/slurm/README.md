# Running the Qwen3-0.6B rewrite-pairs run on MSI with Slurm (A40 / A100)

This is the run launched on the RTX PRO 6000 box, split into resumable Slurm jobs:
```
python scripts/run.py --model qwen3-0.6b --data $ROOT/data/rewrite_pairs_qwen3-0.6b.parquet --val $ROOT/data/valid.parquet \
  --p 1 --p_rows neg --ches yes --ranking yes --rank_score gold --rank_rolling yes --rank_pool 1000 --rank_top 100 \
  --batch 100 --rlplus yes --rlplus_warmup 10 --rlplus_cbar 1.0 --mc_skip yes --mc_samples 4 --mc_z 1.0 \
  --muon yes --entropy 0 --steps 260 --test_freq 20 --save_freq 5 --keep_ckpt 2 --resume auto \
  --gpus $GPUS --hw $HW --root $ROOT --logger '["console","tensorboard"]' ray_init.num_cpus=<cpus-2>
```
- `r1x4_rewrite.sbatch` runs one job. It sets up the MSI env, prints `nvidia-smi`, sets
  `R1X4_DEADLINE` to the Slurm end time minus 15 minutes, runs `run.py`, and prints a marker line at the end.
- `submit_chain.sh <profile> <n_jobs> [extra sbatch args...]` submits `n_jobs` copies, each one
  `--dependency=afterany` on the previous one. Every job resumes the same experiment.

## Profiles

| PROFILE  | `--gres`     | GPUs | `--hw`                         | partition hint | `-t` default | `-c` / `--mem` |
|----------|--------------|------|--------------------------------|----------------|--------------|----------------|
| `a40x4`  | `gpu:a40:4`  | 4    | `a40`                          | `preempt-gpu`  | 24:00:00     | 32 / 160g      |
| `a40x8`  | `gpu:a40:8`  | 8    | `a40`                          | `preempt-gpu`  | 24:00:00     | 64 / 320g      |
| `a100x4` | `gpu:a100:4` | 4    | `a100_80`, or `a100_40` (below) | `a100-4`       | 48:00:00     | 32 / 200g      |

`a100x4` chooses `a100_40` when `A100_MEM=40` is set. If `A100_MEM` is unset, the job reads `nvidia-smi` and uses
`a100_40` when the card has less than 60 GB. MSI lists the A100s in `a100-4` / `a100-8` as 40 GB, so expect `a100_40`
there. The job prints which one it chose. The partitions and times are only hints. Extra sbatch args come last and
override them (`-p`, `-t`, `-A`, `-c`, `--mem`, ...).

`--hw` changes only the memory knobs (values for Qwen3-0.6B; batch 100 and the recipe stay the same):

| `--hw`              | rollout `gpu_memory_utilization` | actor `ppo_max_token_len_per_gpu` | rollout `log_prob_max_token_len_per_gpu` | attention |
|---------------------|----------------------------------|-----------------------------------|------------------------------------------|-----------|
| `rtx6000` (default) | 0.45                             | 32768                             | 65536                                    | `$R1X4_ATTN_IMPL`, else flash_attention_2 |
| `a40`, `a100_40`    | 0.5                              | 16384                             | 32768                                    | flash_attention_2 |
| `a100_80`           | 0.45                             | 32768                             | 65536                                    | flash_attention_2 |

A40 cards have 48 GB, but the A40 profile plans for 40 GB. The bigger models in `run.py`'s `MODELS` also have values in
`HW`: on 40 GB the 4B/7B token budgets are halved and optimizer offload is on. Those values are untested.

## 1. Copy the kit and the data to MSI (run on the RTX box; MSI cannot ssh to it)
```bash
source /mnt/data/radke149/r1x4kit_1p7b_4b/scripts/rtx_env.sh       # sets R1X4_ROOT
cd /mnt/data/radke149
rsync -a --exclude __pycache__ --exclude outputs --exclude data/src --exclude '*.egg-info' \
  r1x4kit_1p7b_4b/ radke149@agate.msi.umn.edu:/scratch.global/radke149/r1x4kit_1p7b_4b/
ssh radke149@agate.msi.umn.edu mkdir -p /scratch.global/radke149/r1x4_runs/data
rsync -a $R1X4_ROOT/data/rewrite_pairs_qwen3-0.6b.parquet $R1X4_ROOT/data/valid.parquet \
  radke149@agate.msi.umn.edu:/scratch.global/radke149/r1x4_runs/data/
```
The first rsync overwrites the older kit copy on MSI with this branch. To keep that copy, rsync to a different
directory and `export KIT=<that dir>` before submitting. `run.py` runs `python -m verl...` from `$KIT/verl`, so the
`ampo` env does not need a reinstall.

On an MSI login node, check that the model is in the offline HF cache (the jobs set `HF_HUB_OFFLINE=1`):
```bash
HF_HOME=/scratch.global/radke149/hf-cache /scratch.global/radke149/envs/ampo/bin/python -c \
  "from huggingface_hub import snapshot_download as d; print(d('Qwen/Qwen3-0.6B'))"
```

## 2. Smoke test first (2 steps)
The MSI `ampo` env has torch 2.6 / vLLM 0.8.5. This box ran torch 2.7 / vLLM 0.9.2, and the Muon, RL-PLUS,
MC-skip and rolling-ranking code paths were only exercised here. So either run this 2-step smoke on each profile, or
watch the first chain job closely. It uses its own experiment name, so it never touches the real checkpoint.

Inside an allocation, e.g. `srun -p preempt-gpu --gres=gpu:a40:4 -c 32 --mem=160g -t 01:00:00 --pty bash`:
```bash
export HF_HOME=/scratch.global/radke149/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_DISABLE_COMPILE_CACHE=1
export PATH=/scratch.global/radke149/envs/ampo/bin:$PATH RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID; mkdir -p $RAY_TMPDIR
cd /scratch.global/radke149/r1x4kit_1p7b_4b; R=/scratch.global/radke149/r1x4_runs
python scripts/run.py --model qwen3-0.6b --data $R/data/rewrite_pairs_qwen3-0.6b.parquet --val $R/data/valid.parquet \
  --p 1 --p_rows neg --ches yes --ranking yes --rank_score gold --rank_rolling yes \
  --rlplus yes --rlplus_warmup 10 --rlplus_cbar 1.0 --mc_skip yes --mc_samples 4 --mc_z 1.0 --muon yes --entropy 0 \
  --steps 2 --batch 16 --rank_pool 64 --rank_top 16 --no_val_before_train --test_freq 0 --save_freq 5 \
  --gpus 4 --hw a40 --root $R --resume disable --name SMOKE-A40X4 --logger '["console"]' ray_init.num_cpus=30
```
(Use `--gpus 8` for `a40x8`, and `--hw a100_40` or `--hw a100_80` on A100.) Or submit the same smoke as a single
job through the chain script. It ends with `R1X4_DONE SMOKE-... step 2`:
```bash
NAME=SMOKE-A40X4-$(date +%m%d%H%M) STEPS=2 \
RUN_ARGS="--batch 16 --rank_pool 64 --rank_top 16 --no_val_before_train --test_freq 0 --resume disable" \
  scripts/slurm/submit_chain.sh a40x4 1 -t 01:00:00
```
Afterwards: `rm -rf /scratch.global/radke149/r1x4_runs/ckpt/SMOKE-*`.

## 3. Submit
```bash
cd /scratch.global/radke149/r1x4kit_1p7b_4b
scripts/slurm/submit_chain.sh a40x4 4                    # 4 x 24 h on 4 A40s
scripts/slurm/submit_chain.sh a40x8 3                    # 8 A40s (its own checkpoint, see below)
scripts/slurm/submit_chain.sh a100x4 2 -A <group>        # extra sbatch args override the defaults
OPTIM=logra scripts/slurm/submit_chain.sh a100x4 2       # muon (default) | logra | adamw
AFTER=<jobid> scripts/slurm/submit_chain.sh a40x4 2      # extend a running chain
```
Environment read by the jobs (set it in the submitting shell; `--export=ALL`): `STEPS` (260), `OPTIM`, `KIT`,
`ROOT` (`/scratch.global/radke149/r1x4_runs`), `A100_MEM`, `NAME`, `RUN_ARGS` (extra `run.py` flags),
`OVERRIDES` (extra Hydra overrides, space-separated), `MARGIN_MIN` (15), `PYENV` (`/scratch.global/radke149/envs/ampo`).

## 4. What to expect
- **Speed:** the per-step time on A40 / A100 is **unknown** (nothing has run there yet). Read `timing_s/step`
  in the first job's training log, then size `n_jobs`: 260 steps in total, validation every 20 steps.
- **Logs:**
  - `$ROOT/logs/r1x4_<profile>_<jobid>.out`: env, GPUs, deadline, experiment name, marker.
  - `$ROOT/logs/<name>_<mmdd_HHMM>.log`: the full training log, one per job.
  - `$ROOT/tensorboard/<name>`.
  - Check progress with `grep -h 'R1X4_' $ROOT/logs/r1x4_*.out`.
- **Markers:** every job ends with one of:
  - `R1X4_DONE <name> step N`: `latest_checkpointed_iteration.txt` is at least `STEPS`. Any later chain job prints it
    at once and exits within a minute; `scancel` the rest if you like.
  - `R1X4_RESUMABLE <name> step N`: stopped before the deadline (exit 0), or crashed (exit = run.py's code, plus the
    tail of the training log).

## 5. Resume behaviour
- **Deadline stop.** With `--resume auto`, each job restores the model, the optimizer, the Muon/LoGRA state and the
  rolling-pool curriculum (`data.pt`) from `$ROOT/ckpt/<name>/global_step_N`. After every step the trainer checks
  whether the next step, plus validation if one is due, plus a save would end after `R1X4_DEADLINE`. If so, it saves
  and exits cleanly. Validation before training runs only on a fresh start, not on resume.
- **Lost steps.** A save happens every 5 steps, and the last 2 checkpoints are kept. A preemption (`preempt-gpu`),
  node failure or walltime kill loses at most the steps since the last save. The `afterany` chain then continues.
- **Crashes.** A crash also continues the chain. A deterministic crash therefore repeats in every remaining job, so
  look at the first one.
- **Experiment name.** It is `run.py`'s descriptive name, the same as on the RTX box. It does not depend on `--hw` and
  never contains the job id. FSDP checkpoints are sharded per GPU count (`model_world_size_<G>_rank_<r>.pt`), so
  `a40x8` adds `-8GPU` to the name. A 4-GPU checkpoint moves between `a40x4`, `a100x4` and the RTX box. To continue
  the RTX run on MSI, copy `$R1X4_ROOT/ckpt/<name>/` to `$ROOT/ckpt/<name>/`. A job that finds a checkpoint written
  with another GPU count stops with an error instead of crashing in the loader.
- **One chain per name.** Two chains with the same name and `ROOT` would overwrite each other's checkpoints. To run
  profiles side by side, give each its own `NAME` or `ROOT`.

## 6. If a 40 GB GPU runs out of memory
Batch 100 is part of the recipe; do not lower it. Lower the memory knobs with Hydra overrides instead. They only change
how the batch is split into micro-batches or how much memory vLLM takes; the experiment name stays the same, so the
chain keeps resuming. `scancel` the pending jobs and resubmit with, for example:
```bash
# OOM in update_actor / compute_log_prob: smaller token budgets (keep log_prob = 2 x ppo, ppo >= 9728 = 1536 + 8192)
OVERRIDES="actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576" \
  scripts/slurm/submit_chain.sh a40x4 3
# OOM in vLLM (init / generation): a smaller vLLM share
OVERRIDES="actor_rollout_ref.rollout.gpu_memory_utilization=0.4" scripts/slurm/submit_chain.sh a40x4 3
# last resort: offload the optimizer state to CPU (slower)
OVERRIDES="actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" scripts/slurm/submit_chain.sh a40x4 3
```
If flash-attn fails to load in the `ampo` env, use `RUN_ARGS="--attn sdpa"`.
