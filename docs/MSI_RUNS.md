# MSI runs (agate), Oct 2026

## Environment

One conda env for everything, `/scratch.global/kamel026/envs/r1x4`: python 3.10, torch 2.6.0 (cu124), vLLM 0.8.5,
transformers 4.51.3, flash-attn 2.7.4.post1.

- build: `sbatch scripts/setup_msi.sbatch` (CPU job; runs `scripts/setup_msi.sh` with `requirements.txt`), then `sbatch scripts/gpu_check_msi.sbatch`
- `source scripts/msi_env.sh` sets `R1X4_ROOT`, `HF_HOME` and caches under `/scratch.global/$USER`
- exact snapshot of the running env: `env/r1x4_pip_freeze.txt`, `env/r1x4_conda_explicit.txt`
- `setup_msi.sh` installs `verl/` editable, but `run.py` runs `python -m verl...` from the kit's `verl/` directory, so the kit's verl is the one imported

Data (not in git, under `R1X4_ROOT=/scratch.global/kamel026/r1x4_runs`): `data/valid.parquet` (2,019 problems),
`data/clean/rw_pairs_qwen3-0.6b_clean_lp_tlp_first0_ctlp.parquet` (26,801 problems; built by `scripts/clean_and_score.py`
on the r14x `msi-setup` branch), `data/ampo/openr1_math_4longcots.parquet`.

## Main run: R1x4 rewrite pairs, Qwen3-0.6B

`R1X4-RW_PAIRS_QWEN3-0.6B_CLEAN_LP_TLP_FIRST0_CTLP-QWEN3-0.6B-P1PCT-NOCHES-LUFFYORDER-PT1PCT-TSUPP-DV-0.05-ENT0-RLPLUS0TR100NOCLIP-MCSKIP0-MUON`,
started 2026-10-28, Slurm chain 4985906 -> 4985907 -> 4985908 -> 4985913 -> 4985917 -> 4985918 (a100-4, 24 h links).

```bash
cd r1x4kit
DATA=/scratch.global/kamel026/r1x4_runs/data/clean/rw_pairs_qwen3-0.6b_clean_lp_tlp_first0_ctlp.parquet \
P_TEACHER=1 CHES=no MC_SAMPLES=0 DELTA_V=-0.05 ABORT_VAL_FRAC=0.5 STEPS=269 OPTIM=muon \
ROOT=/scratch.global/kamel026/r1x4_runs KIT=$PWD PYENV=/scratch.global/kamel026/envs/r1x4 HF_HOME=/scratch.global/kamel026/hf \
RUN_ARGS="--ranking no --rlplus_warmup 0 --rlplus_cbar 1e9 --rlplus_denom teacher --rlplus_refresh 100 --teacher_support_drop yes" \
scripts/slurm/submit_chain.sh a100x4 6 -t 24:00:00
```

(`run.py --dry_run` with these arguments gives exactly the Hydra overrides printed in the job's log.)

Recipe: LUFFY visit order (no ranking), 100 prompts/step, 4 students + 4 teachers per prompt (exactly 4C/4W),
skip prompts whose 4 students are all correct, RL-PLUS ratio 2 pi / (pi_ref + pi_teacher) with the reference student
refreshed every 100 steps and no clip, teacher tokens outside the generator's sampling support dropped on every row,
wrong-teacher tokens below p = 1% dropped, AsymRE delta V = -0.05, entropy 0, no KL, Muon, val every 20 steps,
abort below 50% of the step-0 val. Step-0 val/avg_score 0.349 (AIME 0.088, MATH 0.662, Olympiad 0.399, AMC 0.416, Minerva 0.180).

Smoke tests of this recipe (1 L40S): `scripts/msi_smoke/smoke_msi.sbatch` (abort, delta V, 4/4 skip),
`scripts/msi_smoke/smoke_tden.sbatch` (teacher denominator, refresh, no clip, support drop).

## Baselines

GRPO, AMPO and LUFFY in the official AMPO repo on Qwen3-0.6B: `baselines/ampo_official/README.md`.
v1 (2026-10-28, AMPO's data/prompt/grader): GRPO chain 5024031-5024040, AMPO chain 5024041-5024052, superseded (see below).

## Progress (2026-10-29 09:15 CDT)

Validation average over the 5 sets (AIME, MATH, Olympiad, AMC, Minerva). The main run uses the LUFFY `\boxed{}` prompt and
grader, the baselines use AMPO's `<answer>` prompt and grader, so the columns are not directly comparable; all final
checkpoints will be scored with one common eval.

| step | main run | GRPO | AMPO |
|---|---|---|---|
| 0  | 0.349 | 0.314 | 0.316 |
| 10 | –     | 0.326 | 0.311 |
| 20 | 0.348 | 0.333 | 0.296 |
| 30 | –     | 0.335 | –     |
| 40 | 0.343 | –     | –     |
| 60 | 0.338 | –     | –     |

Main run at step 77/269 (~9-12 min/step), GRPO at 36/510 (~18 min/step), AMPO at 29/510 (~20 min/step).
Baseline checkpoints are 7.3 GB each (last 3 kept).

## Main run extended to 1500 steps (2026-10-29)

Ivan: continue until at least 150k prompts (100 prompts/step -> 1500 steps, ~5.6 passes over the 26,801 problems).
The first chain stopped cleanly at step 119 (deadline stop). Its remaining jobs were cancelled and the chain was resubmitted
with the same command and `STEPS=1500` (same experiment name, so `--resume auto` continues from `global_step_119`; the
dry run differs only in `trainer.total_training_steps`; LR is constant, so nothing else depends on the step count):
`... STEPS=1500 ... scripts/slurm/submit_chain.sh a100x4 11 -t 24:00:00 -A interran`, jobs 5138136-5138148.

## A100 or H100 (2026-10-30)

All queued jobs (main run and both baselines) were switched in place with
`scontrol update JobId=<id> Partition=a100-4,a100-8,msigpu Features="a100|h100" TresPerNode=gres/gpu:4`, so each 24 h job
starts on whichever 4-GPU A100 or H100 node frees up first (V100 nodes excluded). Note: `scontrol update` resets
the jobs' age priority (AccrueTime), so they lost their accumulated queue age. The main run picks its
`--hw` memory profile from the GPU memory (40 GB -> a100_40, 80 GB -> a100_80); `--hw` changes only memory knobs, not the recipe
or the experiment name. New baseline jobs: `baselines/ampo_official/baseline.sbatch` requests the same.

## Baselines v2: our data, prompts and grader (2026-10-30)

Ivan: every baseline must use exactly our prompts and math_verify v4, with no format reward. The v1 chains were cancelled
(GRPO at step 70, val 0.362; AMPO at step 60, val ~0.298; checkpoints kept in `ckpt/Qwen3-0.6B-{GRPO,AMPO}`).
v2 (`baselines/ampo_official/README.md`): our 26,801 problems and prompt column, R1 trace as teacher (4 copies for
AMPO's 4 slots), reward v4 for train and val, `format_coefficient=0`, `max_prompt_length=1536`, and LUFFY added
(LUFFY train.sh settings in the AMPO trainer). Smoke 5277855 (L40S): all three exit 0, format reward 0, LUFFY
prefix ratio 1/8, AMPO 2-3/8.

Chains (8 links each, ~510 steps = 65k prompts): GRPO 5283405-5283413, AMPO 5283414-5283422, LUFFY 5283424-5283431,
experiment names `Qwen3-0.6B-{GRPO,AMPO,LUFFY}-OURSDATA-V4`.

2026-10-31: the scripts' default val file was still AMPO's `data/valid.parquet` (`<answer>` prompt). It now points to our
`data/valid.parquet` (same problems, golds and order, `\boxed{}` prompt). AMPO and LUFFY had not started yet. GRPO
(5283405, 4xA100 40GB, ~17 min/step) was stopped after its step-5 checkpoint and resumes from it in the next link with
the correct val file; its step-0 val (AMPO prompt) is not comparable and is replaced by AMPO's/LUFFY's step-0 (same base model).

## Decision point at step 400 (agreed with Ivan, 2026-10-31)

The main run's val falls from 0.349 (step 0) to 0.329 (step 100). A dip is expected; the question is whether it rebounds.
At step 400 (40k prompts, ~1.5 passes over the 26,801 problems): if val is still going down, rethink the method; if it is
flat or going up, continue to step 1500.

## A40/L40S nodes and no re-validation on resume (2026-11-01)

The A100/H100 queues had ~1,900 pending jobs (a new 4-GPU job: estimated start ~34 h later), `preempt-gpu` (6x 8xA40, 5x 4xL40S,
48 GB) ~140 (estimated start ~2 h). All 33 queued jobs (main run, GRPO, AMPO, LUFFY) were updated with
`scontrol update JobId=<id> Partition=a100-4,a100-8,msigpu,preempt-gpu Features="a100|h100|a40|l40s"` (this resets their
queue age). `preempt-gpu` jobs can be preempted by `interactive-gpu` users and are then requeued (PreemptMode=REQUEUE,
JobRequeue=1); every run saves every 5 steps and resumes automatically, so a preemption loses at most 5 steps. On a 48 GB
GPU the main run's a100x4 profile picks the `a100_40` memory settings (memory knobs only); the baselines' settings fit.
Steps on A40/L40S are expected to be slower than on A100 (to be measured).

Every job used to validate before its first step even when resuming (~30 min per 24 h job; AMPO re-scored step 60).
`scripts/run.py` and the baseline `train_*_msi.sh` now skip that validation when a checkpoint exists; a new run still
validates at step 0, and the next test_freq step validates as before.

## Main run crash at the step-140 validation (2026-11-01), NCCL timeout raised to 1 h

Job 5138136 (4xA40, agc07) resumed at step 119, validated step 120 (0.320, ~41 min on A40 vs ~30 on A100) and crashed in the
step-140 validation at 22:11: `Watchdog caught collective operation timeout: WorkNCCL(... ALLREDUCE ...) ran for 600037 ms`.
One rank finished its share of the 2,019 val prompts and waited > 10 min (the default `nccl_timeout`) for the others.
Lost: steps 136-139 (the chain resumes from the step-135 save). `scripts/run.py` and the baseline `train_*_msi.sh` now pass
`++actor_rollout_ref.nccl_timeout=3600`. Main-run val at step 120: 0.320.
