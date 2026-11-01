# GRPO, AMPO and LUFFY baselines (official AMPO repo) on Qwen3-0.6B, MSI

The baselines run the AMPO authors' own code, not this kit's fork:
[EnigmaYYYY/AMPO](https://github.com/EnigmaYYYY/AMPO) at commit `e102d5c66057594b52fe3578d9e45a4481743c40`,
`exp_scripts/train_ampo.sh` and `exp_scripts/train_grpo.sh` (GRPO = the same trainer with no off-policy targets).
LUFFY is AMPO's `train_ampo.sh` with LUFFY's settings (see below).

Every baseline uses **our training problems and prompts, our val file and our grader**, so the only difference from our run
is the algorithm:

- train = our 26,801 problems with our exact `prompt` column (`make_train_ours.py`); the teacher (AMPO `target_lst`,
  LUFFY off-policy trace) is the original DeepSeek-R1 trace of the same problem in LUFFY's `openr1.parquet` (row
  `luffy_index`; gold answers checked equal for every row). All 26,801 traces fit in 8,192 tokens and contain `</think>` and `\boxed{}`.
- val = our `data/valid.parquet` (`\boxed{}` system prompt). It has the same 2,019 problems, golds and row order as AMPO's
  `data/valid.parquet`, which differs only in its `<answer>`-tag system prompt.
- grader = R1x4 reward v4 for training and validation (`reward_impl_version=6`, added to the AMPO repo by `grader_v4.diff`):
  `math_verify(parse('$gold$'), parse(whole response))`, 0/1, with `<answer>X</answer>` rewritten to `\boxed{X}` first.
  No format reward (`format_coefficient=0`).
- `max_prompt_length=1536` as in our run (2 val prompts exceed 1,024 tokens).

| file | what |
|---|---|
| `train_grpo_msi.sh`, `train_ampo_msi.sh`, `train_luffy_msi.sh` | the official scripts with the changes above plus paths, logging, checkpointing and speed settings; `official.diff` is the full diff (LUFFY vs `train_ampo.sh`) |
| `grader_v4.diff` | patch to `verl/verl/adaptive_mix_src/math_verify_reward.py` in the AMPO checkout |
| `make_train_ours.py` | builds `data/baseline/train_ours_r1teacher.parquet` from our train file and LUFFY's `openr1.parquet` |
| `env.sh` | environment for every job: the r1x4 conda env, official AMPO verl first on `PYTHONPATH`, `stubs/`, `unset ROCR_VISIBLE_DEVICES` |
| `stubs/sitecustomize.py` | placeholders for `vertexai`, `google-cloud-aiplatform`, `sentence_transformers`, imported at load time by `verl/deepscaler/utils.py` for an LLM grader that only runs with `use_math_orm=True` (never set) |
| `smoke.sbatch` | 2 steps of each script on 1 L40S, 8 prompts, 32-row val (`make_valid32.py`) |
| `baseline.sbatch`, `submit_chain.sh` | one 24 h link on 4 GPUs (A100, H100, or A40/L40S in `preempt-gpu`, whichever starts first), `resume_mode=auto`; `submit_chain.sh grpo|ampo|luffy [links]` queues the chain |

## AMPO teachers

The official AMPO data has 4 different teachers per problem (`max_available_targets=4`, `num_off_policy_targets=2`,
adaptive k). Ours has one (R1), so `target_lst` holds 4 copies of the R1 trace (`N_COPIES=4`), and AMPO can still
replace up to 2 rollouts per group as in the official setting. With `use_adaptive_k=True` it may insert up to 4 copies
when the student solves almost nothing, i.e. 4 identical R1 traces in a group of 8.

## LUFFY

The official LUFFY repo ([ElliottYan/LUFFY](https://github.com/ElliottYan/LUFFY) `01f4ae0`) pins vLLM 0.6.3 /
transformers 4.46.3, too old for Qwen3 (vLLM >= 0.8.4, transformers >= 4.51). AMPO's trainer is built on LUFFY's, so
LUFFY runs there with the settings of LUFFY's `exp_scripts/train.sh`: 1 off-policy trace in every group
(`num_off_policy_targets=1`, `injection_strategy=always`, no adaptive k), token-level off-policy loss, `loss_remove_clip=True`,
`p/(p+0.1)` shaping, lr 1e-6, entropy 1e-3, 8 rollouts, batch 128. This is a re-implementation, not LUFFY's code.

## Other changes vs the official scripts

Batch sizes (128 prompts x 8 rollouts, mini-batch 64), lr 1e-6, entropy 1e-3, 8,192-token responses, val every 10 steps
and 510 steps are unchanged.

- model `Qwen/Qwen3-0.6B`; console logging, wandb offline
- `val_before_train=True` (step-0 score) for a new run only; a job resuming from a checkpoint skips it, `save_freq` 50 -> 5 and `max_actor_ckpt_to_keep=3` (the trainer has no deadline
  stop, so a 24 h timeout loses the steps since the last save)
- speed only: no CPU offload, vLLM tensor parallel 1 (0.6B fits on one GPU), `ray_init.num_cpus=30`
- `BSZ`, `MINI_BSZ`, `NGPUS`, `STEPS`, `TEST_FREQ`, `SAVE_FREQ`, `TRAIN_FILE`, `TEST_FILE`, `EXP_NAME`, `KEEP_CKPT` env
  overrides (the smoke test uses them)

## Setup and launch

```bash
git clone https://github.com/EnigmaYYYY/AMPO.git /scratch.global/$USER/ampo_official
git -C /scratch.global/$USER/ampo_official checkout e102d5c66057594b52fe3578d9e45a4481743c40
git -C /scratch.global/$USER/ampo_official/ampo apply $PWD/baselines/ampo_official/grader_v4.diff
B=/scratch.global/$USER/r1x4_runs/baselines; mkdir -p $B/logs $B/ckpt
cp -r baselines/ampo_official/* $B/
source $B/env.sh; D=/scratch.global/$USER/r1x4_runs/data; mkdir -p $D/baseline
python $B/make_train_ours.py $D/clean/rw_pairs_qwen3-0.6b_clean_lp_tlp_first0_ctlp.parquet $D/src/openr1.parquet \
       $D/baseline/train_ours_r1teacher.parquet
python $B/make_valid32.py $D/valid.parquet $B/valid32.parquet
sbatch $B/smoke.sbatch                  # all three must print SMOKE_*_EXIT 0
for a in grpo ampo luffy; do $B/submit_chain.sh $a 8; done
```

The scripts hardcode `/scratch.global/kamel026`; change `ROOT` in `train_*_msi.sh` and the paths in `env.sh`,
`*.sbatch` for another user. Logs: `$B/logs/Qwen3-0.6B-{GRPO,AMPO,LUFFY}-OURSDATA-V4.log`; checkpoints
`$B/ckpt/Qwen3-0.6B-{GRPO,AMPO,LUFFY}-OURSDATA-V4/`.

## v1 (superseded)

The first GRPO/AMPO chains (2026-10-28 to 10-30, `Qwen3-0.6B-{GRPO,AMPO}`) used AMPO's own data (8,491 problems,
4 teachers), its `<answer>` prompt and its prime_math grader with a 0.1 format reward. They were stopped at steps 70 / 60
when the setup was switched to ours; their scripts are in the git history of this directory.
