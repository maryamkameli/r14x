# AMPO and GRPO baselines (official AMPO repo) on Qwen3-0.6B, MSI

The baselines for "beats AMPO" run the authors' own code and scripts, not this kit's fork:
[EnigmaYYYY/AMPO](https://github.com/EnigmaYYYY/AMPO) at commit `e102d5c66057594b52fe3578d9e45a4481743c40`,
`exp_scripts/train_ampo.sh` and `exp_scripts/train_grpo.sh` (GRPO = the same script with no off-policy targets).

| file | what |
|---|---|
| `train_ampo_msi.sh`, `train_grpo_msi.sh` | the official scripts with only paths, logging, checkpointing and speed settings changed; `official.diff` is the full diff |
| `env.sh` | environment for every job: the r1x4 conda env, official AMPO verl first on `PYTHONPATH`, `stubs/`, `unset ROCR_VISIBLE_DEVICES` |
| `stubs/sitecustomize.py` | placeholders for `vertexai`, `google-cloud-aiplatform`, `sentence_transformers`, imported at load time by `verl/deepscaler/utils.py` for an LLM grader that only runs with `use_math_orm=True` (AMPO never sets it) |
| `smoke.sbatch` | 2 steps of each script on 1 L40S, 8 prompts, 32-row val (`make_valid32.py`) |
| `baseline.sbatch`, `submit_chain.sh` | one 24 h link on 4xA100 (`a100-4`), `resume_mode=auto`; `submit_chain.sh ampo|grpo [links]` queues the chain |

## Changes vs the official scripts (`official.diff`)

Algorithm, data, batch sizes (128 prompts x 8 rollouts, mini-batch 64), rewards (format coefficient 0.1, repetition
penalty), lr 1e-6, entropy 1e-3, 8,192-token responses, val every 10 steps and 510 steps are unchanged.

- model `Qwen/Qwen3-0.6B`; train file = AMPO's `openr1_math_4longcots.parquet` (8,491 prompts, `$R1X4_ROOT/data/ampo/`); val = the repo's `data/valid.parquet` (the same 2,019 problems as ours, in AMPO's `<answer>` prompt format)
- console logging, wandb offline
- `val_before_train=True` (step-0 score), `save_freq` 50 -> 10 and `max_actor_ckpt_to_keep=3` (24 h job limit, disk)
- speed only: no CPU offload, vLLM tensor parallel 1 (0.6B fits on one GPU), `ray_init.num_cpus=30`
- `BSZ`, `MINI_BSZ`, `NGPUS`, `STEPS`, `TEST_FREQ`, `SAVE_FREQ`, `TEST_FILE`, `EXP_NAME`, `KEEP_CKPT` env overrides (defaults = the values above; the smoke test uses them)

## Setup and launch

```bash
git clone https://github.com/EnigmaYYYY/AMPO.git /scratch.global/$USER/ampo_official
git -C /scratch.global/$USER/ampo_official checkout e102d5c66057594b52fe3578d9e45a4481743c40
B=/scratch.global/$USER/r1x4_runs/baselines; mkdir -p $B/logs $B/ckpt
cp -r baselines/ampo_official/* $B/
python $B/make_valid32.py /scratch.global/$USER/ampo_official/data/valid.parquet $B/valid32.parquet
sbatch $B/smoke.sbatch                  # both must print SMOKE_*_EXIT 0
$B/submit_chain.sh grpo 10; $B/submit_chain.sh ampo 10
```

The scripts hardcode `/scratch.global/kamel026`; change `ROOT` in `train_*_msi.sh` and the paths in `env.sh`,
`*.sbatch` for another user. Logs: `$B/logs/Qwen3-0.6B-{GRPO,AMPO}.log`; checkpoints `$B/ckpt/Qwen3-0.6B-{GRPO,AMPO}/`.

## LUFFY

The official LUFFY repo ([ElliottYan/LUFFY](https://github.com/ElliottYan/LUFFY) `01f4ae0`, and its `ExGRPO/`) pins
vLLM 0.6.3 / transformers 4.46.3, and its verl only has vLLM adapters up to 0.6.3; Qwen3 needs vLLM >= 0.8.4 and
transformers >= 4.51. Its `data/valid.parquet` is identical to ours. LUFFY on Qwen3 is deferred.

## Comparing with our runs

Our val prompt asks for `\boxed{}`, AMPO's for `<answer>` tags, and the graders differ slightly, so the final
comparison should score every checkpoint with one shared evaluation script.
