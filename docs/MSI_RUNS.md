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

Official AMPO repo, AMPO and GRPO on Qwen3-0.6B: `baselines/ampo_official/README.md`.
Started 2026-10-28: GRPO chain 5024031-5024040, AMPO chain 5024041-5024052.
