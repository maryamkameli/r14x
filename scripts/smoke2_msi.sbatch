#!/bin/bash
#SBATCH -J r1x4-smoke2
#SBATCH -c 16
#SBATCH --mem=120g
#SBATCH -t 02:00:00
#SBATCH -o /scratch.global/radke149/r1x4_runs/logs/smoke2_%j.out
# (1) resume the variant-I smoke from global_step_3 to step 4; (2) variant III (LUFFY order, no CHES) for 2 steps.
export HF_HOME=/scratch.global/radke149/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_DISABLE_COMPILE_CACHE=1
export PATH=/scratch.global/radke149/envs/ampo/bin:$PATH RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
S=/scratch.global/radke149/r1x4_runs/smoke; cd /scratch.global/radke149/r1x4kit_1p7b_4b
C="--model qwen3-1.7b --data $S/data/smoke.parquet --val $S/data/valid.parquet --root $S --gpus 1 --batch 8 --logger [\"console\"] ray_init.num_cpus=14"
python scripts/run.py $C --p 1 --ches yes --ranking yes --rank_pool 32 --rank_top 16 --steps 4 --test_freq 0 --save_freq 4 --resume auto --name SMOKE-yes-yes-$RESUME_FROM
echo RESUME_EXIT $?
python scripts/run.py $C --p 1 --ches no --ranking no --steps 2 --test_freq 0 --save_freq 0 --resume disable --no_val_before_train --name SMOKE-III-$SLURM_JOB_ID
echo III_EXIT $?
