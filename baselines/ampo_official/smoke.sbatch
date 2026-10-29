#!/bin/bash
#SBATCH -J ampo-off-smoke
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=96g
#SBATCH -t 02:00:00
#SBATCH -o /scratch.global/kamel026/r1x4_runs/baselines/logs/smoke_%j.out
# 2-step smoke of the official AMPO repo's train_grpo.sh and train_ampo.sh on Qwen3-0.6B, 1 GPU, 8 prompts, 32-row val.
source /scratch.global/kamel026/r1x4_runs/baselines/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
export NGPUS=1 BSZ=8 MINI_BSZ=8 STEPS=2 TEST_FREQ=2 SAVE_FREQ=2 VAL_BEFORE=True RAY_CPUS=14 TEST_FILE=$B/valid32.parquet
for k in grpo ampo; do
  EXP_NAME=SMOKE-$k-$SLURM_JOB_ID bash $B/train_${k}_msi.sh > $B/logs/smoke_${k}_$SLURM_JOB_ID.log 2>&1
  echo "SMOKE_${k}_EXIT $?"
done
rm -rf $RAY_TMPDIR
