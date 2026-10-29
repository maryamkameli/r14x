#!/bin/bash
#SBATCH -J r1x4-smoke06b
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=96g
#SBATCH -t 02:00:00
#SBATCH -o /scratch.global/kamel026/r1x4_runs/logs/smoke_%j.out
# Smoke test of Ivan's 2026-10-27 recipe on Qwen3-0.6B, 1 GPU: 200 cleaned prompts, 64-row valid, batch 8, ranking
# pool 32 / top 16. RL-PLUS warmup 1 so step 2 uses the IS ratio; abort_val_frac 10 forces the abort at the step-2 val.
export HF_HOME=/scratch.global/kamel026/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_DISABLE_COMPILE_CACHE=1
export PATH=/scratch.global/kamel026/envs/r1x4/bin:$PATH RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
mkdir -p $RAY_TMPDIR
nvidia-smi --query-gpu=name,memory.total --format=csv
S=/scratch.global/kamel026/r1x4_runs/smoke
cd /scratch.global/kamel026/r1x4kit_ivan
python scripts/run.py --model qwen3-0.6b --data $S/data/train200.parquet --val $S/data/valid64.parquet --root $S \
  --p 1 --p_rows neg --p_teacher 1 --ches ${CHES:-no} --ranking yes --rank_score gold --rank_rolling yes --rank_pool 32 --rank_top 16 \
  --batch 8 --rlplus yes --rlplus_warmup 1 --rlplus_cbar 1.0 --mc_skip yes --mc_samples 0 --mc_z 1.0 \
  --delta_v -0.05 --abort_val_frac 10 --muon yes --entropy 0 --steps 3 --test_freq 2 --save_freq 2 --keep_ckpt 1 \
  --gpus 1 --hw a40 --resume disable --name SMOKE06B-$SLURM_JOB_ID --logger '["console"]' ray_init.num_cpus=14
echo SMOKE_EXIT $?
ls $S/ckpt/SMOKE06B-$SLURM_JOB_ID; cat $S/ckpt/SMOKE06B-$SLURM_JOB_ID/initial_val.json $S/ckpt/SMOKE06B-$SLURM_JOB_ID/ABORTED
