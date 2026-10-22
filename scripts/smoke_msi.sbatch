#!/bin/bash
#SBATCH -J r1x4-smoke17b
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=120g
#SBATCH -t 03:00:00
#SBATCH -o smoke_%j.out
# MSI smoke test (ampo env: torch 2.6 / vllm 0.8.5): 3 steps of variant I on Qwen3-1.7B, 1 GPU.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source scripts/msi_env.sh   # paths: scripts/msi_env.sh (R1X4_ROOT, HF_HOME, env)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
nvidia-smi --query-gpu=name,memory.total --format=csv
S=$R1X4_ROOT/smoke
python scripts/run.py --model qwen3-1.7b --data $S/data/smoke.parquet --val $S/data/valid.parquet --root $S \
  --p 1 --ches ${CHES:-yes} --ranking ${RANK:-yes} --gpus 1 --steps 3 --batch 8 --rank_pool 32 --rank_top 16 \
  --test_freq 3 --save_freq 3 --resume disable --name SMOKE17B-${CHES:-yes}-${RANK:-yes}-$SLURM_JOB_ID --dump_rollouts --logger '["console"]' \
  ray_init.num_cpus=14 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.rollout.gpu_memory_utilization=0.35
echo SMOKE_EXIT $?
