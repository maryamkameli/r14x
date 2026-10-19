#!/bin/bash
#SBATCH -J ampo-smoke
#SBATCH -c 16
#SBATCH --mem=120g
#SBATCH -t 02:00:00
#SBATCH -o /scratch.global/radke149/r1x4_runs/logs/ampo_smoke_%j.out
# MSI smoke test of the AMPO fork (per-update loglik selection over all teacher traces), Qwen3-1.7B, 1 GPU, 2 steps.
export HF_HOME=/scratch.global/radke149/hf-cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_DISABLE_COMPILE_CACHE=1
export PATH=/scratch.global/radke149/envs/ampo/bin:$PATH RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
S=/scratch.global/radke149/r1x4_runs/smoke
cd /scratch.global/radke149/r1x4kit_1p7b_4b
python scripts/run_ampo.py --model qwen3-1.7b --selection ${SEL:-loglik} --root $S --steps 2 --gpus 1 --name AMPO-SMOKE-${SEL:-loglik}-$SLURM_JOB_ID \
  data.train_batch_size=8 actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  trainer.save_freq=2 trainer.test_freq=2 ray_init.num_cpus=14 'trainer.logger=["console"]'
echo SMOKE_EXIT $?
