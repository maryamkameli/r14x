#!/bin/bash
#SBATCH -J ampo-smoke
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=120g
#SBATCH -t 02:00:00
#SBATCH -o ampo_smoke_%j.out
# MSI smoke test of the AMPO fork (per-update loglik selection over all teacher traces), Qwen3-1.7B, 1 GPU, 2 steps.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source scripts/msi_env.sh   # paths: scripts/msi_env.sh (R1X4_ROOT, HF_HOME, env)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
S=$R1X4_ROOT/smoke
python scripts/run_ampo.py --model qwen3-1.7b --selection ${SEL:-loglik} --root $S --steps 2 --gpus 1 --name AMPO-SMOKE-${SEL:-loglik}-$SLURM_JOB_ID \
  data.train_batch_size=8 actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  trainer.save_freq=2 trainer.test_freq=2 ray_init.num_cpus=14 'trainer.logger=["console"]'
echo SMOKE_EXIT $?
