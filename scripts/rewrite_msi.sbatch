#!/bin/bash
#SBATCH -J r1x4-rewrite
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 04:00:00
#SBATCH -o rewrite_%j.out
# Rewrite R1 traces with the two small models (scripts/rewrite_traces.py), 1 GPU, one model after the other.
#   sbatch scripts/rewrite_msi.sbatch                                  # pilot: first 256 problems, 4 samples each
#   FIRST=0 sbatch -t 24:00:00 scripts/rewrite_msi.sbatch               # all 45,792 problems (resumable)
#   MODELS="qwen3-0.6b" THINKING=on sbatch scripts/rewrite_msi.sbatch   # Qwen3 with thinking on
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source scripts/msi_env.sh
OUT=$R1X4_ROOT/data/rewrite; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
for M in ${MODELS:-qwen3-0.6b qwen2.5-0.5b}; do
  T=${THINKING:-off}; [ "$M" = qwen2.5-0.5b ] && T=off
  python scripts/rewrite_traces.py --model "$M" --thinking "$T" --source "${SOURCE:-full}" --first "${FIRST:-256}" --n "${N:-4}" \
    --out "$OUT/rw_${M}_think${T}_${SOURCE:-full}_first${FIRST:-256}.parquet"
  echo "EXIT $M $?"
done
