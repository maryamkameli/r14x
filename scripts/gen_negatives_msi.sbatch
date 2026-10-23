#!/bin/bash
#SBATCH -J r1x4-neg
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 04:00:00
#SBATCH -o neg_%j.out
# Our own negative traces (scripts/gen_negatives.py) for the first FIRST openr1 problems, one model after the other.
#   sbatch scripts/gen_negatives_msi.sbatch
#   MODELS="r1d-qwen-1.5b" FIRST=1000 sbatch scripts/gen_negatives_msi.sbatch
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source scripts/msi_env.sh
OUT=$R1X4_ROOT/data/neg; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
for M in ${MODELS:-qwen3-0.6b r1d-qwen-1.5b}; do
  python scripts/gen_negatives.py --model "$M" --first "${FIRST:-256}" --n "${N:-8}" --out "$OUT/neg_${M}_first${FIRST:-256}"
  echo "EXIT $M $?"
  pkill -u "$USER" -f 'EngineCore|gen_negatives.py' 2>/dev/null   # free the GPU for the next model (see rewrite_msi.sbatch)
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ] && break; sleep 5; done
done
