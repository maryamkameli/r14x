#!/bin/bash
#SBATCH -J r1x4-clean
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 04:00:00
#SBATCH -o clean_%j.out
# Clean Ivan's rewrite pairs (cut error-flag sentences, drop CJK, re-grade) + teacher-forced token log-probs.
#   FIRST=64 sbatch scripts/clean_and_score_msi.sbatch      # quick check
#   sbatch scripts/clean_and_score_msi.sbatch               # all 26,904 pairs
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source scripts/msi_env.sh
P=${PAIRS:-$R1X4_ROOT/data/ivan_drive/rewrite_qwen3-0.6b_30k/pairs_right_wrong_qwen3-0.6b_26904.parquet}
mkdir -p "$R1X4_ROOT/data/clean"
O=$R1X4_ROOT/data/clean/rw_pairs_qwen3-0.6b_clean_lp_first${FIRST:-0}.parquet
python scripts/clean_and_score.py --pairs "$P" --first "${FIRST:-0}" --out "$O"
echo "EXIT $?"
pkill -u "$USER" -f 'EngineCore' 2>/dev/null
python scripts/check_clean_output.py "$O"   # token ids == trainer tokenization, logps valid, no flag/CJK left
echo "CHECK_EXIT $?"
