#!/bin/bash
#SBATCH -J r1x4-clean
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 04:00:00
#SBATCH -o clean_%j.out
# Clean Ivan's rewrite pairs (cut error-flag sentences of wrong texts, drop CJK, re-grade) + teacher-forced token
# log-probs under the training prompt (student) and, for wrong texts, under their generation prompt (teacher).
#   FIRST=64 sbatch scripts/clean_and_score_msi.sbatch      # quick check
#   sbatch scripts/clean_and_score_msi.sbatch               # all 26,904 pairs
#   ADD_TEACHER_TO=<cleaned parquet> sbatch scripts/clean_and_score_msi.sbatch   # add correct_teacher_logps only
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source scripts/msi_env.sh
P=${PAIRS:-$R1X4_ROOT/data/ivan_drive/rewrite_qwen3-0.6b_30k/pairs_right_wrong_qwen3-0.6b_26904.parquet}
mkdir -p "$R1X4_ROOT/data/clean"
S=${SOURCE:-$R1X4_ROOT/data/ivan_drive/extended_20261026/extended.parquet}   # the rewrites' source (generation prompts)
O=$R1X4_ROOT/data/clean/rw_pairs_qwen3-0.6b_clean_lp_tlp_first${FIRST:-0}.parquet
if [[ -n ${ADD_TEACHER_TO:-} ]]; then  # only add correct_teacher_logps to an earlier output
  O=${ADD_TEACHER_TO%.parquet}_ctlp${FIRST:+_first$FIRST}.parquet
  python scripts/clean_and_score.py --pairs "$P" --source "$S" --first "${FIRST:-0}" --add_teacher_to "$ADD_TEACHER_TO" --out "$O"
else
  python scripts/clean_and_score.py --pairs "$P" --source "$S" --first "${FIRST:-0}" --out "$O"
fi
echo "EXIT $?"
pkill -u "$USER" -f 'EngineCore' 2>/dev/null
python scripts/check_clean_output.py "$O"   # token ids == trainer tokenization, logps valid, no flag/CJK left
echo "CHECK_EXIT $?"
