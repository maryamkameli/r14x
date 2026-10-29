#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:4
#SBATCH -c 32
#SBATCH --mem=200g
#SBATCH -t 24:00:00
#SBATCH -o /scratch.global/kamel026/r1x4_runs/baselines/logs/%x_%j.out
# One 24h link of a resumable chain running the official AMPO repo's train_${ALGO}.sh (ALGO=ampo|grpo) on Qwen3-0.6B,
# 4xA100. trainer.resume_mode=auto picks up the latest checkpoint (saved every 10 steps); links after the end exit at once.
source /scratch.global/kamel026/r1x4_runs/baselines/env.sh
ALGO=${ALGO:?set ALGO=ampo|grpo} STEPS=${STEPS:-510}
NAME=Qwen3-0.6B-${ALGO^^}
T=$B/ckpt/$NAME/latest_checkpointed_iteration.txt
if [[ -f $T && $(cat $T) -ge $STEPS ]]; then echo "BASELINE_DONE $NAME at step $(cat $T)"; exit 0; fi
echo "== $NAME job $SLURM_JOB_ID on $(hostname) at $(date '+%F %T'), resuming from $(cat $T 2>/dev/null || echo scratch)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
EXP_NAME=$NAME STEPS=$STEPS bash $B/train_${ALGO}_msi.sh >> $B/logs/$NAME.log 2>&1
echo "TRAIN_EXIT $?"
rm -rf $RAY_TMPDIR
