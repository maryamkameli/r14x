#!/bin/bash
#SBATCH -J r1x4-rewrite
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=160g
#SBATCH -t 24:00:00
#SBATCH -o /scratch.global/radke149/r1x4_runs/logs/r1x4_rewrite_%j.out
# One link of a resumable chain (submit with scripts/slurm/submit_chain.sh; see scripts/slurm/README.md):
# the R1x4 rewrite-pairs recipe on Qwen3-0.6B, batch 100, 260 steps, checkpoint every 5 steps, --resume auto.
# The trainer saves and exits cleanly before R1X4_DEADLINE (= Slurm end time - MARGIN_MIN); the next job resumes.
#
# Environment (all optional except PROFILE):
#   PROFILE    a40x4 | a40x8 | a100x4
#   STEPS      total training steps (260)
#   OPTIM      muon | logra | adamw (muon)
#   KIT        kit checkout (/scratch.global/radke149/r1x4kit_1p7b_4b)
#   ROOT       runs root with data/, ckpt/, logs/, tensorboard/ (/scratch.global/radke149/r1x4_runs)
#   A100_MEM   40 | 80: A100 memory profile for a100x4 (default: detected from nvidia-smi, 80 if undetectable)
#   NAME       experiment name override (default: run.py's name, + -<N>GPU when GPUS != 4)
#   RUN_ARGS   extra run.py flags, space-separated (e.g. "--keep_ckpt 3")
#   OVERRIDES  extra Hydra overrides, space-separated (e.g. "actor_rollout_ref.rollout.gpu_memory_utilization=0.4")
#   MARGIN_MIN minutes kept free before the Slurm end time (15)
#   PYENV      python env prefix (/scratch.global/radke149/envs/ampo: torch 2.6 / vLLM 0.8.5)
# Exit: 0 after a finished run or a deadline stop; run.py's code after a crash. Last line: R1X4_DONE or R1X4_RESUMABLE.
set -euo pipefail

PROFILE=${PROFILE:?set PROFILE=a40x4|a40x8|a100x4}
STEPS=${STEPS:-260}
OPTIM=${OPTIM:-muon}
KIT=${KIT:-/scratch.global/radke149/r1x4kit_1p7b_4b}
ROOT=${ROOT:-/scratch.global/radke149/r1x4_runs}
MARGIN_MIN=${MARGIN_MIN:-15}
PYENV=${PYENV:-/scratch.global/radke149/envs/ampo}
JOB=${SLURM_JOB_ID:-local$$}

export HF_HOME=${HF_HOME:-/scratch.global/radke149/hf-cache} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_DISABLE_COMPILE_CACHE=1 PYTHONUNBUFFERED=1
export PATH=$PYENV/bin:$PATH RAY_TMPDIR=/tmp/ray_$JOB
mkdir -p "$RAY_TMPDIR" "$ROOT/logs"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

echo "== r1x4 job $JOB on $(hostname) at $(date '+%F %T')  PROFILE=$PROFILE STEPS=$STEPS OPTIM=$OPTIM"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__)' || true

case $PROFILE in
  a40x4)  GPUS=4 HW=a40 ;;
  a40x8)  GPUS=8 HW=a40 ;;
  a100x4) GPUS=4 HW=a100_80 ;;
  *) echo "unknown PROFILE=$PROFILE (a40x4 | a40x8 | a100x4)" >&2; exit 2 ;;
esac
if [[ $PROFILE == a100x4 ]]; then
  mem=${A100_MEM:-}
  if [[ -z $mem ]]; then  # MiB of the first visible GPU; < 60 GB -> the 40 GB A100
    mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -dc '0-9' || true)
    mem=80; [[ -n $mib && $mib -lt 60000 ]] && mem=40
  fi
  case $mem in 40) HW=a100_40 ;; 80) HW=a100_80 ;; *) echo "A100_MEM=$mem: use 40 or 80" >&2; exit 2 ;; esac
fi
case $OPTIM in
  muon)  OPT=(--muon yes) ;;
  logra) OPT=(--logra yes) ;;
  adamw) OPT=() ;;
  *) echo "unknown OPTIM=$OPTIM (muon | logra | adamw)" >&2; exit 2 ;;
esac

if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  ngpu=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c . || true)
else
  ngpu=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
fi
if (( ngpu != GPUS )); then
  echo "PROFILE=$PROFILE needs $GPUS GPUs but $ngpu are visible (check --gres)" >&2; exit 2
fi

# ---- wall-clock deadline: Slurm end time - MARGIN_MIN
to_sec() {  # Slurm duration [D-]HH:MM:SS | MM:SS | SS -> seconds; non-zero for UNLIMITED / NOT_SET / garbage
  local t=$1 d=0 h=0 m=0 s=0 p
  [[ $t =~ ^([0-9]+-)?[0-9]+(:[0-9]+){0,2}$ ]] || return 1
  if [[ $t == *-* ]]; then d=${t%%-*}; t=${t#*-}; fi
  IFS=: read -r -a p <<< "$t"
  case ${#p[@]} in
    3) h=${p[0]} m=${p[1]} s=${p[2]} ;;
    2) m=${p[0]} s=${p[1]} ;;
    1) s=${p[0]} ;;
  esac
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
END=
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  for _ in 1 2 3; do  # time left now (squeue %L), the current limit even after `scontrol update TimeLimit`
    left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | head -n1 | tr -d ' ' || true)
    if sec=$(to_sec "$left") && (( sec > 0 )); then END=$(( $(date +%s) + sec )); break; fi
    sleep 5
  done
  if [[ -z $END ]]; then  # scontrol: EndTime, else StartTime + TimeLimit
    info=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null || true)
    if [[ $info =~ EndTime=([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}) ]]; then
      END=$(date -d "${BASH_REMATCH[1]}" +%s 2>/dev/null || true)
    fi
    if [[ -z $END && $info =~ StartTime=([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}) ]]; then
      st=$(date -d "${BASH_REMATCH[1]}" +%s 2>/dev/null || true)
      if [[ -n $st && $info =~ TimeLimit=([0-9:-]+) ]] && lim=$(to_sec "${BASH_REMATCH[1]}"); then END=$(( st + lim )); fi
    fi
  fi
  if [[ -z $END && ${SLURM_JOB_END_TIME:-} =~ ^[0-9]+$ ]]; then END=$SLURM_JOB_END_TIME; fi
fi
if [[ -n $END ]]; then
  export R1X4_DEADLINE=$(( END - MARGIN_MIN * 60 ))
  echo "Slurm end $(date -d "@$END" '+%F %T'); R1X4_DEADLINE=$R1X4_DEADLINE ($(date -d "@$R1X4_DEADLINE" '+%F %T'), ${MARGIN_MIN} min margin)"
else
  unset R1X4_DEADLINE
  echo "WARNING: could not determine the job end time; running without R1X4_DEADLINE (a walltime kill loses the steps since the last save)"
fi

# ---- the run
read -r -a RA <<< "${RUN_ARGS:-}"
read -r -a OV <<< "${OVERRIDES:-}"
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
RAY_CPUS=$(( NCPU > 4 ? NCPU - 2 : 2 ))
ARGS=(--model qwen3-0.6b --data "$ROOT/data/rewrite_pairs_qwen3-0.6b.parquet" --val "$ROOT/data/valid.parquet"
      --p 1 --p_rows neg --ches yes --ranking yes --rank_score gold --rank_rolling yes --rank_pool 1000 --rank_top 100
      --batch 100 --rlplus yes --rlplus_warmup 10 --rlplus_cbar 1.0 --mc_skip yes --mc_samples 4 --mc_z 1.0
      ${OPT[@]+"${OPT[@]}"} --entropy 0 --steps "$STEPS" --gpus "$GPUS" --hw "$HW" --test_freq 20 --save_freq 5
      --keep_ckpt 2 --root "$ROOT" --resume auto --logger '["console","tensorboard"]'
      ${RA[@]+"${RA[@]}"})
cd "$KIT"
if [[ -z ${NAME:-} ]]; then  # same name in every job of the chain; FSDP shards are per world size -> GPU count suffix
  NAME=$(python scripts/run.py "${ARGS[@]}" --dry_run | sed -n '1s/^# //p' || true)
  [[ -n $NAME ]] || { echo "could not derive the experiment name (run.py --dry_run failed)" >&2; exit 2; }
  (( GPUS == 4 )) || NAME=$NAME-${GPUS}GPU
fi
CKPT=$ROOT/ckpt/$NAME
latest() { local n=; [[ -s $CKPT/latest_checkpointed_iteration.txt ]] && n=$(tr -dc '0-9' < "$CKPT/latest_checkpointed_iteration.txt"); echo "${n:-0}"; }
echo "experiment $NAME  hw=$HW gpus=$GPUS ray_cpus=$RAY_CPUS  ckpt $CKPT (latest step $(latest))"

n=$(latest)
if (( n >= STEPS )); then
  echo "R1X4_DONE $NAME step $n >= $STEPS (nothing to run)"; exit 0
fi
if (( n > 0 )); then  # a checkpoint written with another GPU count cannot be loaded
  shopt -s nullglob
  mine=("$CKPT/global_step_$n/actor/"*"world_size_${GPUS}_rank_0.pt")
  any=("$CKPT/global_step_$n/actor/"*world_size_*_rank_0.pt)
  shopt -u nullglob
  if (( ${#any[@]} > 0 && ${#mine[@]} == 0 )); then
    echo "checkpoint $CKPT/global_step_$n was written with another GPU count (${any[0]##*/}); this profile has $GPUS" >&2; exit 2
  fi
fi

set +e
python scripts/run.py "${ARGS[@]}" --name "$NAME" ray_init.num_cpus="$RAY_CPUS" ${OV[@]+"${OV[@]}"}
rc=$?
set -e
log=$(ls -t "$ROOT/logs/${NAME}"_*.log 2>/dev/null | head -n1 || true)
deadline_stop=0
[[ -n $log ]] && grep -q '\[r1x4\] deadline in' "$log" && deadline_stop=1
n=$(latest)
echo "run.py exit $rc; deadline stop $deadline_stop; latest checkpoint step $n / $STEPS; log $log"
if (( n >= STEPS )); then
  echo "R1X4_DONE $NAME step $n"; exit 0
fi
if (( rc != 0 && ! deadline_stop )); then
  [[ -n $log ]] && tail -n 40 "$log"
  echo "R1X4_RESUMABLE $NAME step $n (run.py FAILED with exit $rc; next job resumes from step $n)"; exit "$rc"
fi
echo "R1X4_RESUMABLE $NAME step $n"
exit 0
