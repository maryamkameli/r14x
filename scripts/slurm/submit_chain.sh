#!/bin/bash
# Submit a chain of N dependent r1x4_rewrite.sbatch jobs (afterany) for one hardware profile; every job resumes the
# same experiment from its last checkpoint and stops cleanly before its walltime.
#
#   scripts/slurm/submit_chain.sh <a40x4|a40x8|a100x4> <n_jobs> [extra sbatch args...]
#
# Extra sbatch args come last and override the profile defaults (-p, -t, -c, --mem, -A ...), e.g.
#   scripts/slurm/submit_chain.sh a100x4 3 -A mygroup -t 24:00:00
# Exported to the jobs (set them in the calling shell): STEPS (260), OPTIM (muon|logra|adamw), KIT, ROOT, A100_MEM,
# NAME, RUN_ARGS, OVERRIDES, MARGIN_MIN, PYENV (see r1x4_rewrite.sbatch). AFTER=<jobid> makes the first job wait for it.
set -euo pipefail

usage() { sed -n '5p' "$0" | sed 's/^#   //' >&2; exit 2; }
(( $# >= 2 )) || usage
PROFILE=$1 N=$2; shift 2
[[ $N =~ ^[1-9][0-9]*$ ]] || usage
case $PROFILE in  # gres, partition hint (MSI Agate), walltime, cores, host memory
  a40x4)  GRES=gpu:a40:4  PART=preempt-gpu TIME=24:00:00 CPUS=32 MEM=160g ;;
  a40x8)  GRES=gpu:a40:8  PART=preempt-gpu TIME=24:00:00 CPUS=64 MEM=320g ;;
  a100x4) GRES=gpu:a100:4 PART=a100-4      TIME=48:00:00 CPUS=32 MEM=200g ;;
  *) echo "unknown profile $PROFILE (a40x4 | a40x8 | a100x4)" >&2; usage ;;
esac

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
export PROFILE STEPS=${STEPS:-260} OPTIM=${OPTIM:-muon}
export KIT=${KIT:-$(cd "$HERE/../.." && pwd)} ROOT=${ROOT:-/scratch.global/radke149/r1x4_runs}
for f in "$ROOT/data/rewrite_pairs_qwen3-0.6b.parquet" "$ROOT/data/valid.parquet"; do
  [[ -f $f ]] || { echo "missing $f (copy the data first, see scripts/slurm/README.md)" >&2; exit 1; }
done
mkdir -p "$ROOT/logs"

dep=${AFTER:-}
echo "profile $PROFILE: --gres=$GRES -p $PART -t $TIME -c $CPUS --mem=$MEM${*:+ (+ $*)}; STEPS=$STEPS OPTIM=$OPTIM ROOT=$ROOT KIT=$KIT"
for i in $(seq 1 "$N"); do
  jid=$(sbatch --parsable --export=ALL -J "r1x4-$PROFILE-$OPTIM" -o "$ROOT/logs/r1x4_${PROFILE}_%j.out" \
        --gres="$GRES" -p "$PART" -t "$TIME" -c "$CPUS" --mem="$MEM" ${dep:+--dependency=afterany:$dep} \
        "$@" "$HERE/r1x4_rewrite.sbatch")
  jid=${jid%%;*}  # --parsable prints <jobid>[;<cluster>]
  echo "job $i/$N: $jid${dep:+ (afterany:$dep)}  log $ROOT/logs/r1x4_${PROFILE}_$jid.out"
  dep=$jid
done
