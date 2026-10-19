#!/usr/bin/env bash
# Run the lines of a queue file one after another, each on all 4 GPUs. Edit the file while it runs: lines are re-read
# before every launch; a line that finished (exit 0) is recorded in <queue>.done and skipped; '#' lines are skipped.
#   tmux new -s q 'bash scripts/run_queue.sh scripts/queue.txt'
#   line format:  r1x4 <run.py args>   |   ampo <run_ampo.py args>
KIT=$(cd "$(dirname "$0")/.." && pwd); Q=${1:-$KIT/scripts/queue.txt}
source "$KIT/scripts/rtx_env.sh"
touch "$Q.done" "$Q.failed"
while true; do
  line=$(grep -v '^\s*#' "$Q" | grep -v '^\s*$' | grep -vxFf "$Q.done" | grep -vxFf "$Q.failed" | head -1)
  [ -z "$line" ] && { echo "$(date) queue empty"; exit 0; }
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES} | grep -q .; then
    echo "$(date) GPUs ${CUDA_VISIBLE_DEVICES} busy (another user?) - waiting"; sleep 300; continue; fi
  set -- $line; kind=$1; shift
  echo "$(date) START $line"
  case $kind in
    r1x4) python "$KIT/scripts/run.py" "$@" ;;
    ampo) python "$KIT/scripts/run_ampo.py" "$@" ;;
    *) echo "bad line: $line"; false ;;
  esac && echo "$line" >> "$Q.done" || { echo "$line" >> "$Q.failed"; echo "$(date) FAILED $line (moved to $Q.failed; fix and delete it there to retry)"; }
  ray stop --force >/dev/null 2>&1; sleep 20
done
