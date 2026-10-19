#!/bin/bash
# race_cancel.sh JOBID... : when one of the jobs is RUNNING, cancel the rest.
while true; do
  r=$(squeue -h -j "$(echo "$@" | tr ' ' ',')" -t R -o %i 2>/dev/null | head -1)
  if [ -n "$r" ]; then for j in "$@"; do [ "$j" != "$r" ] && scancel $j; done; echo "running: $r"; exit 0; fi
  [ -z "$(squeue -h -j "$(echo "$@" | tr ' ' ',')" -o %i 2>/dev/null)" ] && { echo "none left"; exit 1; }
  sleep 20
done
