#!/bin/bash
# Usage: submit_chain.sh ampo|grpo [links=8]  -- queues <links> 24h jobs, each starting after the previous one ends.
ALGO=$1 N=${2:-8} dep=
for i in $(seq $N); do
  dep=$(sbatch --parsable -J bl-$ALGO --export=ALL,ALGO=$ALGO ${dep:+--dependency=afterany:$dep} \
        /scratch.global/kamel026/r1x4_runs/baselines/baseline.sbatch); echo $dep
done
