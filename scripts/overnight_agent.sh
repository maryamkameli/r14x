#!/usr/bin/env bash
# Start inside tmux on the box BEFORE going to sleep:   tmux new -s night 'bash scripts/overnight_agent.sh'
# At Mon Oct 19 00:00 (start of the GPU window) it
#   1) starts the agent loop (the agent watches the bootstrap and then the training; re-launched whenever it exits),
#   2) bootstraps: env + kernel tests (setup_rtx.sh), data copy, model download, variant-IV mu pool (~1-2 h on 4 GPUs),
#   3) starts the training queue (scripts/queue_overnight.txt) in its own process: training runs even if the agent dies.
# AGENT = claude (default) | codex | custom (then AGENT_CMD='mycli --flags' gets the prompt as its last argument)
KIT=$(cd "$(dirname "$0")/.." && pwd); AGENT=${AGENT:-claude}
source "$KIT/scripts/rtx_env.sh"
A=$R1X4_ROOT/agent; mkdir -p $A; B=$A/bootstrap.log
START=${START:-202610190000}
echo "$(date) waiting for $START (GPU window) ..." | tee -a $B
while [ "$(date +%Y%m%d%H%M)" -lt "$START" ]; do sleep 30; done
until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1,2,3 | grep -q .; do
  echo "$(date) GPUs 0-3 busy (another user), waiting" >> $B; sleep 120; done

agent_loop() {
  cd "$KIT"; local n=0
  while [ "$(date +%Y%m%d%H%M)" -lt 202610232355 ]; do
    n=$((n + 1))
    local P="$(cat $KIT/AGENT_TASK.md)

(Session $n, $(date). KIT=$KIT R1X4_ROOT=$R1X4_ROOT. Bootstrap log: $B (BOOTSTRAP_DONE / BOOTSTRAP_FAILED markers;
while neither is there the bootstrap is still running: watch it, do not start a second copy). Read $A/JOURNAL.md and
$A/STATUS.md first if they exist and continue from there.)"
    echo "$(date) agent session $n ($AGENT)" >> $A/sessions.log
    case $AGENT in
      claude) claude -p "$P" --dangerously-skip-permissions --output-format text >> $A/agent_$n.log 2>&1 ;;
      codex)  codex exec --dangerously-bypass-approvals-and-sandbox "$P" >> $A/agent_$n.log 2>&1 ;;
      *)      $AGENT_CMD "$P" >> $A/agent_$n.log 2>&1 ;;
    esac
    echo "$(date) agent session $n exited ($?)" >> $A/sessions.log
    if grep -q BOOTSTRAP_DONE $B && ! grep -v '^\s*#' $KIT/scripts/queue_overnight.txt | grep -v '^\s*$' \
         | grep -vxFf $KIT/scripts/queue_overnight.txt.done 2>/dev/null | grep -q .; then break; fi
    sleep 600
  done
}
agent_loop &

{ echo "$(date) BOOTSTRAP_START"; nvidia-smi
  ok=1
  [ -f "$R1X4_ROOT/attn_impl" ] || bash "$KIT/scripts/setup_rtx.sh" || ok=0
  source "$KIT/scripts/rtx_env.sh"
  mkdir -p $R1X4_ROOT/data/mu; cp -n $KIT/data/*.parquet $R1X4_ROOT/data/ 2>/dev/null; cp -rn $KIT/data/ampo $R1X4_ROOT/data/ 2>/dev/null
  cp -n $KIT/data/mu/*.parquet $R1X4_ROOT/data/mu/ 2>/dev/null
  python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['Qwen/Qwen3-1.7B','Qwen/Qwen3-4B']]" || ok=0
  # variant IV needs Qwen3-4B mu-traces for the teacher phase: T_a*batch = 60*48 = 2,880 prompts (+ margin); resumable
  have=$(python -c "import glob,pyarrow.parquet as pq; print(sum(pq.read_table(f).num_rows for f in glob.glob('$R1X4_ROOT/data/mu/mu_qwen3-4b_core*.parquet')))" 2>/dev/null || echo 0)
  if [ $ok = 1 ] && [ "${have:-0}" -lt 3000 ]; then
    echo "$(date) generating the mu pool (have $have rows)"
    for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s python $KIT/scripts/gen_teacher_mu.py --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-1.7B \
      --data $R1X4_ROOT/data/core.parquet --first 3000 --n 4 --shard $s --num_shards 4 --gpu_mem 0.9 \
      --out $R1X4_ROOT/data/mu/mu_qwen3-4b_core.parquet > $A/mu_shard$s.log 2>&1 & done; wait
    grep -q MU_DONE $A/mu_shard0.log $A/mu_shard1.log $A/mu_shard2.log $A/mu_shard3.log 2>/dev/null || ok=0
    echo "$(date) mu pool: $(grep -h 'with a correct' $A/mu_shard*.log | tail -4)"
  fi
  if [ $ok = 1 ]; then
    nohup bash $KIT/scripts/run_queue.sh $KIT/scripts/queue_overnight.txt >> $A/queue.log 2>&1 &
    echo "$(date) queue started (log $A/queue.log)"; echo BOOTSTRAP_DONE
  else
    echo "$(date) BOOTSTRAP_FAILED (see above); the agent takes over"
  fi
} >> $B 2>&1
wait
echo "$(date) overnight script ended" >> $B
