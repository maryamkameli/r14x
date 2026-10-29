# source scripts/msi_env.sh   -- environment for MSI (agate). Counterpart of rtx_env.sh.
# Everything big lives on /scratch.global/$USER (home quota is small): env, HF models, caches, runs.
# Override any of these before sourcing, e.g.  R1X4_ROOT=/scratch.global/$USER/other_runs source scripts/msi_env.sh
export R1X4_SCRATCH=${R1X4_SCRATCH:-/scratch.global/$USER}
export R1X4_ROOT=${R1X4_ROOT:-$R1X4_SCRATCH/r1x4_runs}             # data/, ckpt/, logs/, tensorboard/
export R1X4_ENV=${R1X4_ENV:-$R1X4_SCRATCH/envs/r1x4}                # conda env PREFIX (not a name)
C=${R1X4_CACHE:-$R1X4_SCRATCH/cache}
export HF_HOME=${HF_HOME:-$R1X4_SCRATCH/hf} XDG_CACHE_HOME=$C/xdg PIP_CACHE_DIR=$C/pip TRITON_CACHE_DIR=$C/triton
export TORCHINDUCTOR_CACHE_DIR=$C/inductor VLLM_CACHE_ROOT=$C/vllm
# Ray needs a node-local filesystem for its sockets: /tmp on the compute node, one dir per job
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$USER}}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p "$R1X4_ROOT/logs" "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$RAY_TMPDIR"
# written by setup_msi.sh after its kernel test: flash_attention_2 if flash-attn runs, else sdpa
[ -f "$R1X4_ROOT/attn_impl" ] && export R1X4_ATTN_IMPL=$(cat "$R1X4_ROOT/attn_impl")
# conda env: activate by prefix if it exists (module-free; MSI's miniforge is on PATH)
if [ -x "$R1X4_ENV/bin/python" ]; then
  command -v conda >/dev/null && { eval "$(conda shell.bash hook)"; conda activate "$R1X4_ENV"; } || export PATH="$R1X4_ENV/bin:$PATH"
fi
