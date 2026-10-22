# source scripts/rtx_env.sh   -- environment for the RTX PRO 6000 Blackwell box (cs-u-ding6000pro).
# Keeps EVERYTHING off the small system disk / home: caches, Ray temp, HF models -> /mnt/data (fast local disk).
# /mnt/data (shared local disk) filled up on 2026-10-20 and killed a run: everything written goes to /mnt/data2 (NFS),
# except Ray/TMPDIR, which need a local filesystem (sockets): /dev/shm (RAM; / is also near full)
export R1X4_ROOT=${R1X4_ROOT:-/mnt/data2/radke149/r1x4}            # data/, ckpt/, logs/, tensorboard/
export R1X4_ENV=${R1X4_ENV:-r1x4}                                  # conda env name (created under /mnt/data2/radke149/conda)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}       # ONLY GPUs 0-3 are ours (Mon 19 - Fri 23 Oct)
C=${R1X4_CACHE:-/mnt/data2/radke149/cache}
export HF_HOME=$C/hf XDG_CACHE_HOME=$C/xdg PIP_CACHE_DIR=$C/pip TRITON_CACHE_DIR=$C/triton
export TORCHINDUCTOR_CACHE_DIR=$C/inductor VLLM_CACHE_ROOT=$C/vllm RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/ray_radke149} TMPDIR=${R1X4_TMPDIR:-/dev/shm/radke149_tmp}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p "$R1X4_ROOT" "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$RAY_TMPDIR" "$TMPDIR"
# written by setup_rtx.sh after its kernel tests: flash_attention_2 if flash-attn runs on sm_120, else sdpa
[ -f "$R1X4_ROOT/attn_impl" ] && export R1X4_ATTN_IMPL=$(cat "$R1X4_ROOT/attn_impl")
command -v conda >/dev/null && { eval "$(conda shell.bash hook)"; conda activate "$R1X4_ENV" 2>/dev/null || true; }
