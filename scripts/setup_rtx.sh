#!/usr/bin/env bash
# One-time setup on the RTX PRO 6000 Blackwell box (sm_120, CUDA 12.8+ driver). Run inside tmux:
#   cd /mnt/data/radke149/r1x4kit && bash scripts/setup_rtx.sh 2>&1 | tee /mnt/data/radke149/setup.log
# Blackwell needs torch>=2.7 built for CUDA 12.8 and vLLM>=0.9 (the old LUFFY stack, torch 2.4 / vLLM 0.6.3, cannot run).
# Stack: vLLM 0.9.2 (pins torch 2.7.0+cu128), transformers 4.51.3 (Qwen3), AMPO's verl 0.5.0.dev from this kit.
set -euo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd)
source "$KIT/scripts/rtx_env.sh"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv
echo ">>> Check above that GPUs 0-3 are idle before training (shared box)."

if ! conda env list | grep -q "^$R1X4_ENV "; then
  conda create -y -n "$R1X4_ENV" python=3.11
fi
eval "$(conda shell.bash hook)"; conda activate "$R1X4_ENV"
python -m pip install -U pip wheel setuptools packaging ninja
# PyPI's plain torch 2.7.x wheels are cu126 (no sm_120 kernels): the cu128 index supplies torch X+cu128 (sorts above X).
CU128=https://download.pytorch.org/whl/cu128
python -m pip install "vllm==${VLLM_VERSION:-0.9.2}" --extra-index-url "$CU128"
if [ "$(python -c 'import torch; print(torch.version.cuda or "")')" != 12.8 ]; then  # env built before this fix: swap the build
  PINS=$(python -c 'import importlib.metadata as m
for p in ["torch", "torchvision", "torchaudio", "xformers"]:
    try: print(p + "==" + m.version(p).split("+")[0])
    except m.PackageNotFoundError: pass')
  python -m pip install --force-reinstall --no-deps $PINS --index-url "$CU128"
  python -m pip install $(echo "$PINS" | sed '1s/$/+cu128/') --extra-index-url "$CU128"  # cu128 nvidia-* runtime libs
fi
# hydra-core 1.3 / omegaconf 2.3 need antlr4 runtime 4.9.*: with math-verify's antlr4_13_2 extra pip backtracks to
# hydra-core 0.11 + omegaconf 1.4 (no omegaconf.MISSING, verl fails to import). latex2sympy2_extended ships a 4.9.3 parser.
python -m pip install "transformers==4.51.3" "tensordict==0.8.3" "ray[default]==2.46.0" "torchdata==0.11.0" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "antlr4-python3-runtime==4.9.3" codetiming pylatexenc "math-verify[antlr4_9_3]==0.8.0" latex2sympy2_extended datasets pyarrow pandas peft \
  accelerate tensorboard "protobuf>=5.28" dill pybind11 psutil einops "numpy<2.3"
python -m pip install -e "$KIT/verl" --no-deps
# torch 2.7.0+cu128 pulls nvidia-nccl-cu12 2.26.2, a CUDA 12.2 build: every collective on sm_120 dies with
# "illegal memory access" (P2P/SHM transports). The .post1 rebuild (CUDA 12.8) of the same version works.
python -m pip install --no-deps "nvidia-nccl-cu12==2.26.2.post1"

python - <<'EOF'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0),
      'capability', torch.cuda.get_device_capability(0), 'arch list', torch.cuda.get_arch_list())
assert 'sm_120' in torch.cuda.get_arch_list() or torch.cuda.get_device_capability(0) < (12, 0), 'torch build has no sm_120 kernels'
x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16); print('matmul ok', float((x @ x).float().mean()))
EOF

# flash-attn: a prebuilt wheel first, then a source build for sm_120; if neither runs, a pure-torch shim of the 4 padding
# helpers verl imports (bert_padding) + attention via PyTorch SDPA (R1X4_ATTN_IMPL=sdpa, use_remove_padding=False).
ATTN=sdpa
if python -m pip install "flash-attn==2.8.0.post2" --no-build-isolation \
   || TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=${MAX_JOBS:-16} python -m pip install "flash-attn==2.8.0.post2" --no-build-isolation --no-binary flash-attn; then
  if python -c "
import torch; from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16); o = flash_attn_func(q, q, q, causal=True)
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), q.transpose(1,2), q.transpose(1,2), is_causal=True).transpose(1,2)
assert (o - ref).abs().max() < 2e-2; print('flash-attn kernel OK on', torch.cuda.get_device_name(0))"; then ATTN=flash_attention_2; fi
fi
if [ "$ATTN" = sdpa ]; then
  python -m pip uninstall -y flash-attn || true
  SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
  mkdir -p "$SITE/flash_attn"; cp "$KIT/scripts/flash_attn_shim/"*.py "$SITE/flash_attn/"
  echo "flash-attn unusable on this GPU: installed the pure-torch bert_padding shim, attention = SDPA"
fi
echo "$ATTN" > "$R1X4_ROOT/attn_impl"

# vLLM smoke test (sm_120 kernels, V1 engine) on the small student
python - <<'EOF'
from vllm import LLM, SamplingParams
llm = LLM('Qwen/Qwen3-1.7B', gpu_memory_utilization=0.3, max_model_len=4096, enforce_eager=True)
o = llm.generate(['<|im_start|>user\nWhat is 7*6? Put the answer in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'],
                 SamplingParams(max_tokens=512, temperature=0.6))
print('vLLM OK:', o[0].outputs[0].text[-200:])
EOF
python -c "import verl, verl.r1x4.trainer, verl.adaptive_mix_src.mix_fsdp_worker; from torch.utils.tensorboard import SummaryWriter; print('verl kit + tensorboard imports OK')"
echo "SETUP DONE: attention backend = $ATTN"
