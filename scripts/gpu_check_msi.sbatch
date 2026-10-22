#!/bin/bash
#SBATCH -J r1x4-gpucheck
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 01:00:00
#SBATCH -o r1x4_gpucheck_%j.out
# GPU kernel tests after setup_msi.sbatch: torch matmul, flash-attn vs SDPA, vLLM generate on Qwen3-1.7B.
# Downloads Qwen3-1.7B into $HF_HOME (compute nodes have internet). Writes $R1X4_ROOT/attn_impl.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/msi_env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'EOF'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16); print('matmul ok', float((x @ x).float().mean()))
EOF
ATTN=sdpa
if python -c "
import torch; from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16); o = flash_attn_func(q, q, q, causal=True)
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), q.transpose(1,2), q.transpose(1,2), is_causal=True).transpose(1,2)
assert (o - ref).abs().max() < 2e-2; print('flash-attn kernel OK')"; then ATTN=flash_attention_2
else
  echo "flash-attn unusable: installing the kit's pure-torch bert_padding shim, attention = SDPA"
  python -m pip uninstall -y flash-attn || true
  SITE=$(python -c 'import site; print(site.getsitepackages()[0])'); mkdir -p "$SITE/flash_attn"; cp scripts/flash_attn_shim/*.py "$SITE/flash_attn/"
fi
echo "$ATTN" > "$R1X4_ROOT/attn_impl"
python -c "from huggingface_hub import snapshot_download as d; print(d('Qwen/Qwen3-1.7B'))"
python - <<'EOF'
from vllm import LLM, SamplingParams
llm = LLM('Qwen/Qwen3-1.7B', gpu_memory_utilization=0.3, max_model_len=4096, enforce_eager=True)
o = llm.generate(['<|im_start|>user\nWhat is 7*6? Put the answer in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'],
                 SamplingParams(max_tokens=512, temperature=0.6))
print('vLLM OK:', o[0].outputs[0].text[-200:])
EOF
VRC=$?
python -c "
from verl.r1x4.reward import grade_texts; import inspect; print('grader import OK', inspect.signature(grade_texts))"
cd verl && python -m pytest -q ../r1x4_tests 2>&1 | tail -15
echo "GPU_CHECK_DONE attn=$ATTN vllm_rc=$VRC"
