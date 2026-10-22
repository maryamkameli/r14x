#!/usr/bin/env bash
# One-time env build on MSI. Counterpart of setup_rtx.sh (which targets Blackwell / torch 2.7 cu128).
# Run it as a job, NOT on a login node (they kill processes after 15 min of CPU):
#   sbatch scripts/setup_msi.sbatch            # CPU build (msismall)
#   sbatch scripts/gpu_check_msi.sbatch        # then: torch / flash-attn / vLLM kernel tests on 1 GPU
# Stack: python 3.10, vLLM 0.8.5 (torch 2.6.0 cu124), transformers 4.51.3, AMPO's verl 0.5.0.dev from this kit.
set -euo pipefail
KIT=$(cd "$(dirname "$0")/.." && pwd)
source "$KIT/scripts/msi_env.sh"
eval "$(conda shell.bash hook)"

if [ ! -x "$R1X4_ENV/bin/python" ]; then
  conda create -y -p "$R1X4_ENV" python=3.10
fi
conda activate "$R1X4_ENV"
python -m pip install -U pip wheel setuptools packaging ninja
# Two stages: vLLM first (it pins torch/xformers/opentelemetry), then the rest is resolved around those pins.
# One big resolve of everything at once backtracks forever (resolution-too-deep).
python -m pip install "vllm==0.8.5" "torch==2.6.0" "transformers==4.51.3"
python -m pip install -r "$KIT/requirements.txt"

# flash-attn: prebuilt wheel for torch 2.6 / cu12 / py3.10 (pip resolves it from PyPI's sdist + the release wheel URL).
# If it fails, fall back to the kit's pure-torch bert_padding shim + PyTorch SDPA.
FA_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -m pip install "$FA_WHL" || python -m pip install "flash-attn==2.7.4.post1" --no-build-isolation || echo "flash-attn install failed"

python -m pip install -e "$KIT/verl" --no-deps
python -m pip check || true   # informational

python - <<'EOF'
import importlib
for m in ['torch', 'vllm', 'transformers', 'ray', 'tensordict', 'hydra', 'omegaconf', 'math_verify', 'flash_attn']:
    try:
        mod = importlib.import_module(m); print(f'{m:14s}', getattr(mod, '__version__', '?'))
    except Exception as e:
        print(f'{m:14s} IMPORT FAILED: {e!r}')
import torch; print('torch cuda build', torch.version.cuda)
EOF
python -c "import verl, verl.r1x4.trainer, verl.r1x4.reward, verl.adaptive_mix_src.mix_fsdp_worker; print('verl kit imports OK')"
echo "SETUP DONE (CPU part). Next: sbatch scripts/gpu_check_msi.sbatch"
