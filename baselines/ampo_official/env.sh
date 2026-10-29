# Shared environment for the official AMPO-repo baselines (sourced by the sbatch files).
export HF_HOME=/scratch.global/kamel026/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_DISABLE_COMPILE_CACHE=1 PYTHONUNBUFFERED=1
export PATH=/scratch.global/kamel026/envs/r1x4/bin:$PATH RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID:-$$}
# official AMPO verl ahead of the kit's editable install, also for the Ray workers
export PYTHONPATH=/scratch.global/kamel026/ampo_official/ampo/verl:/scratch.global/kamel026/r1x4_runs/baselines/stubs${PYTHONPATH:+:$PYTHONPATH}
mkdir -p $RAY_TMPDIR
B=/scratch.global/kamel026/r1x4_runs/baselines
# set by Slurm; verl refuses to start when it is set together with CUDA_VISIBLE_DEVICES (same as the kit's run.py)
unset ROCR_VISIBLE_DEVICES
