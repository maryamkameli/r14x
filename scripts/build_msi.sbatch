#!/bin/bash
#SBATCH -J r1x4-build
#SBATCH -p msismall
#SBATCH -c 32
#SBATCH --mem=96g
#SBATCH -t 08:00:00
#SBATCH -o build_%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source scripts/msi_env.sh   # paths: scripts/msi_env.sh (R1X4_ROOT, HF_HOME, env)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
cd data
python -u build_datasets.py --out $R1X4_ROOT/data --workers 30 --sets ${1:-luffy21k,core,extended,r1qwen,qwen3}
