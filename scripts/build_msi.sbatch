#!/bin/bash
#SBATCH -J r1x4-build
#SBATCH -p msismall
#SBATCH -c 32
#SBATCH --mem=96g
#SBATCH -t 08:00:00
#SBATCH -o /scratch.global/radke149/r1x4_runs/logs/build_%j.out
export HF_HOME=/scratch.global/radke149/hf-cache TOKENIZERS_PARALLELISM=false
cd /scratch.global/radke149/r1x4kit_1p7b_4b/data
/scratch.global/radke149/envs/ampo/bin/python -u build_datasets.py --out /scratch.global/radke149/r1x4_runs/data --workers 30 --sets ${1:-luffy21k,core,extended,r1qwen,qwen3}
