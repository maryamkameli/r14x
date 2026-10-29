#!/bin/bash
#SBATCH -J r1x4-setup
#SBATCH -p msismall
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 03:00:00
#SBATCH -o r1x4_setup_%j.out
# CPU-only env build on MSI (login nodes kill long pip installs). Submit from the kit root: sbatch scripts/setup_msi.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-16}
bash scripts/setup_msi.sh
