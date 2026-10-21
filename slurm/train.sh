#!/bin/bash
#SBATCH --job-name=potts-train
#SBATCH -A plgnewneuralsamplers-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --time=24:00:00

# usage: sbatch slurm/train.sh --L 16 --Q 6 [more train_potts_transformer.py options]

module load CUDA/12.8.0
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/.venv/bin/activate"

python "$REPO/train_potts_transformer.py" "$@"
