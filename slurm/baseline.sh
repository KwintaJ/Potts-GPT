#!/bin/bash
# resubmits the L=16 measurements at beta_c with the fixed sampler: plain spin tokens
# as the baseline and 2x2 block tokens with the physics prior and Z_Q canonical form
# usage: bash slurm/baseline.sh
cd "$(dirname "$0")/.."

COMMON="--L 16 --n-layer 4 --n-embd 128 --batch-size 4096 --iterations 20000 --annealing-steps 10000"

for Q in 6 12 18; do
    sbatch slurm/train.sh $COMMON --Q $Q --name "Q=${Q}_base" --out "checkpoints/L16_Q${Q}_base.pt"
    sbatch slurm/train.sh $COMMON --Q $Q --patch 2 2 --ap --canonical \
        --name "Q=${Q}_p22_ap_canon" --out "checkpoints/L16_Q${Q}_p22_ap_canon.pt"
done
