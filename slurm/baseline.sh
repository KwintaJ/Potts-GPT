#!/bin/bash
# resubmits the L=16 measurements at beta_c with the fixed sampler: plain spin tokens
# as the baseline and block tokens with the physics prior and Z_Q canonical form.
# 2x2 blocks are used for Q=6 only: the (batch, tokens, Q^4) logit tensor does not fit
# in GPU memory for Q >= 12 at this batch size, so those use 1x2 blocks
# usage: bash slurm/baseline.sh
cd "$(dirname "$0")/.."

COMMON="--L 16 --n-layer 4 --n-embd 128 --batch-size 4096 --iterations 20000 --annealing-steps 10000"

for Q in 6 12 18; do
    if [ "$Q" -eq 6 ]; then PATCH="2 2"; TAG="p22"; else PATCH="1 2"; TAG="p12"; fi
    sbatch slurm/train.sh $COMMON --Q $Q --name "Q=${Q}_base" --out "checkpoints/L16_Q${Q}_base.pt"
    sbatch slurm/train.sh $COMMON --Q $Q --patch $PATCH --ap --canonical \
        --name "Q=${Q}_${TAG}_ap_canon" --out "checkpoints/L16_Q${Q}_${TAG}_ap_canon.pt"
done
