#!/bin/bash
# reduced L=16 reference runs for an Apple Silicon laptop (no cuda), 1 layer, small
# batches, a few thousand iterations; two queues so that mps and cpu work in parallel
# usage: bash scripts/mac_baseline.sh [mps|cpu]
cd "$(dirname "$0")/.."
mkdir -p checkpoints
QUEUE=${1:-mps}
PROJECT=potts-mac-16
COMMON="--L 16 --n-layer 1 --n-embd 128 --iterations 2500 --annealing-steps 800 --lr 3e-4 --seed 0 --log-every 25 --wandb-project $PROJECT"

if [ "$QUEUE" = "mps" ]; then
    python train_potts_transformer.py $COMMON --device mps --batch-size 512 --Q 6 --patch 2 2 --ap --canonical \
        --name "Q=6_p22_ap_canon" --out checkpoints/mac_L16_Q6_p22_ap_canon.pt
    python train_potts_transformer.py $COMMON --device mps --batch-size 512 --Q 12 --patch 1 2 --ap --canonical \
        --name "Q=12_p12_ap_canon" --out checkpoints/mac_L16_Q12_p12_ap_canon.pt
    python train_potts_transformer.py $COMMON --device mps --batch-size 512 --Q 18 --patch 1 2 --ap --canonical \
        --name "Q=18_p12_ap_canon" --out checkpoints/mac_L16_Q18_p12_ap_canon.pt
else
    export OMP_NUM_THREADS=4
    python train_potts_transformer.py $COMMON --device cpu --batch-size 256 --Q 6 \
        --name "Q=6_base" --out checkpoints/mac_L16_Q6_base.pt
    python train_potts_transformer.py $COMMON --device cpu --batch-size 256 --Q 12 \
        --name "Q=12_base" --out checkpoints/mac_L16_Q12_base.pt
fi
