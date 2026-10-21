# Potts-GPT

Autoregressive transformer (nanoGPT-style decoder) trained as a neural sampler for the
2D Q-state Potts model. Training minimises the variational free energy
`F_q = E_q[log q(s) + beta E(s)]` with the REINFORCE (score function) gradient,
following the Neural Markov Chain Monte Carlo / VAN line of work.

## Layout

- `transformer.py` – the model, `GPT.generate` (KV-cached sampling) and `GPT.log_prob`
- `potts.py` – energy, critical coupling and an exact transfer matrix free energy for small L
- `train_potts_transformer.py` – training loop, wandb logging, final importance sampled F
- `test_potts.py` – sampler/energy/exact solution consistency tests
- `slurm/train.sh` – PLGrid (Athena, A100) batch script

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Training

```bash
# defaults: L=16, Q=6, beta = beta_c, 4 layers, batch 4096, wandb online
python train_potts_transformer.py --L 16 --Q 6

# quick local check against the exact solution (CPU / Apple MPS)
python train_potts_transformer.py --L 4 --Q 3 --n-layer 2 --n-embd 64 --batch-size 512 \
    --iterations 600 --annealing-steps 200 --lr 1e-3 --exact --wandb offline
```

`python train_potts_transformer.py --help` lists every option. The run logs `ess`,
`free_energy` (variational bound per site), `M` and `beta`, and prints the importance
sampled free energy at the end.
