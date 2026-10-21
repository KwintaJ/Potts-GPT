# Potts-GPT

Autoregressive transformer (nanoGPT-style decoder) trained as a neural sampler for the
2D Q-state Potts model. Training minimises the variational free energy
`F_q = E_q[log q(s) + beta E(s)]` with the REINFORCE (score function) gradient,
following the Neural Markov Chain Monte Carlo / VAN line of work.

## Layout

- `transformer.py` – nanoGPT decoder over patch tokens with a KV cache for sampling
- `lattice.py` – patch tokenisation, bond counting for the physics prior, Z_Q canonical
  relabelling
- `sampler.py` – `PottsSampler`: `generate` and `log_prob` over spin configurations,
  combining the three options below
- `potts.py` – energy, critical coupling and an exact transfer matrix free energy for small L
- `train_potts_transformer.py` – training loop, wandb logging, final importance sampled F
- `test_potts.py` – sampler/energy/exact solution consistency tests
- `slurm/train.sh` – PLGrid (Athena, A100) batch script, `slurm/baseline.sh` submits the
  L=16 measurement set

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

Sampler options:

- `--patch R C` – one token per R x C block of spins (vocabulary `Q**(R*C)`), shortens
  the sequence from `L*L` to `L*L/(R*C)`
- `--ap` – approximate probabilities: `beta` times the number of satisfied bonds of every
  candidate block (internal and to already generated neighbours) is added to the logits,
  so the network learns only the correction to the local Boltzmann weight
- `--canonical` – the network models only the Z_Q canonical form (states relabelled by
  first appearance, invalid labels masked); `log_prob` is exactly invariant under
  state permutations and samples are relabelled uniformly at random

```bash
python train_potts_transformer.py --L 32 --Q 12 --patch 2 2 --ap --canonical
```

`python train_potts_transformer.py --help` lists every option. The run logs `ess`,
`free_energy` (variational bound per site), `M` and `beta`, and prints the importance
sampled free energy at the end.
