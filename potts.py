import math

import torch


def beta_critical(Q):
    """critical coupling of the 2D Q-state Potts model on the square lattice"""
    return math.log(1 + math.sqrt(Q))


def energy2D(S, r, c):
    """energy -sum_<ij> delta(s_i, s_j) of flattened r x c periodic lattices, shape (B,)"""
    state = S.reshape(-1, r, c)
    return -torch.sum(torch.roll(state, 1, 1) == state, dim=(1, 2)) - torch.sum(
        torch.roll(state, 1, 2) == state, dim=(1, 2)
    )


def exact_free_energy(L, Q, beta):
    """free energy per site -log(Z) / (beta L^2) of the periodic L x L lattice

    Computed with the row transfer matrix, so it is only feasible while Q**L stays in the
    low thousands (e.g. Q=6 up to L=5, Q=3 up to L=8). Used to validate the sampler.
    """
    n = Q**L
    idx = torch.arange(n)
    rows = torch.stack([(idx // Q**i) % Q for i in range(L)], dim=1)  # (n, L)

    # bonds inside a row and bonds between two stacked rows
    row_bonds = (rows == rows.roll(1, dims=1)).sum(dim=1).double()
    inter = torch.zeros(n, n, dtype=torch.float64)
    for i in range(L):
        inter += (rows[:, None, i] == rows[None, :, i]).double()

    # symmetric transfer matrix: half of each row's internal weight on both sides
    log_t = beta * (inter + 0.5 * (row_bonds[:, None] + row_bonds[None, :]))
    shift = log_t.max()
    lam = torch.linalg.eigvalsh(torch.exp(log_t - shift))

    # Z = Tr T^L = sum lam_i^L, keep the signs for odd L
    log_terms = L * (torch.log(lam.abs().clamp_min(1e-300)) + shift)
    positive = lam > 0
    log_pos = torch.logsumexp(log_terms[positive], dim=0)
    if positive.all() or L % 2 == 0:
        log_z = torch.logsumexp(log_terms, dim=0) if L % 2 == 0 else log_pos
    else:
        log_neg = torch.logsumexp(log_terms[~positive], dim=0)
        log_z = log_pos + torch.log1p(-torch.exp(log_neg - log_pos))

    return -log_z.item() / (beta * L * L)
