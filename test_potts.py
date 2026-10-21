import math

import pytest
import torch

from lattice import BondCounter, PatchCodec, canonicalize, log_relabelings
from potts import beta_critical, energy2D, exact_free_energy
from sampler import PottsSampler


def all_configurations(L, Q):
    n = Q ** (L * L)
    idx = torch.arange(n)
    return torch.stack([(idx // Q**i) % Q for i in range(L * L)], dim=1)


def brute_force_log_z(L, Q, beta):
    energy = energy2D(all_configurations(L, Q), L, L).double()
    return torch.logsumexp(-beta * energy, dim=0).item()


def small_sampler(L, Q, seed=0, **kwargs):
    torch.manual_seed(seed)
    model = PottsSampler(L, Q, n_layer=2, n_head=2, n_embd=32, **kwargs).eval()
    # scale the weights up so that conditionals really depend on the context
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(10)
    return model


VARIANTS = [
    dict(),
    dict(patch=(2, 1)),
    dict(patch=(1, 2), approximate=True),
    dict(patch=(2, 2), canonical=True),
    dict(patch=(2, 1), approximate=True, canonical=True),
]


def test_energy_of_uniform_state_counts_every_bond():
    L = 3
    state = torch.full((2, L * L), 1)
    assert energy2D(state, L, L).tolist() == [-2 * L * L, -2 * L * L]


def test_transfer_matrix_matches_enumeration():
    for L, Q in [(3, 2), (3, 3), (4, 2)]:
        beta = beta_critical(Q)
        expected = -brute_force_log_z(L, Q, beta) / (beta * L * L)
        assert abs(exact_free_energy(L, Q, beta) - expected) < 1e-9


@pytest.mark.parametrize("r,c", [(1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (4, 4)])
def test_patch_codec_roundtrip(r, c):
    L, Q = 4, 3
    codec = PatchCodec(L, Q, r, c)
    spins = torch.randint(0, Q, (5, L * L))
    assert torch.equal(codec.decode(codec.encode(spins)), spins)
    assert torch.equal(codec.from_generation_order(codec.generation_order(spins)), spins)


@pytest.mark.parametrize("r,c", [(1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (1, 4), (4, 4)])
def test_bond_counts_of_chosen_patches_add_up_to_minus_energy(r, c):
    L, Q = 4, 3
    codec = PatchCodec(L, Q, r, c)
    bonds = BondCounter(codec)
    spins = torch.randint(0, Q, (6, L * L))
    tokens = codec.encode(spins)
    counts = bonds(spins.view(-1, L, L), torch.arange(codec.n_patches))
    chosen = counts.gather(2, tokens.unsqueeze(-1)).squeeze(-1).sum(1)
    assert torch.equal(chosen.long(), -energy2D(spins, L, L))


def test_canonicalize_relabels_by_first_appearance():
    ordered = torch.tensor([[2, 2, 0, 2, 1], [1, 1, 1, 1, 1]])
    canonical, n_distinct = canonicalize(ordered, Q=3)
    assert canonical.tolist() == [[0, 0, 1, 0, 2], [0, 0, 0, 0, 0]]
    assert n_distinct.tolist() == [3, 1]
    assert torch.allclose(log_relabelings(n_distinct, 3), torch.tensor([math.log(6), math.log(3)]))


@pytest.mark.parametrize("kwargs", VARIANTS)
def test_generate_uses_the_same_conditionals_as_log_prob(kwargs):
    L, Q = 4, 3
    model = small_sampler(L, Q, **kwargs)
    beta = beta_critical(Q)
    torch.manual_seed(1)
    sample = model.generate(8, beta)
    assert sample.shape == (8, L * L)
    # replay the sampling step by step through the training forward
    log_q = model.log_prob(sample, beta)
    assert torch.isfinite(log_q).all()
    assert (log_q <= 1e-5).all()


@pytest.mark.parametrize("kwargs", VARIANTS)
def test_log_prob_is_normalised(kwargs):
    L, Q = 2, 3
    model = small_sampler(L, Q, **kwargs)
    with torch.no_grad():
        log_q = model.log_prob(all_configurations(L, Q), beta_critical(Q))
    assert abs(torch.logsumexp(log_q, dim=0).item()) < 1e-4


@pytest.mark.parametrize("kwargs", VARIANTS)
def test_sampling_frequencies_match_log_prob(kwargs):
    L, Q = 2, 2
    model = small_sampler(L, Q, **kwargs)
    beta = beta_critical(Q)
    torch.manual_seed(2)
    n = 20000
    sample = model.generate(n, beta)
    codes = (sample * (Q ** torch.arange(L * L))).sum(dim=1)
    freq = torch.bincount(codes, minlength=Q ** (L * L)).double() / n
    with torch.no_grad():
        prob = model.log_prob(all_configurations(L, Q), beta).exp().double()
    # 4 sigma on the binomial error of every configuration
    assert torch.all((freq - prob).abs() < 4 * torch.sqrt(prob * (1 - prob) / n) + 1e-3)


def test_log_prob_is_relabelling_invariant_when_canonical():
    L, Q = 4, 3
    model = small_sampler(L, Q, patch=(2, 2), canonical=True, approximate=True)
    beta = beta_critical(Q)
    spins = torch.randint(0, Q, (6, L * L))
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        assert torch.allclose(model.log_prob(spins, beta), model.log_prob(perm[spins], beta),
                              atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
