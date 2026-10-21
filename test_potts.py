import math

import torch

from potts import beta_critical, energy2D, exact_free_energy
from transformer import GPT, GPTConfig


def all_configurations(L, Q):
    n = Q ** (L * L)
    idx = torch.arange(n)
    return torch.stack([(idx // Q**i) % Q for i in range(L * L)], dim=1)


def brute_force_log_z(L, Q, beta):
    energy = energy2D(all_configurations(L, Q), L, L).double()
    return torch.logsumexp(-beta * energy, dim=0).item()


def small_model(L, Q, seed=0):
    torch.manual_seed(seed)
    config = GPTConfig(vocab_size=Q, block_size=L * L, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config).eval()
    # scale the weights up so that conditionals really depend on the context
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(10)
    return model


def test_energy_of_uniform_state_counts_every_bond():
    L = 3
    state = torch.full((2, L * L), 1)
    assert energy2D(state, L, L).tolist() == [-2 * L * L, -2 * L * L]


def test_transfer_matrix_matches_enumeration():
    for L, Q in [(3, 2), (3, 3), (4, 2)]:
        beta = beta_critical(Q)
        expected = -brute_force_log_z(L, Q, beta) / (beta * L * L)
        assert abs(exact_free_energy(L, Q, beta) - expected) < 1e-9


def test_generate_uses_the_same_conditionals_as_the_full_forward():
    L, Q = 4, 3
    model = small_model(L, Q)
    torch.manual_seed(1)
    sample = model.generate(torch.randint(0, Q, (8, 1)), L * L - 1)
    assert sample.shape == (8, L * L)

    full = torch.log_softmax(model(sample[:, :-1]), dim=-1)
    caches = model.new_caches(8, L * L, sample.device)
    for pos in range(L * L - 1):
        step = torch.log_softmax(model(sample[:, pos : pos + 1], caches, pos)[:, -1], dim=-1)
        assert torch.allclose(step, full[:, pos], rtol=1e-3, atol=1e-3), f"mismatch at {pos}"


def test_generate_conditionals_depend_on_the_whole_prefix():
    L, Q = 4, 3
    model = small_model(L, Q)
    a = torch.tensor([[0, 1, 2, 0, 1]])
    b = torch.tensor([[2, 2, 2, 2, 1]])
    assert not torch.allclose(model(a)[:, -1], model(b)[:, -1])


def test_log_prob_is_normalised():
    L, Q = 2, 3
    model = small_model(L, Q)
    with torch.no_grad():
        log_q = model.log_prob(all_configurations(L, Q))
    assert abs(torch.logsumexp(log_q, dim=0).item()) < 1e-5


def test_sampling_frequencies_match_log_prob():
    L, Q = 2, 2
    model = small_model(L, Q)
    torch.manual_seed(2)
    n = 20000
    sample = model.generate(torch.randint(0, Q, (n, 1)), L * L - 1)
    codes = (sample * (Q ** torch.arange(L * L))).sum(dim=1)
    freq = torch.bincount(codes, minlength=Q ** (L * L)).double() / n
    with torch.no_grad():
        prob = model.log_prob(all_configurations(L, Q)).exp()
    # 4 sigma on the binomial error of every configuration
    assert torch.all((freq - prob).abs() < 4 * torch.sqrt(prob * (1 - prob) / n) + 1e-3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
