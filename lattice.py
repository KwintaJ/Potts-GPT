###################################################
# Lattice <-> token helpers for the Potts sampler #
#                                                 #
# Jan Kwinta                                      #
# October 2026                                    #
###################################################

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class PatchCodec(nn.Module):
    """maps L x L configurations to sequences of r x c patch tokens in raster order

    A patch token is the base-Q number of its spins read row by row. Tokens are
    generated patch by patch in raster order, spins inside a patch are ordered the same
    way; that is the "generation order" used by the canonical relabelling.
    """

    def __init__(self, L, Q, r=1, c=1):
        super().__init__()
        assert L % r == 0 and L % c == 0, "patch must tile the lattice"
        self.L, self.Q, self.r, self.c = L, Q, r, c
        self.n_rows, self.n_cols = L // r, L // c
        self.n_patches = self.n_rows * self.n_cols
        self.n_spins = r * c
        self.vocab = Q ** (r * c)

        weights = Q ** torch.arange(r * c)
        k = torch.arange(self.vocab)
        states = torch.stack([(k // weights[i]) % Q for i in range(r * c)], dim=1)
        self.register_buffer("weights", weights, persistent=False)
        self.register_buffer("states", states.view(self.vocab, r, c), persistent=False)

        p = torch.arange(self.n_patches)
        self.register_buffer("rows", p // self.n_cols, persistent=False)
        self.register_buffer("cols", p % self.n_cols, persistent=False)

    def spins_to_patches(self, spins):
        B = spins.shape[0]
        s = spins.view(B, self.n_rows, self.r, self.n_cols, self.c)
        return s.permute(0, 1, 3, 2, 4).reshape(B, self.n_patches, self.r, self.c)

    def patches_to_spins(self, patches):
        B = patches.shape[0]
        s = patches.view(B, self.n_rows, self.n_cols, self.r, self.c)
        return s.permute(0, 1, 3, 2, 4).reshape(B, self.L * self.L)

    def encode(self, spins):
        """(B, L*L) spins -> (B, P) tokens"""
        flat = self.spins_to_patches(spins).reshape(spins.shape[0], self.n_patches, -1)
        return (flat * self.weights).sum(dim=-1)

    def decode(self, tokens):
        """(B, P) tokens -> (B, L*L) spins"""
        return self.patches_to_spins(self.states[tokens])

    def generation_order(self, spins):
        return self.spins_to_patches(spins).reshape(spins.shape[0], -1)

    def from_generation_order(self, ordered):
        return self.patches_to_spins(ordered.view(ordered.shape[0], self.n_patches, self.r, self.c))

    def write_patch(self, lattice, p, tokens):
        """store the patch states `tokens` (B,) at patch p of the (B, L, L) lattice in place"""
        B = lattice.shape[0]
        row, col = self.rows[p].item(), self.cols[p].item()
        lattice.view(B, self.n_rows, self.r, self.n_cols, self.c)[:, row, :, col, :] = self.states[tokens]


class BondCounter(nn.Module):
    """satisfied bonds a candidate patch state adds, given the patches generated before it

    Counts the bonds inside the patch plus the bonds to neighbouring spins that are
    already known in raster order (periodic boundaries included), so that summing the
    counts of the chosen states over all patches gives -E of the configuration. Adding
    beta times this count to the logits is the "approximate probabilities" trick of
    Białas et al. (arXiv:2604.27738).
    """

    def __init__(self, codec: PatchCodec):
        super().__init__()
        st, Q, L = codec.states, codec.Q, codec.L
        r, c = codec.r, codec.c
        rows, cols = codec.rows, codec.cols

        internal = (st[:, 1:, :] == st[:, :-1, :]).sum(dim=(1, 2))
        internal = internal + (st[:, :, 1:] == st[:, :, :-1]).sum(dim=(1, 2))
        # the wrap-around bond is internal when the patch spans the whole lattice
        if codec.n_rows == 1:
            internal = internal + (st[:, 0, :] == st[:, -1, :]).sum(dim=1)
        if codec.n_cols == 1:
            internal = internal + (st[:, :, 0] == st[:, :, -1]).sum(dim=1)
        self.register_buffer("internal", internal.float(), persistent=False)

        # for every edge of the patch: which lattice sites border it, whether those are
        # already generated, and a (Q+1, K, n) lookup table of matches; row Q is used for
        # unknown neighbours and matches nothing
        self.edge_names = ("top", "bottom", "left", "right")
        span_c = torch.arange(c)
        span_r = torch.arange(r)
        geometry = {
            "top": ((rows * r - 1) % L, cols[:, None] * c + span_c, rows > 0, st[:, 0, :]),
            "bottom": ((rows * r + r) % L, cols[:, None] * c + span_c,
                       (rows == codec.n_rows - 1) & (codec.n_rows > 1), st[:, -1, :]),
            "left": (rows[:, None] * r + span_r, (cols * c - 1) % L, cols > 0, st[:, :, 0]),
            "right": (rows[:, None] * r + span_r, (cols * c + c) % L,
                      (cols == codec.n_cols - 1) & (codec.n_cols > 1), st[:, :, -1]),
        }
        for name, (site_rows, site_cols, known, edge) in geometry.items():
            table = F.one_hot(edge, Q).permute(2, 0, 1).float()  # (Q, K, n)
            table = torch.cat([table, torch.zeros_like(table[:1])], dim=0)
            self.register_buffer(f"{name}_rows", site_rows, persistent=False)
            self.register_buffer(f"{name}_cols", site_cols, persistent=False)
            self.register_buffer(f"{name}_known", known, persistent=False)
            self.register_buffer(f"{name}_table", table, persistent=False)
        self.Q = Q

    def forward(self, lattice, patch_ids):
        """(B, L, L) spins known so far, (P',) patch ids -> (B, P', K) bond counts"""
        B = lattice.shape[0]
        out = self.internal.expand(B, len(patch_ids), -1)
        for name in self.edge_names:
            site_rows = getattr(self, f"{name}_rows")[patch_ids]
            site_cols = getattr(self, f"{name}_cols")[patch_ids]
            known = getattr(self, f"{name}_known")[patch_ids]
            table = getattr(self, f"{name}_table")
            if site_rows.dim() == 1:
                site_rows = site_rows[:, None]
            else:
                site_cols = site_cols[:, None]
            strip = lattice[:, site_rows, site_cols]  # (B, P', n)
            strip = torch.where(known[None, :, None], strip, self.Q)
            for j in range(strip.shape[-1]):
                out = out + table[:, :, j][strip[:, :, j]]
        return out


def canonicalize(ordered, Q):
    """relabel states by order of first appearance along dim 1

    Returns the relabelled sequence and the number of distinct states per row. The
    Potts energy is invariant under relabelling, so a model only needs to learn the
    canonical representative of every Z_Q orbit.
    """
    B, N = ordered.shape
    first = torch.full((B, Q), N, dtype=torch.long, device=ordered.device)
    positions = torch.arange(N, device=ordered.device).expand(B, N)
    first.scatter_reduce_(1, ordered, positions, reduce="amin")
    new_label = torch.argsort(torch.argsort(first, dim=1), dim=1)
    return new_label.gather(1, ordered), (first < N).sum(dim=1)


def log_relabelings(n_distinct, Q):
    """log of the number of configurations in the Z_Q orbit of a canonical one, Q!/(Q-m)!"""
    m = n_distinct.double()
    return (torch.lgamma(torch.tensor(Q + 1.0, dtype=torch.float64)) - torch.lgamma(Q - m + 1)).float()


class CanonicalMask(nn.Module):
    """which patch states keep a sequence canonical, given how many labels were used so far"""

    def __init__(self, codec: PatchCodec):
        super().__init__()
        flat = codec.states.reshape(codec.vocab, -1)
        Q = codec.Q
        valid = torch.zeros(Q + 1, codec.vocab, dtype=torch.bool)
        for m in range(Q + 1):
            used = torch.full((codec.vocab,), m)
            ok = torch.ones(codec.vocab, dtype=torch.bool)
            for j in range(flat.shape[1]):
                label = flat[:, j]
                ok &= label <= used
                used = torch.maximum(used, label + 1)
            valid[m] = ok
        self.register_buffer("valid", valid, persistent=False)
        self.register_buffer("patch_max", flat.amax(dim=1), persistent=False)

    def labels_used_before(self, tokens):
        """(B, P) canonical tokens -> (B, P) number of labels used before each patch"""
        running = torch.cummax(self.patch_max[tokens], dim=1).values + 1
        return torch.cat([torch.zeros_like(running[:, :1]), running[:, :-1]], dim=1)
