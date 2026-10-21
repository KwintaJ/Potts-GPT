###################################################
# Potts sampler: GPT over patch tokens            #
#                                                 #
# Jan Kwinta                                      #
# October 2026                                    #
###################################################

import torch
import torch.nn as nn
from torch.nn import functional as F

from lattice import BondCounter, CanonicalMask, PatchCodec, canonicalize, log_relabelings
from transformer import GPT, GPTConfig


class PottsSampler(nn.Module):
    """autoregressive distribution q(s) over L x L Potts configurations

    Options:
    - patch (r, c): spins are grouped into r x c patches, one token each
    - approximate: add beta times the satisfied-bond count of every candidate patch
      to the logits, so the network only learns the correction to the local Boltzmann
      weight (Białas et al., arXiv:2604.27738)
    - canonical: the network models only the Z_Q canonical representative (states
      relabelled by first appearance); q(s) of a raw configuration is the canonical
      probability divided by the size of its orbit. Sampling returns raw configurations.
    """

    def __init__(self, L, Q, patch=(1, 1), approximate=False, canonical=False, **gpt_kwargs):
        super().__init__()
        self.L, self.Q = L, Q
        self.codec = PatchCodec(L, Q, *patch)
        self.gpt = GPT(GPTConfig(block_size=self.codec.n_patches, vocab_size=self.codec.vocab,
                                 **gpt_kwargs))
        self.bonds = BondCounter(self.codec) if approximate else None
        self.mask = CanonicalMask(self.codec) if canonical else None
        # the training forward may be swapped for a torch.compile'd version
        self.train_gpt = self.gpt
        all_patches = torch.arange(self.codec.n_patches)
        self.register_buffer("all_patches", all_patches, persistent=False)

    @property
    def canonical(self):
        return self.mask is not None

    def log_prob(self, spins, beta):
        """log q(spins) for (B, L*L) configurations"""
        B = spins.shape[0]
        if self.canonical:
            ordered, n_distinct = canonicalize(self.codec.generation_order(spins), self.Q)
            spins = self.codec.from_generation_order(ordered)
        tokens = self.codec.encode(spins)

        bos = torch.full((B, 1), self.gpt.bos, dtype=torch.long, device=spins.device)
        logits = self.train_gpt(torch.cat([bos, tokens[:, :-1]], dim=1)).float()
        if self.bonds is not None:
            logits = logits + beta * self.bonds(spins.view(B, self.L, self.L), self.all_patches)
        if self.canonical:
            allowed = self.mask.valid[self.mask.labels_used_before(tokens)]
            logits = logits.masked_fill(~allowed, float("-inf"))

        log_q = F.log_softmax(logits, dim=-1).gather(2, tokens.unsqueeze(-1)).squeeze(-1).sum(1)
        if self.canonical:
            log_q = log_q - log_relabelings(n_distinct, self.Q)
        return log_q

    @torch.no_grad()
    def generate(self, batch_size, beta):
        """sample (batch_size, L*L) configurations from q"""
        codec, device = self.codec, self.all_patches.device
        lattice = torch.zeros(batch_size, self.L, self.L, dtype=torch.long, device=device)
        caches = self.gpt.new_caches(batch_size, codec.n_patches, device,
                                     self.gpt.transformer.wte.weight.dtype)
        previous = torch.full((batch_size, 1), self.gpt.bos, dtype=torch.long, device=device)
        labels_used = torch.zeros(batch_size, dtype=torch.long, device=device)

        for p in range(codec.n_patches):
            logits = self.gpt(previous, caches, p)[:, -1, :].float()
            if self.bonds is not None:
                logits = logits + beta * self.bonds(lattice, self.all_patches[p : p + 1])[:, 0]
            if self.canonical:
                logits = logits.masked_fill(~self.mask.valid[labels_used], float("-inf"))
            previous = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            codec.write_patch(lattice, p, previous[:, 0])
            if self.canonical:
                labels_used = torch.maximum(labels_used, self.mask.patch_max[previous[:, 0]] + 1)

        spins = lattice.view(batch_size, -1)
        if self.canonical:
            # a uniformly random relabelling turns the canonical sample into a raw one
            perm = torch.argsort(torch.rand(batch_size, self.Q, device=device), dim=1)
            spins = perm.gather(1, spins)
        return spins
