###################################################
# GPT model for Potts Physics                     #
#                                                 #
# Based on nanoGPT                                #
# https://github.com/karpathy/nanoGPT             #
#                                                 #
# Jan Kwinta                                      #
# March 2026                                      #
###################################################

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # sequence length (number of patch tokens)
    vocab_size: int = 2     # number of patch states
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, cache=None, pos=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        drop = self.dropout if self.training else 0.0

        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        else:
            # single-token step: store k, v at `pos` and attend over everything up to it,
            # so the sampled token sees the same context as in the full causal forward
            assert T == 1
            k_cache, v_cache = cache
            k_cache[:, :, pos] = k[:, :, 0]
            v_cache[:, :, pos] = v[:, :, 0]
            y = F.scaled_dot_product_attention(
                q, k_cache[:, :, : pos + 1], v_cache[:, :, : pos + 1], dropout_p=drop
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cache=None, pos=None):
        x = x + self.attn(self.ln_1(x), cache, pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """decoder over token sequences; the extra token `bos` starts every sequence

    Feeding [bos, t_0, ..., t_{T-2}] returns the logits of t_0, ..., t_{T-1}.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.bos = config.vocab_size

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size + 1, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size + 1, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, caches=None, pos=None):
        """next-token logits (B, T, vocab) for every position of `idx`

        Without caches this is the usual causal forward over the whole sequence.
        With caches `idx` holds a single token which sits at position `pos`.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        if caches is None:
            positions = torch.arange(0, t, dtype=torch.long, device=device)
            caches = [None] * len(self.transformer.h)
        else:
            assert t == 1
            positions = torch.full((1,), pos, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(positions)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block, cache in zip(self.transformer.h, caches):
            x = block(x, cache, pos)
        x = self.transformer.ln_f(x)
        # the tied embedding also holds the bos row, which is never a prediction target
        return self.lm_head(x)[:, :, : self.config.vocab_size]

    def new_caches(self, batch_size, length, device, dtype=torch.float32):
        hs = self.config.n_embd // self.config.n_head
        shape = (batch_size, self.config.n_head, length, hs)
        return [
            (torch.zeros(shape, device=device, dtype=dtype),
             torch.zeros(shape, device=device, dtype=dtype))
            for _ in self.transformer.h
        ]
