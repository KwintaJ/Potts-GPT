###################################################
# GPT training script for Potts Physics           #
#                                                 #
# Jan Kwinta                                      #
# March 2026                                      #
###################################################

# ML imports
import os
import torch
import numpy as np
from dataclasses import asdict
from functools import partial
from scipy.special import logsumexp

# logging
import wandb
from tqdm import tqdm

# physics engine, GPT model
from potts import energy2D as energy_nD
from transformer import GPT, GPTConfig


# --- hardware ---
torch.set_float32_matmul_precision("high")
dev = "cuda"

# --- physics ---
L = 16           # lattice size
vocab_size = 18  # Q
beta = 0.44

# --- model hyperparams ---
n_layer = 1 
n_head = 4
n_embd = 128
dropout = 0.0
bias = True
batch_size = 1024 
D4_symmetry = True

# --- training hyperparams ---
iterations = 40000
lr = 0.001
beta1 = 0.9
beta2 = 0.95
clip_grad = 1.0

# --- GPT config ---
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=L * L,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    batch_size=batch_size,
)

# --- complie model ---
model = GPT(config).to(dev)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2))
batched_bincount = torch.vmap(partial(torch.bincount, minlength=config.vocab_size))
model = torch.compile(model)

# --- WandB config ---
config_keys = [k for k in globals().items() if not k[0].startswith("_") and isinstance(k[1], (int, float, bool, str))]
config_wb = {k[0]: k[1] for k in config_keys}

wandb_run = wandb.init(
        entity="jankwinta-ju",
        project=f"potts-train-{L}",
        name=f"Q={vocab_size}_symmetry",
        config=config_wb,
)



# --- log probability ---
def log_prob(model, sample):
    logits, _ = model(sample)

    probs = torch.softmax(logits, dim=-1)
    out = probs[:, :-1].gather(2, sample[:, 1:].unsqueeze(-1)).squeeze()
    return torch.sum(out.log(), dim=-1) - np.log(config.vocab_size)

# --- D4_symmetry ---
def apply_symmetry(sample, L, Q):
    B = sample.shape[0]
    s = sample.view(B, L, L)
    
    s = torch.rot90(s, k=torch.randint(0, 4, (1,)).item(), dims=(1, 2))
    if torch.rand(1) > 0.5:
        s = torch.flip(s, dims=[2])
        
    perm = torch.randperm(Q, device=sample.device)
    s = perm[s]
    
    return s.reshape(B, -1)



#############################################
try:
    for i in tqdm(range(iterations)):
        # --- training ---
        beta_train = beta

        model.eval()
        with torch.no_grad():
            start = torch.randint(0, vocab_size, (batch_size, 1), device=dev)
            sample_raw = model._orig_mod.generate(start, L * L - 1)

        if D4_symmetry:
            sample = apply_symmetry(sample_raw, L, vocab_size)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_p = log_prob(model, sample)
            energy = energy_nD(sample, L, L)
    
            loss = log_p.float() + energy.float() * beta_train

        loss_detached = loss.detach()
        loss_reinforce = torch.mean((loss_detached - loss_detached.mean()) * log_p.float())

        loss_reinforce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- log ----
        if (i + 1) % 10 == 0:
            # effective sample size
            log_sum_w = torch.logsumexp(-loss, 0)
            log_sum_w_sq = torch.logsumexp(-2 * loss, 0)
            ess = torch.exp(2 * log_sum_w - log_sum_w_sq).item() / batch_size

            # free energy
            f_val = loss.mean().item() / (beta_train * L * L)
            f_err = loss.std().item() / (beta_train * L * L * np.sqrt(batch_size))

            # magnetisation
            m_val = batched_bincount(sample).amax(dim=1).float().mean().item() / (L * L)

            print(
                f"ESS: {ess:.4f} | "
                f"Free Energy: {f_val:.6f} ± {f_err:.6f} | "
                f"M: {m_val:.4f}"
            )
            wandb_run.log(dict(ess=ess, free_energy=f_val, M=m_val), i)

except KeyboardInterrupt:
    pass

finally:
    # --- validation ---
    with torch.inference_mode():
        loss_arr = []
        for i in range(10):
            start = torch.randint(0, config.vocab_size, (batch_size, 1), device=dev)
            sample = model.generate(start, L * L - 1)

            log_p = log_prob(model, sample)
            energy = energy_nD(sample, L, L)
            loss = log_p + energy * beta
            loss_arr.append(loss.cpu())

        loss_np = np.array(loss_arr).flatten()

        f = (logsumexp(-loss_np) - np.log(len(loss_np))) / beta / L / L
        df = np.std(-loss_np) / np.sqrt(len(loss_np)) / beta / L / L
        print(f"{f} +- {df}")

wandb_run.finish()