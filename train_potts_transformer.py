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
vocab_size = 6   # Q
beta = np.log(1 + np.sqrt(vocab_size))

# --- model hyperparams ---
n_layer = 4 
n_head = 4
n_embd = 128
dropout = 0.0
bias = True
batch_size = 4096 
D4_symmetry = True

# --- training hyperparams ---
iterations = 40000
annealing_steps = 20000
lr = 0.0001
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
        name=f"Q={vocab_size}_reward-scaling",
        config=config_wb,
)

# --- log probability ---
def log_prob(model, sample):
    logits, _ = model(sample)

    log_probs = torch.log_softmax(logits, dim=-1)
    out = log_probs[:, :-1].gather(2, sample[:, 1:].unsqueeze(-1)).squeeze()
    return torch.sum(out, dim=-1) - np.log(config.vocab_size)

# --- Full Symmetry ---
def apply_symmetry(sample, L, Q):
    B = sample.shape[0]
    s = sample.view(B, L, L)
    
    # rotation
    s = torch.rot90(s, k=torch.randint(0, 4, (1,)).item(), dims=(1, 2))
    # symmetry
    if torch.rand(1) > 0.5:
        s = torch.flip(s, dims=[2])
    # translation
    shift_x, shift_y = torch.randint(0, L, (2,)).tolist()
    s = torch.roll(s, shifts=(shift_x, shift_y), dims=(1, 2))
    # state permutation
    perm = torch.randperm(Q, device=sample.device)
    s = perm[s]
    
    return s.reshape(B, -1)


#############################################
try:
    for i in tqdm(range(iterations)):
        # --- annealing ---
        if i < annealing_steps:
            beta_train = (i / annealing_steps) * beta
        else:
            beta_train = beta

        # --- sampling ---
        model.eval()
        with torch.no_grad():
            start = torch.randint(0, vocab_size, (batch_size, 1), device=dev)
            sample_raw = model._orig_mod.generate(start, L * L - 1)
            
            if D4_symmetry:
                sample_sym = apply_symmetry(sample_raw, L, vocab_size)
            else:
                sample_sym = sample_raw

        # --- training ---
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            log_p_raw = log_prob(model, sample_raw)
            log_p_sym = log_prob(model, sample_sym)
            
            log_p_avg = torch.logaddexp(log_p_raw, log_p_sym) - np.log(2.0)
            
            energy = energy_nD(sample_raw, L, L)
            
            loss_val = log_p_avg.float() + energy.float() * beta_train

        # --- reinforce update ---
        loss_val = log_p_avg.float() + energy.float() * beta_train
        reward = loss_val.detach() - loss_val.detach().mean()
        loss_reinforce = torch.mean(reward * log_p_avg.float())
        loss_reinforce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # --- log ----
        if (i + 1) % 10 == 0:
            # effective sample size
            log_sum_w = torch.logsumexp(-loss_val, 0)
            log_sum_w_sq = torch.logsumexp(-2 * loss_val, 0)
            ess = torch.exp(2 * log_sum_w - log_sum_w_sq).item() / batch_size

            f_val = loss_val.mean().item() / (beta_train * L * L) if beta_train > 0 else 0
            
            # magnetisation
            m_val = batched_bincount(sample_raw).amax(dim=1).float().mean().item() / (L * L)

            wandb_run.log(dict(
                ess=ess, 
                free_energy=f_val, 
                M=m_val, 
                beta=beta_train,
                loss_reinforce=loss_reinforce.item()
            ), i)

except KeyboardInterrupt:
    pass

finally:
    # --- validation ---
    model.eval()
    with torch.inference_mode():
        loss_arr = []
        for i in range(10):
            start = torch.randint(0, config.vocab_size, (batch_size, 1), device=dev)
            sample = model._orig_mod.generate(start, L * L - 1)

            log_p = log_prob(model, sample)
            energy = energy_nD(sample, L, L)
            loss = log_p + energy * beta
            loss_arr.append(loss.cpu())

        loss_np = np.array(loss_arr).flatten()
        f = (logsumexp(-loss_np) - np.log(len(loss_np))) / beta / L / L
        df = np.std(-loss_np) / np.sqrt(len(loss_np)) / beta / L / L
        print(f"Final Free Energy at Beta_c: {f} +- {df}")

wandb_run.finish()
