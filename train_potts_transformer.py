###################################################
# GPT training script for Potts Physics           #
#                                                 #
# Jan Kwinta                                      #
# April 2026                                      #
###################################################

import argparse
import math
from dataclasses import asdict
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from potts import beta_critical, energy2D, exact_free_energy
from transformer import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description="train a GPT sampler for the 2D Potts model")
    phys = p.add_argument_group("physics")
    phys.add_argument("--L", type=int, default=16, help="lattice size")
    phys.add_argument("--Q", type=int, default=6, help="number of Potts states")
    phys.add_argument("--beta", type=float, default=None, help="default: critical ln(1+sqrt(Q))")

    model = p.add_argument_group("model")
    model.add_argument("--n-layer", type=int, default=4)
    model.add_argument("--n-head", type=int, default=4)
    model.add_argument("--n-embd", type=int, default=128)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--no-bias", action="store_true")

    train = p.add_argument_group("training")
    train.add_argument("--iterations", type=int, default=40000)
    train.add_argument("--annealing-steps", type=int, default=20000,
                       help="linear ramp of beta from 0 over this many iterations")
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--beta1", type=float, default=0.9)
    train.add_argument("--beta2", type=float, default=0.95)
    train.add_argument("--clip-grad", type=float, default=1.0)
    train.add_argument("--no-symmetry", action="store_true",
                       help="disable averaging q over a random lattice/state symmetry")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--eval-batches", type=int, default=10)

    hw = p.add_argument_group("hardware")
    hw.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    hw.add_argument("--no-compile", action="store_true")

    log = p.add_argument_group("logging")
    log.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    log.add_argument("--wandb-entity", default="jankwinta-ju")
    log.add_argument("--wandb-project", default=None, help="default: potts-train-{L}")
    log.add_argument("--name", default=None, help="run name, default: Q={Q}")
    log.add_argument("--log-every", type=int, default=10)
    log.add_argument("--exact", action="store_true",
                     help="also log the transfer matrix free energy (small L only)")
    log.add_argument("--out", default=None, help="where to save the final checkpoint")
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_symmetry(sample, L, Q):
    """one random element of the lattice symmetry group times a random state permutation"""
    B = sample.shape[0]
    s = sample.view(B, L, L)

    s = torch.rot90(s, k=torch.randint(0, 4, (1,)).item(), dims=(1, 2))
    if torch.rand(1) > 0.5:
        s = torch.flip(s, dims=[2])
    shift_x, shift_y = torch.randint(0, L, (2,)).tolist()
    s = torch.roll(s, shifts=(shift_x, shift_y), dims=(1, 2))
    perm = torch.randperm(Q, device=sample.device)
    s = perm[s]

    return s.reshape(B, -1)


def effective_sample_size(loss):
    log_sum_w = torch.logsumexp(-loss, 0)
    log_sum_w_sq = torch.logsumexp(-2 * loss, 0)
    return torch.exp(2 * log_sum_w - log_sum_w_sq).item() / loss.numel()


def main():
    args = parse_args()
    L, Q = args.L, args.Q
    beta = args.beta if args.beta is not None else beta_critical(Q)
    dev = pick_device(args.device)
    use_autocast = dev == "cuda"
    use_compile = dev == "cuda" and not args.no_compile
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    config = GPTConfig(
        vocab_size=Q,
        block_size=L * L,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=not args.no_bias,
    )
    raw_model = GPT(config).to(dev)
    model = torch.compile(raw_model) if use_compile else raw_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    batched_bincount = torch.vmap(partial(torch.bincount, minlength=Q))
    print(f"device={dev} params={raw_model.get_num_params() / 1e6:.2f}M beta={beta:.5f}")

    f_exact = exact_free_energy(L, Q, beta) if args.exact else None
    if f_exact is not None:
        print(f"exact free energy per site: {f_exact:.6f}")

    import wandb

    run = wandb.init(
        mode=args.wandb,
        entity=args.wandb_entity,
        project=args.wandb_project or f"potts-train-{L}",
        name=args.name or f"Q={Q}",
        config=dict(vars(args), beta=beta, device=dev, **asdict(config)),
    )

    try:
        for i in tqdm(range(args.iterations)):
            beta_train = beta * min(1.0, i / args.annealing_steps) if args.annealing_steps else beta

            # --- sampling ---
            model.eval()
            with torch.no_grad():
                start = torch.randint(0, Q, (args.batch_size, 1), device=dev)
                sample_raw = raw_model.generate(start, L * L - 1)
                sample_sym = sample_raw if args.no_symmetry else apply_symmetry(sample_raw, L, Q)

            # --- training ---
            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                log_p_raw = model.log_prob(sample_raw)
                if args.no_symmetry:
                    log_p = log_p_raw
                else:
                    log_p_sym = model.log_prob(sample_sym)
                    log_p = torch.logaddexp(log_p_raw, log_p_sym) - math.log(2.0)

            energy = energy2D(sample_raw, L, L)
            loss_val = log_p.float() + energy.float() * beta_train

            # --- reinforce update ---
            reward = loss_val.detach() - loss_val.detach().mean()
            loss_reinforce = torch.mean(reward * log_p.float())
            loss_reinforce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            # --- log ---
            if (i + 1) % args.log_every == 0:
                f_val = loss_val.mean().item() / (beta_train * L * L) if beta_train > 0 else 0.0
                m_val = batched_bincount(sample_raw).amax(dim=1).float().mean().item() / (L * L)
                stats = dict(
                    ess=effective_sample_size(loss_val),
                    free_energy=f_val,
                    M=m_val,
                    beta=beta_train,
                    loss_reinforce=loss_reinforce.item(),
                )
                if f_exact is not None and beta_train == beta:
                    stats["free_energy_error"] = f_val - f_exact
                run.log(stats, i)

    except KeyboardInterrupt:
        pass

    finally:
        # --- validation ---
        model.eval()
        with torch.inference_mode():
            loss_arr = []
            for _ in range(args.eval_batches):
                start = torch.randint(0, Q, (args.batch_size, 1), device=dev)
                sample = raw_model.generate(start, L * L - 1)
                loss = raw_model.log_prob(sample) + energy2D(sample, L, L) * beta
                loss_arr.append(loss.cpu())

            loss_t = torch.cat(loss_arr).double()
            # mean of w = exp(-loss) estimates Z, hence F = -log(mean w) / beta
            f = -(torch.logsumexp(-loss_t, 0) - math.log(len(loss_t))).item() / (beta * L * L)
            df = loss_t.std().item() / math.sqrt(len(loss_t)) / (beta * L * L)
            f_var = loss_t.mean().item() / (beta * L * L)
            print(f"free energy (importance sampled): {f:.6f} +- {df:.6f}")
            print(f"free energy (variational bound):  {f_var:.6f}")
            if f_exact is not None:
                print(f"free energy (exact):              {f_exact:.6f}")
            run.summary.update(dict(final_free_energy=f, final_free_energy_err=df,
                                    final_free_energy_variational=f_var,
                                    final_ess=effective_sample_size(loss_t)))

        if args.out:
            torch.save(dict(config=asdict(config), state_dict=raw_model.state_dict(),
                            args=vars(args), beta=beta), args.out)

        run.finish()


if __name__ == "__main__":
    main()
