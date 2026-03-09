import torch

def energy2D(S, r, c):
    state = S.reshape(-1, r, c)
    return -torch.sum(torch.roll(state, 1, 1) == state, dim=(1, 2)) - torch.sum(
        torch.roll(state, 1, 2) == state, dim=(1, 2)
    )
