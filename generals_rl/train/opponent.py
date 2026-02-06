from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch

from ..env.generals_env import Owner


def _pad_first_time(x: torch.Tensor, T: int) -> torch.Tensor:
    """x: (t, ...) -> (T, ...) by left-padding with first frame"""
    t = int(x.shape[0])
    if t == T:
        return x
    if t > T:
        return x[-T:]
    pad_len = T - t
    pad = x[:1].expand(pad_len, *x.shape[1:]).contiguous()
    return torch.cat([pad, x], dim=0)


@torch.no_grad()
def choose_opponent_action_batched(
    envs,
    opp,  # None or policy net
    *,
    device: torch.device,
    random_prob: float,
    seq_padding: bool,
) -> List[int]:
    """Return opponent actions (P1) for all envs.

    Uses the env's internal model-obs cache; does NOT call encode_obs_sequence.
    """
    E = len(envs)
    actions = [0] * E

    use_random = [False] * E
    for i in range(E):
        if opp is None:
            use_random[i] = True
        else:
            use_random[i] = (random.random() < float(random_prob))

    # 1) random
    for i in range(E):
        if use_random[i]:
            mask_np = envs[i].legal_action_mask(Owner.P1)
            legal = np.flatnonzero(mask_np)
            actions[i] = int(random.choice(list(legal))) if len(legal) > 0 else 0

    # 2) policy batch for remaining
    idxs = [i for i in range(E) if not use_random[i]]
    if len(idxs) == 0:
        return actions

    x_img_seq_list = []
    x_meta_seq_list = []
    mask_list = []
    lens = []

    for i in idxs:
        img_np, meta_np = envs[i].get_model_obs_seq(Owner.P1, padded=bool(seq_padding))
        x_img_seq = torch.from_numpy(img_np)   # (t,C,H,W)
        x_meta_seq = torch.from_numpy(meta_np) # (t,M)
        x_img_seq_list.append(x_img_seq)
        x_meta_seq_list.append(x_meta_seq)
        lens.append(int(x_img_seq.shape[0]))

        mask_np = envs[i].legal_action_mask(Owner.P1)
        mask_list.append(torch.from_numpy(mask_np).to(torch.bool))

    T_step = max(lens)
    x_img = torch.stack([_pad_first_time(x, T_step) for x in x_img_seq_list], dim=0).to(device)
    x_meta = torch.stack([_pad_first_time(x, T_step) for x in x_meta_seq_list], dim=0).to(device)
    maskA = torch.stack(mask_list, dim=0).to(device).to(torch.bool)

    a, _, _, _ = opp.act(x_img, x_meta, maskA)

    for j, i in enumerate(idxs):
        actions[i] = int(a[j].item())

    return actions
