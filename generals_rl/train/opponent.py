from __future__ import annotations
import random
import numpy as np
import torch
from typing import Optional
from ..env.generals_env import Owner
from ..env.generals_env_memory import GeneralsEnvWithMemory
from ..data.encoding import encode_obs_sequence
from ..data.history import ObsHistory

@torch.no_grad()
def choose_opponent_action(env: GeneralsEnvWithMemory, opp_policy, opp_hist: ObsHistory,
                          device, random_prob: float, T: int) -> int:
    mask_np = env.legal_action_mask(Owner.P1)
    legal = np.flatnonzero(mask_np)
    if len(legal) == 0:
        return 0
    if opp_policy is None or random.random() < float(random_prob):
        return int(np.random.choice(legal))
    seq = opp_hist.get_padded_seq()
    x_img_seq, x_meta_seq = encode_obs_sequence(seq, player_id=1)
    x_img = x_img_seq.unsqueeze(0).to(device)
    x_meta = x_meta_seq.unsqueeze(0).to(device)
    maskA = torch.from_numpy(mask_np).unsqueeze(0).to(device).to(torch.bool)
    a, _, _, _ = opp_policy.act(x_img, x_meta, maskA)
    return int(a.item())

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
    opp,                       # None or policy net (same type as your main policy)
    h_list: List[ObsHistory],  # histories for player 1
    *,
    device: torch.device,
    random_prob: float,
    T: int,
    seq_padding: bool,
) -> List[int]:
    """
    Return a list of opponent actions for all envs.
    - some envs act randomly w.p. random_prob
    - others use opp policy in ONE batched forward
    """
    E = len(envs)
    actions = [0] * E

    # decide random/policy per env
    use_random = [False] * E
    for i in range(E):
        if opp is None:
            use_random[i] = True
        else:
            use_random[i] = (random.random() < float(random_prob))

    # 1) fill random actions
    for i in range(E):
        if use_random[i]:
            mask_np = envs[i].legal_action_mask(Owner.P1)
            legal = np.nonzero(mask_np)[0]
            actions[i] = int(random.choice(legal))

    # 2) batched policy actions for the rest
    idxs = [i for i in range(E) if not use_random[i]]
    if len(idxs) == 0:
        return actions

    x_img_seq_list = []
    x_meta_seq_list = []
    mask_list = []
    lens = []

    for i in idxs:
        seq1 = h_list[i].get_padded_seq() if seq_padding else h_list[i].get_seq()
        x_img_seq, x_meta_seq = encode_obs_sequence(seq1, player_id=1)  # (t,20,H,W), (t,10)
        x_img_seq_list.append(x_img_seq)
        x_meta_seq_list.append(x_meta_seq)
        lens.append(int(x_img_seq.shape[0]))

        mask_np = envs[i].legal_action_mask(Owner.P1)
        mask_list.append(torch.from_numpy(mask_np).to(torch.bool))

    T_step = max(lens)
    x_img = torch.stack([_pad_first_time(x, T_step) for x in x_img_seq_list], dim=0).to(device)   # (B,T,C,H,W)
    x_meta = torch.stack([_pad_first_time(x, T_step) for x in x_meta_seq_list], dim=0).to(device) # (B,T,M)
    maskA = torch.stack(mask_list, dim=0).to(device).to(torch.bool)                                # (B,A)

    a, _, _, _ = opp.act(x_img, x_meta, maskA)  # (B,)

    for j, i in enumerate(idxs):
        actions[i] = int(a[j].item())

    return actions