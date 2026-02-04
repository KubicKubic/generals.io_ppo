from __future__ import annotations
import random
import numpy as np
import torch
from typing import Optional
from ..env.generals_env import Owner
from ..env.generals_env_memory import GeneralsEnvWithMemory
from ..data.encoding import encode_obs_sequence
from ..data.history import ObsHistory
from ..models.policy_rope_factorized import SeqPPOPolicyRoPEFactorized

@torch.no_grad()
def choose_opponent_action(env: GeneralsEnvWithMemory, opp_policy: Optional[SeqPPOPolicyRoPEFactorized], opp_hist: ObsHistory,
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
