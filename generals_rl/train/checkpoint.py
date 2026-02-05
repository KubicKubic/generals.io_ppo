from __future__ import annotations
import os, copy, torch
from typing import Dict, List

def save_checkpoint(
    path: str,
    policy,
    optimizer,
    update: int,
    ep_count: int,
    win_count: int,
    lose_count: int,
    opponent_pool: List,
    rng_state: Dict,
    config: Dict
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pool_state = [p.state_dict() for p in opponent_pool]
    torch.save({
        "update": update,
        "ep_count": ep_count,
        "win_count": win_count,
        "lose_count": lose_count,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "opponent_pool": pool_state,
        "rng_state": rng_state,
        "config": config,
    }, path)

def load_checkpoint(path: str, policy, optimizer, device, opponent_pool_max: int) -> Dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    optimizer.load_state_dict(ckpt["optimizer"])
    pool_state = ckpt.get("opponent_pool", [])
    opponent_pool = []
    for sd in pool_state[-opponent_pool_max:]:
        p = copy.deepcopy(policy).to(device)
        p.load_state_dict(sd)
        p.eval()
        opponent_pool.append(p)
    return {
        "update": int(ckpt.get("update", 0)),
        "ep_count": int(ckpt.get("ep_count", 0)),
        "win_count": int(ckpt.get("win_count", 0)),
        "lose_count": int(ckpt.get("lose_count", 0)),
        "opponent_pool": opponent_pool,
        "rng_state": ckpt.get("rng_state", None),
        "config": ckpt.get("config", {}),
    }
