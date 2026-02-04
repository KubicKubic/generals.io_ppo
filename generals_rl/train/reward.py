from __future__ import annotations
from typing import Dict
import numpy as np

def phi_from_scores(obs0: Dict[str, np.ndarray], w_army: float, w_land: float) -> float:
    total_army = obs0["total_army"].astype(np.float32)
    total_land = obs0["total_land"].astype(np.float32)
    return float(w_army) * float(total_army[0] - total_army[1]) + float(w_land) * float(total_land[0] - total_land[1])

def potential_shaped_reward(env_r0: float, phi_s: float, phi_sp: float, gamma: float, done: bool) -> float:
    return float(env_r0) + gamma * float(phi_sp) * (0.0 if done else 1.0) - float(phi_s)
