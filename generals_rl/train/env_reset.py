from __future__ import annotations
from typing import Any, Optional, Tuple
import numpy as np
from ..env.generals_env_memory import GeneralsEnvWithMemory
from .config import EnvConfig

class EpisodeSeeder:
    def __init__(self, env_cfg: EnvConfig):
        self.cfg = env_cfg
        self.episode_idx = 0

    def next_seed(self) -> int:
        mode = str(self.cfg.reset_seed_mode).lower()
        if mode == "fixed":
            s = int(self.cfg.base_seed)
        elif mode == "random":
            s = int(np.random.randint(0, 2**31 - 1))
        else:
            s = int(self.cfg.base_seed) + self.episode_idx * int(self.cfg.seed_increment)
        self.episode_idx += 1
        return s

def make_env(env_cfg: EnvConfig, seed: int) -> GeneralsEnvWithMemory:
    kwargs = {"seed": int(seed)}
    if env_cfg.max_halfturns is not None:
        kwargs["max_halfturns"] = int(env_cfg.max_halfturns)
    return GeneralsEnvWithMemory(**kwargs)

def reset_episode(env: Optional[GeneralsEnvWithMemory], env_cfg: EnvConfig, seeder: EpisodeSeeder) -> Tuple[GeneralsEnvWithMemory, Any, Any]:
    seed = seeder.next_seed()
    if env is None:
        env = make_env(env_cfg, seed)
        (o0, o1), _ = env.reset()
        return env, o0, o1
    try:
        (o0, o1), _ = env.reset(seed=seed)  # type: ignore
        return env, o0, o1
    except TypeError:
        env = make_env(env_cfg, seed)
        (o0, o1), _ = env.reset()
        return env, o0, o1
