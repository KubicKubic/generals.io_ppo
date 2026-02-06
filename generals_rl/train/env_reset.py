from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np

from ..env.generals_env import GeneralsEnv
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


def make_env(env_cfg: EnvConfig, seed: int) -> GeneralsEnv:
    """Create a fresh env instance from config."""
    kwargs: dict[str, Any] = {
        "H": int(env_cfg.H),
        "W": int(env_cfg.W),
        "seed": int(seed),
        "round_len": int(env_cfg.round_len),
        "city_initial_low": int(env_cfg.city_initial_low),
        "city_initial_high": int(env_cfg.city_initial_high),
        "general_initial": int(env_cfg.general_initial),
        "vision_radius": int(env_cfg.vision_radius),
        "round_includes_cities": bool(env_cfg.round_includes_cities),
        "enable_chase_priority": bool(env_cfg.enable_chase_priority),
        "min_real_size": int(env_cfg.min_real_size),
        "max_real_size": int(env_cfg.max_real_size),
        "min_general_dist": int(env_cfg.min_general_dist),
        "mountain_ratio_low": float(env_cfg.mountain_ratio_low),
        "mountain_ratio_high": float(env_cfg.mountain_ratio_high),
        "city_count_low": int(env_cfg.city_count_low),
        "city_count_high": int(env_cfg.city_count_high),
        "fog_of_war": bool(env_cfg.fog_of_war),
        "forbid_mode1": bool(env_cfg.forbid_mode1),
    }
    if env_cfg.max_halfturns is not None:
        kwargs["max_halfturns"] = int(env_cfg.max_halfturns)
    if env_cfg.obs_T is not None:
        kwargs["obs_T"] = int(env_cfg.obs_T)
    return GeneralsEnv(**kwargs)


def reset_episode(env: Optional[GeneralsEnv], env_cfg: EnvConfig, seeder: EpisodeSeeder) -> Tuple[GeneralsEnv, Any, Any]:
    seed = seeder.next_seed()
    if env is None:
        env = make_env(env_cfg, seed)
        (o0, o1), _ = env.reset()
        return env, o0, o1
    try:
        (o0, o1), _ = env.reset(seed=seed)  # type: ignore
        return env, o0, o1
    except TypeError:
        # older env.reset signature fallback
        env = make_env(env_cfg, seed)
        (o0, o1), _ = env.reset()
        return env, o0, o1
