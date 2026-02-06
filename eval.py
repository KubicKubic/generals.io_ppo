
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from generals_rl.env.generals_env import GeneralsEnv, Owner
from generals_rl.models.registry import make_policy


# -----------------------------
# Config
# -----------------------------
@dataclass
class EvalEnvConfig:
    # map / rules (must match checkpoints' expected H/W and action-space)
    H: int = 25
    W: int = 25
    min_real_size: int = 5
    max_real_size: int = 6
    min_general_dist: int = 5

    round_len: int = 25
    vision_radius: int = 1
    city_initial_low: int = 40
    city_initial_high: int = 50
    general_initial: int = 1
    round_includes_cities: bool = True
    enable_chase_priority: bool = True

    mountain_ratio_low: float = 1.0 / 8.0
    mountain_ratio_high: float = 1.0 / 4.0
    city_count_low: int = 0
    city_count_high: int = 1

    fog_of_war: bool = True

    # evaluation step limit (in half-turns)
    max_halfturns: int = 128

    # sequence cache length used by the model (T)
    obs_T: int = 1

    # action-space constraint (should match training)
    forbid_mode1: bool = True


@dataclass
class EvalConfig:
    # checkpoints
    ckpt_a: str = ""
    ckpt_b: str = ""

    # parallelism
    num_envs: int = 32

    # evaluation size
    num_maps: int = 200
    base_seed: int = 0
    seed_increment: int = 1

    # policy inference
    device: str = "cuda"
    predict_mode: str = "sample"  # "sample" | "argmax"
    seq_padding: bool = False

    # io
    out_dir: str = "eval_runs"
    run_name: str = ""  # if empty, auto timestamp

    # env + model overrides (usually inferred from ckpt["config"] if present)
    env: EvalEnvConfig = EvalEnvConfig()
    model: Dict[str, Any] = None  # {name:..., st_axial2d:{...}} etc


def _try_load_yaml(path: str) -> Dict[str, Any]:
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML config requested but PyYAML is not available. "
                "Either install pyyaml or provide a .json config."
            ) from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _deep_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_from_ckpt_config(cfg: EvalConfig, ckpt_cfg: Dict[str, Any]) -> None:
    """
    If checkpoint saved a resolved training config, use it as defaults for model/env
    (but keep eval-only options from cfg).
    """
    if not isinstance(ckpt_cfg, dict):
        return

    # env
    env_d = ckpt_cfg.get("env", None)
    if isinstance(env_d, dict):
        for k in asdict(cfg.env).keys():
            if k in env_d:
                setattr(cfg.env, k, env_d[k])

    # model
    m = ckpt_cfg.get("model", None)
    if isinstance(m, dict):
        cfg.model = m


# -----------------------------
# Helpers
# -----------------------------
def _pad_first_time(x: torch.Tensor, T: int) -> torch.Tensor:
    t = int(x.shape[0])
    if t == T:
        return x
    if t > T:
        return x[-T:]
    pad_len = T - t
    pad = x[:1].expand(pad_len, *x.shape[1:]).contiguous()
    return torch.cat([pad, x], dim=0)


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -1e9)
    return masked.argmax(dim=-1)


def _masked_sample(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -1e9)
    dist = torch.distributions.Categorical(logits=masked)
    return dist.sample()


def _swap_generals_inplace(env: GeneralsEnv) -> None:
    """
    Swap the two generals' positions while keeping the rest of the map identical.

    We do this by swapping the owner+army on the two GENERAL tiles,
    and swapping env.general_pos entries.

    NOTE: After swapping, we rebuild the env's internal model-obs cache so that the
    initial observation reflects the swap.
    """
    p0 = env.general_pos.get(Owner.P0, None)
    p1 = env.general_pos.get(Owner.P1, None)
    if p0 is None or p1 is None:
        return

    r0, c0 = p0
    r1, c1 = p1

    # swap owner
    o0 = int(env.owner[r0, c0])
    o1 = int(env.owner[r1, c1])
    env.owner[r0, c0] = o1
    env.owner[r1, c1] = o0

    # swap army
    a0 = int(env.army[r0, c0])
    a1 = int(env.army[r1, c1])
    env.army[r0, c0] = a1
    env.army[r1, c1] = a0

    # swap general_pos bookkeeping
    env.general_pos[Owner.P0], env.general_pos[Owner.P1] = env.general_pos[Owner.P1], env.general_pos[Owner.P0]

    # rebuild cache + push initial obs
    env._cache_reset()
    obs0 = env._make_obs(Owner.P0)
    obs1 = env._make_obs(Owner.P1)
    env._cache_push(obs0, obs1)


def _total_army(env: GeneralsEnv) -> Tuple[int, int]:
    # army on all tiles owned by each player
    o = env.owner
    a = env.army
    s0 = int(a[o == int(Owner.P0)].sum())
    s1 = int(a[o == int(Owner.P1)].sum())
    return s0, s1


def _winner_by_army(env: GeneralsEnv) -> Optional[int]:
    s0, s1 = _total_army(env)
    if s0 > s1:
        return int(Owner.P0)
    if s1 > s0:
        return int(Owner.P1)
    return None  # draw


@torch.no_grad()
def _batched_actions_for_player(
    envs: List[GeneralsEnv],
    idxs: List[int],
    player: Owner,
    policy,
    device: torch.device,
    predict_mode: str,
    seq_padding: bool,
) -> torch.Tensor:
    """
    Returns actions for envs at indices `idxs` for `player`, batched.
    """
    if len(idxs) == 0:
        return torch.empty((0,), dtype=torch.long, device="cpu")

    x_img_list = []
    x_meta_list = []
    mask_list = []
    lens = []

    for i in idxs:
        img_np, meta_np = envs[i].get_model_obs_seq(player, padded=bool(seq_padding))
        x_img = torch.from_numpy(img_np)   # (t,C,H,W)
        x_meta = torch.from_numpy(meta_np) # (t,M)
        x_img_list.append(x_img)
        x_meta_list.append(x_meta)
        lens.append(int(x_img.shape[0]))

        mask_np = envs[i].legal_action_mask(player)
        mask_list.append(torch.from_numpy(mask_np).to(torch.bool))

    T_step = max(lens)
    x_img_b = torch.stack([_pad_first_time(x, T_step) for x in x_img_list], dim=0).to(device)
    x_meta_b = torch.stack([_pad_first_time(x, T_step) for x in x_meta_list], dim=0).to(device)
    mask_b = torch.stack(mask_list, dim=0).to(device).to(torch.bool)

    logits, _v = policy.logits_and_value(x_img_b, x_meta_b)

    if predict_mode == "argmax":
        a = _masked_argmax(logits, mask_b)
    else:
        a = _masked_sample(logits, mask_b)

    return a.long().to("cpu")


def _make_env(cfg: EvalConfig) -> GeneralsEnv:
    e = cfg.env
    return GeneralsEnv(
        H=int(e.H),
        W=int(e.W),
        min_real_size=int(e.min_real_size),
        max_real_size=int(e.max_real_size),
        min_general_dist=int(e.min_general_dist),
        round_len=int(e.round_len),
        vision_radius=int(e.vision_radius),
        city_initial_low=int(e.city_initial_low),
        city_initial_high=int(e.city_initial_high),
        general_initial=int(e.general_initial),
        round_includes_cities=bool(e.round_includes_cities),
        enable_chase_priority=bool(e.enable_chase_priority),
        mountain_ratio_low=float(e.mountain_ratio_low),
        mountain_ratio_high=float(e.mountain_ratio_high),
        city_count_low=int(e.city_count_low),
        city_count_high=int(e.city_count_high),
        fog_of_war=bool(e.fog_of_war),
        max_halfturns=int(e.max_halfturns),
        forbid_mode1=bool(e.forbid_mode1),
        obs_T=int(e.obs_T),
    )


def _infer_policy_from_ckpt(cfg: EvalConfig, ckpt_path: str) -> Tuple[Any, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    return ckpt, ckpt_cfg


def _build_policy(cfg: EvalConfig, ckpt_state: Dict[str, Any], device: torch.device):
    assert cfg.model is not None and isinstance(cfg.model, dict), \
        "Model config not found. Provide it in eval config, or use checkpoints saved with config['model']."

    model_name = str(cfg.model.get("name", "st_axial2d"))
    kwargs = dict(cfg.model)
    kwargs.pop("name", None)

    action_size = int(cfg.env.H) * int(cfg.env.W) * 4 * 2

    policy = make_policy(
        model_name,
        action_size=action_size,
        H=int(cfg.env.H),
        W=int(cfg.env.W),
        T=int(cfg.env.obs_T),
        img_channels=20,
        meta_dim=10,
        st_rope2d=kwargs.get("st_rope2d", None),
        st_axial2d=kwargs.get("st_axial2d", None),
    ).to(device)

    if "policy" in ckpt_state:
        policy.load_state_dict(ckpt_state["policy"], strict=True)
    else:
        # allow raw state_dict
        policy.load_state_dict(ckpt_state, strict=True)

    policy.eval()
    return policy


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# -----------------------------
# Main eval
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--ckpt_a", type=str, default="")
    ap.add_argument("--ckpt_b", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    cfg = EvalConfig()

    if args.config:
        d = _try_load_yaml(args.config)
        # shallow merge
        for k, v in d.items():
            if k == "env" and isinstance(v, dict):
                for ek, ev in v.items():
                    if hasattr(cfg.env, ek):
                        setattr(cfg.env, ek, ev)
            elif k == "model":
                cfg.model = v
            elif hasattr(cfg, k):
                setattr(cfg, k, v)

    if args.ckpt_a:
        cfg.ckpt_a = args.ckpt_a
    if args.ckpt_b:
        cfg.ckpt_b = args.ckpt_b
    if args.out_dir:
        cfg.out_dir = args.out_dir

    assert cfg.ckpt_a and cfg.ckpt_b, "Need --ckpt_a and --ckpt_b (or specify in config)."

    # load checkpoint configs (if present) and use them as defaults
    ckpt_a, ckpt_a_cfg = _infer_policy_from_ckpt(cfg, cfg.ckpt_a)
    ckpt_b, ckpt_b_cfg = _infer_policy_from_ckpt(cfg, cfg.ckpt_b)

    # Prefer ckpt_a config as base; then ckpt_b (should match)
    _resolve_from_ckpt_config(cfg, ckpt_a_cfg)
    _resolve_from_ckpt_config(cfg, ckpt_b_cfg)

    if cfg.model is None:
        # try again after resolve
        cfg.model = _deep_get(ckpt_a_cfg, ["model"], None) or _deep_get(ckpt_b_cfg, ["model"], None)

    # finalize run directory
    run_name = cfg.run_name.strip() or _timestamp_name()
    out_dir = os.path.join(cfg.out_dir, run_name)
    _ensure_dir(out_dir)

    # device
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    # build policies
    policy_a = _build_policy(cfg, ckpt_a, device)
    policy_b = _build_policy(cfg, ckpt_b, device)

    # seeds: each map => two games (normal + swapped generals)
    seeds: List[Tuple[int, bool]] = []
    for m in range(int(cfg.num_maps)):
        seed = int(cfg.base_seed) + m * int(cfg.seed_increment)
        seeds.append((seed, False))
        seeds.append((seed, True))  # swap generals

    total_games = len(seeds)

    # stats
    wins_a_final = 0.0
    wins_b_final = 0.0
    draws_final = 0

    wins_a_nolimit = 0.0
    wins_b_nolimit = 0.0
    draws_nolimit = 0
    denom_nolimit = 0

    # logging
    games_log_path = os.path.join(out_dir, "games.jsonl")
    cfg_path = os.path.join(out_dir, "resolved_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "eval": asdict(cfg),
            "note": "policy A controls Owner.P0; policy B controls Owner.P1; each seed is played twice, swapping generals' starting positions.",
        }, f, indent=2)

    with open(games_log_path, "w", encoding="utf-8") as fout:
        # process in chunks for parallelism
        cursor = 0
        while cursor < total_games:
            chunk = seeds[cursor: cursor + int(cfg.num_envs)]
            B = len(chunk)
            cursor += B

            envs = [_make_env(cfg) for _ in range(B)]

            # reset
            for i, (seed, do_swap) in enumerate(chunk):
                envs[i].reset(seed=seed)
                if do_swap:
                    _swap_generals_inplace(envs[i])

            active = [True] * B
            terminated_before_limit = [False] * B
            winner_final: List[Optional[int]] = [None] * B
            steps = [0] * B

            # rollout
            while any(active):
                idxs = [i for i, a in enumerate(active) if a]
                if len(idxs) == 0:
                    break

                # batched actions for both sides
                a0 = _batched_actions_for_player(
                    envs, idxs, Owner.P0, policy_a, device,
                    predict_mode=str(cfg.predict_mode).lower(),
                    seq_padding=bool(cfg.seq_padding),
                )
                a1 = _batched_actions_for_player(
                    envs, idxs, Owner.P1, policy_b, device,
                    predict_mode=str(cfg.predict_mode).lower(),
                    seq_padding=bool(cfg.seq_padding),
                )

                # step each active env (env itself isn't vectorized)
                for j, i in enumerate(idxs):
                    sr = envs[i].step(int(a0[j].item()), int(a1[j].item()))
                    steps[i] += 1

                    if sr.terminated:
                        terminated_before_limit[i] = True
                        w = sr.info.get("winner", None)
                        winner_final[i] = w
                        active[i] = False
                    elif sr.truncated:
                        # step limit reached: decide by total army
                        w = _winner_by_army(envs[i])
                        winner_final[i] = w
                        active[i] = False

            # accumulate stats + write logs
            for i, (seed, do_swap) in enumerate(chunk):
                w = winner_final[i]
                is_draw = (w is None)
                if is_draw:
                    draws_final += 1
                    wins_a_final += 0.5
                    wins_b_final += 0.5
                else:
                    if int(w) == int(Owner.P0):
                        wins_a_final += 1.0
                    elif int(w) == int(Owner.P1):
                        wins_b_final += 1.0

                if terminated_before_limit[i]:
                    denom_nolimit += 1
                    if is_draw:
                        draws_nolimit += 1
                        wins_a_nolimit += 0.5
                        wins_b_nolimit += 0.5
                    else:
                        if int(w) == int(Owner.P0):
                            wins_a_nolimit += 1.0
                        elif int(w) == int(Owner.P1):
                            wins_b_nolimit += 1.0

                rec = {
                    "seed": int(seed),
                    "swap_generals": bool(do_swap),
                    "steps": int(steps[i]),
                    "terminated_before_limit": bool(terminated_before_limit[i]),
                    "winner": None if w is None else int(w),
                    "army_final": list(_total_army(envs[i])),
                }
                fout.write(json.dumps(rec) + "\n")

    # summarize
    final_wr_a = wins_a_final / float(total_games) if total_games > 0 else 0.0
    final_wr_b = wins_b_final / float(total_games) if total_games > 0 else 0.0

    if denom_nolimit > 0:
        nolimit_wr_a = wins_a_nolimit / float(denom_nolimit)
        nolimit_wr_b = wins_b_nolimit / float(denom_nolimit)
    else:
        nolimit_wr_a = 0.0
        nolimit_wr_b = 0.0

    summary = {
        "total_games": int(total_games),
        "total_maps": int(cfg.num_maps),
        "each_map_played_twice_swap_generals": True,
        "predict_mode": str(cfg.predict_mode).lower(),
        "step_limit_halfturns": int(cfg.env.max_halfturns),
        "final": {
            "wins_a": float(wins_a_final),
            "wins_b": float(wins_b_final),
            "draws": int(draws_final),
            "winrate_a": float(final_wr_a),
            "winrate_b": float(final_wr_b),
        },
        "no_step_limit_reached_only": {
            "games": int(denom_nolimit),
            "wins_a": float(wins_a_nolimit),
            "wins_b": float(wins_b_nolimit),
            "draws": int(draws_nolimit),
            "winrate_a": float(nolimit_wr_a),
            "winrate_b": float(nolimit_wr_b),
        },
        "artifacts": {
            "out_dir": out_dir,
            "resolved_config": os.path.join(out_dir, "resolved_config.json"),
            "games_jsonl": os.path.join(out_dir, "games.jsonl"),
        }
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
