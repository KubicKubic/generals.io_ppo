from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


# -----------------------------
# Leaf configs
# -----------------------------
@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.1
    epochs: int = 1
    minibatch: int = 256
    max_grad_norm: float = 1.0


@dataclass
class RewardShapingConfig:
    w_army: float = 0.1
    w_land: float = 0.1


@dataclass
class OpponentConfig:
    random_opp_prob: float = 0.25
    pool_pick_prob: float = 0.8
    pool_empty_mode: str = "random"   # "random" | "self"
    fallback_mode: str = "random"     # "random" | "self"
    snapshot_every: int = 25
    opponent_pool_max: int = 0


@dataclass
class VideoConfig:
    make_video: bool = False


@dataclass
class VizConfig:
    enable: bool = True
    out_dir: str = "samples_viz"
    every_updates: int = 8
    frames_per_update: int = 512
    save_mp4: bool = True
    mp4_fps: int = 2
    save_trace_jsonl: bool = True
    cell: int = 20
    draw_text: bool = True
    pov_player: int = 0
    reset_episode_before_viz: bool = False
    topk_actions: int = 3



@dataclass
class MCTSConfig:
    enabled: bool = False
    actor_mode: str = "ppo"            # "ppo" | "mcts"
    num_simulations: int = 100
    max_depth: int = 40
    c_puct: float = 1.5
    tau: float = 1.0
    deterministic: bool = False
    dirichlet_alpha: float = 0.0
    dirichlet_eps: float = 0.0
    topk_actions: int = 0
    opponent_model: str = "policy"     # "policy" | "random"
    opponent_sample: str = "sample"    # "sample" | "argmax"


@dataclass
class EnvConfig:
    # vectorization / seeding
    num_envs: int = 1
    base_seed: int = 0
    max_halfturns: Optional[int] = 50
    reset_seed_mode: str = "increment"   # "fixed" | "increment" | "random"
    seed_increment: int = 1

    # action-space constraint
    forbid_mode1: bool = True

    # ---- map / rules (these affect model I/O if H/W change) ----
    H: int = 25
    W: int = 25

    # sampled real window (embedded in fixed HxW)
    min_real_size: int = 5
    max_real_size: int = 6
    min_general_dist: int = 5

    # game knobs
    round_len: int = 25          # in turns
    vision_radius: int = 1
    city_initial_low: int = 40
    city_initial_high: int = 50
    general_initial: int = 1
    round_includes_cities: bool = True
    enable_chase_priority: bool = True

    # map distribution knobs
    mountain_ratio_low: float = 1.0 / 8.0
    mountain_ratio_high: float = 1.0 / 4.0
    city_count_low: int = 0
    city_count_high: int = 1

    # observation knobs
    fog_of_war: bool = True

    # model input cache length (will typically be set to TrainConfig.T by train())
    obs_T: Optional[int] = None


@dataclass
class ModelConfig:
    """
    Model selection + kwargs forwarded into generals_rl.models.registry.make_policy.

    This version is aligned to your YAML:
      model:
        name: st_axial2d
        st_rope2d: {...}
        st_axial2d: {...}

    So we only keep: name, st_rope2d, st_axial2d.
    """
    name: str = "st_axial2d"

    st_rope2d: Dict[str, Any] = field(default_factory=lambda: {
        "meta_proj": 16,
        "d_model": 64,
        "nhead": 4,
        "nlayers": 2,
        "dropout": 0.0,
        "enc_hidden": 128,
        "enc_depth": 3,
        "head_hidden": 128,
        "head_depth": 3,
    })

    st_axial2d: Dict[str, Any] = field(default_factory=lambda: {
        "meta_proj": 16,
        "d_model": 64,
        "nhead_time": 4,
        "nlayers_time": 2,
        "dropout": 0.0,
        "nhead_axial": 8,
        "nlayers_axial_enc": 2,
        "nlayers_axial_head": 2,
    })


# -----------------------------
# Top-level config
# -----------------------------
@dataclass
class TrainConfig:
    # main training
    total_updates: int = 1200
    rollout_len: int = 2048
    T: int = 100
    lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95

    # io/logging
    ckpt_dir: str = "checkpoints"
    save_every: int = 8
    log_every: int = 1
    resume_path: Optional[str] = None
    save_policy_path: str = "generals_seq_policy.pt"
    save_resolved_config: bool = True
    seq_padding: bool = False

    # nested
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    opponent: OpponentConfig = field(default_factory=OpponentConfig)
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# Helpers
# -----------------------------
def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dst with src (dict->dict only)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_config(path: str) -> TrainConfig:
    """
    Load TrainConfig from JSON or YAML.

    - .json => json
    - .yaml/.yml => yaml (requires pyyaml)
    - otherwise: try json then yaml
    """
    text = _load_text(path)
    suffix = os.path.splitext(path)[1].lower()

    data: Dict[str, Any]
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML. Please: pip install pyyaml") from e
        data = yaml.safe_load(text) or {}
    else:
        try:
            data = json.loads(text)
        except Exception:
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("Unknown config suffix. Use .json or install PyYAML for .yaml/.yml.") from e
            data = yaml.safe_load(text) or {}

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict.")

    # ----------------------------
    # merge onto defaults (DEEP)
    # ----------------------------
    cfg0 = TrainConfig()
    merged = cfg0.to_dict()
    _deep_update(merged, data)

    env_d = merged.get("env", {}) or {}
    model_d = merged.get("model", {}) or {}

    # ----------------------------
    # build EnvConfig (ALL FIELDS)
    # ----------------------------
    env_cfg = EnvConfig(
        # vectorization / seeding
        num_envs=int(env_d.get("num_envs", cfg0.env.num_envs)),
        base_seed=int(env_d.get("base_seed", cfg0.env.base_seed)),
        max_halfturns=env_d.get("max_halfturns", cfg0.env.max_halfturns),
        reset_seed_mode=str(env_d.get("reset_seed_mode", cfg0.env.reset_seed_mode)),
        seed_increment=int(env_d.get("seed_increment", cfg0.env.seed_increment)),

        # action-space constraint
        forbid_mode1=bool(env_d.get("forbid_mode1", cfg0.env.forbid_mode1)),

        # ---- map / rules ----
        H=int(env_d.get("H", cfg0.env.H)),
        W=int(env_d.get("W", cfg0.env.W)),

        # sampled real window
        min_real_size=int(env_d.get("min_real_size", cfg0.env.min_real_size)),
        max_real_size=int(env_d.get("max_real_size", cfg0.env.max_real_size)),
        min_general_dist=int(env_d.get("min_general_dist", cfg0.env.min_general_dist)),

        # game knobs
        round_len=int(env_d.get("round_len", cfg0.env.round_len)),
        vision_radius=int(env_d.get("vision_radius", cfg0.env.vision_radius)),
        city_initial_low=int(env_d.get("city_initial_low", cfg0.env.city_initial_low)),
        city_initial_high=int(env_d.get("city_initial_high", cfg0.env.city_initial_high)),
        general_initial=int(env_d.get("general_initial", cfg0.env.general_initial)),
        round_includes_cities=bool(env_d.get("round_includes_cities", cfg0.env.round_includes_cities)),
        enable_chase_priority=bool(env_d.get("enable_chase_priority", cfg0.env.enable_chase_priority)),

        # map distribution knobs
        mountain_ratio_low=float(env_d.get("mountain_ratio_low", cfg0.env.mountain_ratio_low)),
        mountain_ratio_high=float(env_d.get("mountain_ratio_high", cfg0.env.mountain_ratio_high)),
        city_count_low=int(env_d.get("city_count_low", cfg0.env.city_count_low)),
        city_count_high=int(env_d.get("city_count_high", cfg0.env.city_count_high)),

        # observation knobs
        fog_of_war=bool(env_d.get("fog_of_war", cfg0.env.fog_of_war)),

        # model input cache length
        obs_T=env_d.get("obs_T", cfg0.env.obs_T),
    )

    # ----------------------------
    # build ModelConfig (DEEP MERGE defaults)
    # ----------------------------
    # 关键：不要让用户只写了几个key就把默认字典清空
    m0 = ModelConfig()
    name = str(model_d.get("name", m0.name))

    st_rope2d = dict(m0.st_rope2d)
    _deep_update(st_rope2d, dict(model_d.get("st_rope2d", {}) or {}))

    st_axial2d = dict(m0.st_axial2d)
    _deep_update(st_axial2d, dict(model_d.get("st_axial2d", {}) or {}))

    model_cfg = ModelConfig(
        name=name,
        st_rope2d=st_rope2d,
        st_axial2d=st_axial2d,
    )

    # ----------------------------
    # build top-level TrainConfig
    # ----------------------------
    viz_d = merged.get("viz", {}) or {}

    cfg = TrainConfig(
        # main training
        total_updates=int(merged.get("total_updates", cfg0.total_updates)),
        rollout_len=int(merged.get("rollout_len", cfg0.rollout_len)),
        T=int(merged.get("T", cfg0.T)),
        lr=float(merged.get("lr", cfg0.lr)),
        gamma=float(merged.get("gamma", cfg0.gamma)),
        lam=float(merged.get("lam", cfg0.lam)),

        # io/logging
        ckpt_dir=str(merged.get("ckpt_dir", cfg0.ckpt_dir)),
        save_every=int(merged.get("save_every", cfg0.save_every)),
        log_every=int(merged.get("log_every", cfg0.log_every)),
        resume_path=merged.get("resume_path", cfg0.resume_path),
        save_policy_path=str(merged.get("save_policy_path", cfg0.save_policy_path)),
        save_resolved_config=bool(merged.get("save_resolved_config", cfg0.save_resolved_config)),
        seq_padding=bool(merged.get("seq_padding", cfg0.seq_padding)),

        # nested
        env=env_cfg,
        model=model_cfg,
        ppo=PPOConfig(**(merged.get("ppo", {}) or {})),
        opponent=OpponentConfig(**(merged.get("opponent", {}) or {})),
        reward_shaping=RewardShapingConfig(**(merged.get("reward_shaping", {}) or {})),
        video=VideoConfig(**(merged.get("video", {}) or {})),
        viz=VizConfig(**(viz_d or {})),
        mcts=MCTSConfig(**(merged.get("mcts", {}) or {})),
    )
    return cfg

def save_resolved_config(cfg: TrainConfig, ckpt_dir: str, filename: str = "config_resolved.json"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[config] wrote {path}")
