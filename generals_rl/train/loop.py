from __future__ import annotations

from typing import List

import os
import copy
import random

import torch
import torch.optim as optim

from ..env.generals_env import Owner, TileType
from ..models.registry import make_policy
from ..mcts.alphazero_mcts import AlphaZeroMCTSPolicy, MCTSConfig as _MCTSConfig

from .config import TrainConfig
from .rng import set_seed, get_rng_state, set_rng_state
from .buffer import RolloutBuffer
from .reward import phi_from_scores, potential_shaped_reward
from .ppo import ppo_update
from .opponent import choose_opponent_action_batched
from .checkpoint import save_checkpoint, load_checkpoint
from .env_reset import EpisodeSeeder, reset_episode

from ..viz.rollout_viz import (
    viz_begin_update,
    viz_end_update,
    maybe_visualize_rollout_step_concat,
)
from ..video.checkpoint_video import make_video_for_checkpoint

from tqdm import tqdm


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


def train(cfg: TrainConfig, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.env.base_seed))

    # ---- vectorized envs ----
    num_envs = int(getattr(cfg.env, "num_envs", 1))
    if num_envs < 1:
        raise ValueError(f"env.num_envs must be >= 1, got {num_envs}")

    seeder = EpisodeSeeder(cfg.env)

    envs: List = []
    obs0_list = []
    obs1_list = []

    # Make env obs cache length follow TrainConfig.T unless user overrides env.obs_T explicitly
    if getattr(cfg.env, "obs_T", None) is None:
        cfg.env.obs_T = int(cfg.T)

    for _ in range(num_envs):
        env_i, o0, o1 = reset_episode(None, cfg.env, seeder)
        envs.append(env_i)
        obs0_list.append(o0)
        obs1_list.append(o1)

    # ---- model selection via config ----
    env0 = envs[0]
    policy = make_policy(
        cfg.model.name,
        action_size=env0.action_size,
        H=env0.H,
        W=env0.W,
        T=int(cfg.T),
        img_channels=20,
        meta_dim=10,
        st_rope2d=cfg.model.st_rope2d,
        st_axial2d=cfg.model.st_axial2d,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=float(cfg.lr))
    mcts_policy = None
    if getattr(cfg, 'mcts', None) is not None and bool(getattr(cfg.mcts, 'enabled', False)):
        mcts_policy = AlphaZeroMCTSPolicy(policy, cfg.mcts, device=device, T=int(cfg.T))
        print(f"[mcts] enabled: actor_mode={cfg.mcts.actor_mode}, sims={cfg.mcts.num_simulations}")

    opponent_pool = []
    ep_count = 0
    start_update = 1

    def _terminal_stats_from_env(env_):
        e = env_
        owner = e.owner
        army = e.army
        tile = e.tile_type

        o0 = int(Owner.P0)
        o1 = int(Owner.P1)
        t_city = int(TileType.CITY)

        m0 = (owner == o0)
        m1 = (owner == o1)

        p0_land = int(m0.sum())
        p1_land = int(m1.sum())
        p0_army = int(army[m0].sum())
        p1_army = int(army[m1].sum())

        city_mask = (tile == t_city)
        p0_cities = int((city_mask & m0).sum())
        p1_cities = int((city_mask & m1).sum())

        return p0_army, p1_army, p0_land, p1_land, p0_cities, p1_cities

    # ---- resume ----
    if cfg.resume_path is not None:
        ck = load_checkpoint(
            cfg.resume_path,
            policy,
            optimizer,
            device=device,
            opponent_pool_max=int(cfg.opponent.opponent_pool_max),
        )
        opponent_pool = ck["opponent_pool"]
        ep_count = ck["ep_count"]
        win_count = ck["win_count"]
        lose_count = ck["lose_count"]
        start_update = ck["update"] + 1

        if ck.get("rng_state") is not None:
            set_rng_state(ck["rng_state"])

        for i in range(num_envs):
            envs[i], obs0_list[i], obs1_list[i] = reset_episode(envs[i], cfg.env, seeder)

        print(f"[resume] loaded {cfg.resume_path} at update={ck['update']} pool={len(opponent_pool)}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if cfg.save_resolved_config:
        from .config import save_resolved_config
        save_resolved_config(cfg, cfg.ckpt_dir)

    config_for_ckpt = cfg.to_dict()
    config_for_ckpt["bootstrap_gae"] = True
    config_for_ckpt["vary_map_across_resets"] = True

    for upd in range(start_update, int(cfg.total_updates) + 1):

        term_n = 0
        term_sum_p0_army = 0
        term_sum_p1_army = 0
        term_sum_p0_land = 0
        term_sum_p1_land = 0
        term_sum_p0_cities = 0
        term_sum_p1_cities = 0
        win_count = 0
        lose_count = 0

        buf = RolloutBuffer(device=device)

        do_viz = bool(cfg.viz.enable) and (upd % int(cfg.viz.every_updates) == 0)
        if do_viz and bool(cfg.viz.reset_episode_before_viz):
            for i in range(num_envs):
                envs[i], obs0_list[i], obs1_list[i] = reset_episode(envs[i], cfg.env, seeder)

        vs = viz_begin_update(
            do_viz=do_viz,
            out_dir=str(cfg.viz.out_dir),
            frames_per_update=int(cfg.viz.frames_per_update),  # GLOBAL cap
            rollout_len=int(cfg.rollout_len),
            save_mp4=bool(cfg.viz.save_mp4),
            mp4_fps=int(cfg.viz.mp4_fps),
            save_trace_jsonl=bool(cfg.viz.save_trace_jsonl),
            cell=int(cfg.viz.cell),
            draw_text=bool(cfg.viz.draw_text),
            pov_player=int(cfg.viz.pov_player),
            upd=upd,
            topk_actions=int(cfg.viz.topk_actions),
            tag="",
        )

        # ---- choose opponent ----
        opp = None
        if len(opponent_pool) == 0:
            opp = policy if str(cfg.opponent.pool_empty_mode).lower() == "self" else None
        else:
            if random.random() < float(cfg.opponent.pool_pick_prob):
                opp = random.choice(opponent_pool)
            else:
                opp = policy if str(cfg.opponent.fallback_mode).lower() == "self" else None

        for step in tqdm(range(int(cfg.rollout_len)), desc=f"rollout upd {upd}", dynamic_ncols=True):

            phi_s_list = [
                phi_from_scores(obs0_list[i], cfg.reward_shaping.w_army, cfg.reward_shaping.w_land)
                for i in range(num_envs)
            ]

            x_img_seq_list = []
            x_meta_seq_list = []
            mask0_t_list = []
            T_lens = []

            for i in range(num_envs):
                img_np, meta_np = envs[i].get_model_obs_seq(Owner.P0, padded=bool(cfg.seq_padding))
                x_img_seq_i = torch.from_numpy(img_np)   # (t,C,H,W)
                x_meta_seq_i = torch.from_numpy(meta_np) # (t,M)
                x_img_seq_list.append(x_img_seq_i)
                x_meta_seq_list.append(x_meta_seq_i)
                T_lens.append(int(x_img_seq_i.shape[0]))

                mask_np = envs[i].legal_action_mask(Owner.P0)
                mask0_t_list.append(torch.from_numpy(mask_np).to(torch.bool))

            T_step = max(T_lens)
            x_img_step = torch.stack([_pad_first_time(x, T_step) for x in x_img_seq_list], dim=0)
            x_meta_step = torch.stack([_pad_first_time(x, T_step) for x in x_meta_seq_list], dim=0)
            mask0_step = torch.stack(mask0_t_list, dim=0)

            x_img0 = x_img_step.to(device)
            x_meta0 = x_meta_step.to(device)
            mask0 = mask0_step.to(device)
            if mcts_policy is not None and str(getattr(cfg.mcts, 'actor_mode', 'ppo')).lower() == 'mcts':
                # MCTS action selection (per-env, sequential)
                a0_list = []
                for i in range(len(envs)):
                    a_i, _ = mcts_policy.select_action(envs[i], player=Owner.P0)
                    a0_list.append(int(a_i))
                a0 = torch.tensor(a0_list, device=device, dtype=torch.long)
                logp0, _, v0 = policy.evaluate_actions(x_img0, x_meta0, mask0, a0)
            else:
                a0, logp0, v0, _ = policy.act(x_img0, x_meta0, mask0)

            a1_list = choose_opponent_action_batched(
                envs,
                opp,
                device=device,
                random_prob=float(cfg.opponent.random_opp_prob),
                seq_padding=bool(cfg.seq_padding),
            )

            r_list: List[float] = []
            done_list: List[bool] = []
            a0_list_int: List[int] = []
            a1_list_int: List[int] = []
            v_list_float: List[float] = []
            logp_list_float: List[float] = []

            for i in range(num_envs):
                a0_i = int(a0[i].item())
                a1_i = int(a1_list[i])

                res = envs[i].step(a0_i, a1_i)
                obs0_next, obs1_next = res.obs
                done = bool(res.terminated or res.truncated)

                phi_sp = phi_from_scores(obs0_next, cfg.reward_shaping.w_army, cfg.reward_shaping.w_land)
                env_r0, _ = res.reward
                r = potential_shaped_reward(env_r0, phi_s_list[i], phi_sp, gamma=float(cfg.gamma), done=done)

                r_list.append(float(r))
                done_list.append(done)
                a0_list_int.append(a0_i)
                a1_list_int.append(a1_i)
                v_list_float.append(float(v0[i].item()))
                logp_list_float.append(float(logp0[i].item()))

                obs0_list[i], obs1_list[i] = obs0_next, obs1_next
                if done:
                    ep_count += 1

                    if res.terminated:
                        if res.info.get("winner", None) == int(Owner.P0):
                            win_count += 1
                        if res.info.get("winner", None) == int(Owner.P1):
                            lose_count += 1

                    if res.terminated or res.truncated:
                        p0a, p1a, p0l, p1l, p0c, p1c = _terminal_stats_from_env(envs[i])
                        term_n += 1
                        term_sum_p0_army += p0a
                        term_sum_p1_army += p1a
                        term_sum_p0_land += p0l
                        term_sum_p1_land += p1l
                        term_sum_p0_cities += p0c
                        term_sum_p1_cities += p1c

                    envs[i], obs0_list[i], obs1_list[i] = reset_episode(envs[i], cfg.env, seeder)

            buf.add_step(
                x_img_seq=x_img_step,
                x_meta_seq=x_meta_step,
                maskA=mask0_step,
                a=a0.detach().cpu(),
                logp=logp0.detach().cpu(),
                v=v0.detach().cpu(),
                r=torch.tensor(r_list, dtype=torch.float32),
                done=torch.tensor([1.0 if d else 0.0 for d in done_list], dtype=torch.float32),
            )

            if do_viz:
                maybe_visualize_rollout_step_concat(
                    vs,
                    envs=envs,
                    upd=upd,
                    step=step,
                    a0_list=a0_list_int,
                    a1_list=a1_list_int,
                    r_list=r_list,
                    v_list=v_list_float,
                    logp_list=logp_list_float,
                    done_list=done_list,
                    policy=policy,
                    x_img_batch=x_img0,
                    x_meta_batch=x_meta0,
                    legal_mask_batch=mask0,
                )

        viz_end_update(vs, upd, tag="")

        # ---- bootstrap V(s_T) ----
        x_img_seq_list = []
        x_meta_seq_list = []
        T_lens = []
        for i in range(num_envs):
            img_np, meta_np = envs[i].get_model_obs_seq(Owner.P0, padded=bool(cfg.seq_padding))
            x_img_seq_i = torch.from_numpy(img_np)
            x_meta_seq_i = torch.from_numpy(meta_np)
            x_img_seq_list.append(x_img_seq_i)
            x_meta_seq_list.append(x_meta_seq_i)
            T_lens.append(int(x_img_seq_i.shape[0]))

        T_step = max(T_lens)
        x_img_last = torch.stack([_pad_first_time(x, T_step) for x in x_img_seq_list], dim=0).to(device)
        x_meta_last = torch.stack([_pad_first_time(x, T_step) for x in x_meta_seq_list], dim=0).to(device)

        with torch.no_grad():
            _, v_last = policy.logits_and_value(x_img_last, x_meta_last)

        last_v = v_last.detach().to(device).view(-1)

        batch = buf.build(gamma=float(cfg.gamma), lam=float(cfg.lam), last_v=last_v)
        ppo_update(policy, optimizer, batch, cfg=cfg.ppo, device=device)

        # ---- snapshot opponent ----
        if upd % int(cfg.opponent.snapshot_every) == 0:
            snap = copy.deepcopy(policy).to(device)
            snap.eval()
            opponent_pool.append(snap)
            if len(opponent_pool) > int(cfg.opponent.opponent_pool_max):
                opponent_pool.pop(0)

        # ---- checkpoint ----
        if upd % int(cfg.save_every) == 0 or upd == int(cfg.total_updates):
            rng_state = get_rng_state()
            ckpt_path = os.path.join(cfg.ckpt_dir, f"ckpt_{upd:06d}.pt")
            save_checkpoint(
                ckpt_path,
                policy,
                optimizer,
                update=upd,
                ep_count=ep_count,
                win_count=win_count,
                lose_count=lose_count,
                opponent_pool=opponent_pool,
                rng_state=rng_state,
                config=config_for_ckpt,
            )

            if bool(cfg.video.make_video):
                try:
                    out_mp4 = make_video_for_checkpoint(
                        ckpt_path=ckpt_path,
                        videos_dir=cfg.video.videos_dir,
                        T=int(cfg.T),
                        fps=int(cfg.video.fps),
                        seed=int(cfg.video.seed),
                        max_halfturns=int(cfg.video.max_halfturns),
                        cell=int(cfg.video.cell),
                        draw_text=bool(cfg.video.draw_text),
                        pov_player=int(cfg.video.pov_player),
                    )
                    print(f"[video] saved {out_mp4}")
                except Exception as e:
                    print(f"[video] failed for {ckpt_path}: {e}")

            latest_path = os.path.join(cfg.ckpt_dir, "latest.pt")
            save_checkpoint(
                latest_path,
                policy,
                optimizer,
                update=upd,
                ep_count=ep_count,
                win_count=win_count,
                lose_count=lose_count,
                opponent_pool=opponent_pool,
                rng_state=rng_state,
                config=config_for_ckpt,
            )
            print(f"[ckpt] saved {ckpt_path} (and latest.pt)")

        # ---- logging ----
        if int(cfg.log_every) > 0 and upd % int(cfg.log_every) == 0:
            win_rate = win_count / max(1, term_n)
            lose_rate = lose_count / max(1, term_n)
            msg = f"[upd {upd:4d}] episodes={ep_count:6d} win_rate={win_rate:.3f} lose_rate={lose_rate:.3f} pool={len(opponent_pool)}"

            if term_n > 0:
                avg_p0_army = term_sum_p0_army / term_n
                avg_p0_land = term_sum_p0_land / term_n
                avg_p0_city = term_sum_p0_cities / term_n
                avg_p1_army = term_sum_p1_army / term_n
                avg_p1_land = term_sum_p1_land / term_n
                avg_p1_city = term_sum_p1_cities / term_n
                msg += (
                    f" | terminal_avg(n={term_n}) "
                    f"P0(a/l/c)={avg_p0_army:.1f}/{avg_p0_land:.1f}/{avg_p0_city:.2f} "
                    f"P1(a/l/c)={avg_p1_army:.1f}/{avg_p1_land:.1f}/{avg_p1_city:.2f}"
                )
            else:
                msg += " | terminal_avg(n=0)"

            print(msg)

    torch.save(policy.state_dict(), cfg.save_policy_path)
    print(f"Saved to {cfg.save_policy_path}")
