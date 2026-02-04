# generals_rl/train/loop.py
from __future__ import annotations

import os
import copy
import random

import torch
import torch.optim as optim

from ..env.generals_env import Owner, TileType
from ..data.history import ObsHistory
from ..data.encoding import encode_obs_sequence
from ..models import make_policy

from .config import TrainConfig
from .rng import set_seed, get_rng_state, set_rng_state
from .buffer import RolloutBuffer
from .reward import phi_from_scores, potential_shaped_reward
from .ppo import ppo_update
from .opponent import choose_opponent_action
from .checkpoint import save_checkpoint, load_checkpoint
from .env_reset import EpisodeSeeder, reset_episode

from ..viz.rollout_viz import viz_begin_update, viz_end_update, maybe_visualize_rollout_step
from ..video.checkpoint_video import make_video_for_checkpoint


def train(cfg: TrainConfig, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.env.base_seed))

    seeder = EpisodeSeeder(cfg.env)
    env = None
    env, obs0, obs1 = reset_episode(env, cfg.env, seeder)

    h0 = ObsHistory(max_len=int(cfg.T))
    h1 = ObsHistory(max_len=int(cfg.T))
    h0.reset(obs0)
    h1.reset(obs1)

    # ---- model selection via config ----
    policy = make_policy(
        cfg.model.name,
        action_size=env.action_size,
        H=env.H,
        W=env.W,
        T=int(cfg.T),
        img_channels=20,
        meta_dim=10,
        rope=cfg.model.rope,
        spatial=cfg.model.spatial,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=float(cfg.lr))

    opponent_pool = []
    ep_count = 0
    win_count = 0
    start_update = 1

    # ---- terminal-only (exclude truncated) averages ----

    def _terminal_stats_from_env(env_):
        """
        Use true env state (no fog): env_.env.*
        Returns: (p0_army, p1_army, p0_land, p1_land, p0_cities, p1_cities)
        """
        e = env_.env
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
        start_update = ck["update"] + 1

        if ck.get("rng_state") is not None:
            set_rng_state(ck["rng_state"])

        env, obs0, obs1 = reset_episode(env, cfg.env, seeder)
        h0.reset(obs0)
        h1.reset(obs1)
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

        buf = RolloutBuffer(device=device)

        # ---- viz ----
        do_viz = bool(cfg.viz.enable) and (upd % int(cfg.viz.every_updates) == 0)
        if do_viz and bool(cfg.viz.reset_episode_before_viz):
            env, obs0, obs1 = reset_episode(env, cfg.env, seeder)
            h0.reset(obs0)
            h1.reset(obs1)

        vs = viz_begin_update(
            do_viz=do_viz,
            out_dir=str(cfg.viz.out_dir),
            frames_per_update=int(cfg.viz.frames_per_update),
            rollout_len=int(cfg.rollout_len),
            save_mp4=bool(cfg.viz.save_mp4),
            mp4_fps=int(cfg.viz.mp4_fps),
            save_trace_jsonl=bool(cfg.viz.save_trace_jsonl),
            cell=int(cfg.viz.cell),
            draw_text=bool(cfg.viz.draw_text),
            pov_player=int(cfg.viz.pov_player),
            upd=upd,
            topk_actions=int(cfg.viz.topk_actions)
        )

        # ---- choose opponent for this update ----
        opp = None
        if len(opponent_pool) == 0:
            opp = policy if str(cfg.opponent.pool_empty_mode).lower() == "self" else None
        else:
            if random.random() < float(cfg.opponent.pool_pick_prob):
                opp = random.choice(opponent_pool)
            else:
                opp = policy if str(cfg.opponent.fallback_mode).lower() == "self" else None

        # ---- rollout ----
        for step in range(int(cfg.rollout_len)):
            phi_s = phi_from_scores(obs0, cfg.reward_shaping.w_army, cfg.reward_shaping.w_land)

            seq0 = h0.get_padded_seq()
            x_img0_seq, x_meta0_seq = encode_obs_sequence(seq0, player_id=0)
            x_img0 = x_img0_seq.unsqueeze(0).to(device)
            x_meta0 = x_meta0_seq.unsqueeze(0).to(device)

            mask0_np = env.legal_action_mask(Owner.P0)
            mask0 = torch.from_numpy(mask0_np).unsqueeze(0).to(device).to(torch.bool)

            a0, logp0, v0, _ = policy.act(x_img0, x_meta0, mask0)

            a1 = choose_opponent_action(
                env,
                opp,
                h1,
                device=device,
                random_prob=float(cfg.opponent.random_opp_prob),
                T=int(cfg.T),
            )

            res = env.step(int(a0.item()), int(a1))
            obs0_next, obs1_next = res.obs
            done = bool(res.terminated or res.truncated)

            phi_sp = phi_from_scores(obs0_next, cfg.reward_shaping.w_army, cfg.reward_shaping.w_land)
            env_r0, _ = res.reward
            r = potential_shaped_reward(env_r0, phi_s, phi_sp, gamma=float(cfg.gamma), done=done)

            buf.add(
                x_img_seq=x_img0_seq,
                x_meta_seq=x_meta0_seq,
                maskA=torch.from_numpy(mask0_np).to(torch.bool),
                a=a0.cpu(),
                logp=logp0.cpu(),
                v=v0.cpu(),
                r=r,
                done=done,
            )

            maybe_visualize_rollout_step(
                vs,
                env,
                upd,
                step,
                a0=int(a0.item()),
                a1=int(a1),
                r=float(r),
                v=float(v0.item()),
                logp=float(logp0.item()),
                done=done,
                policy=policy,
                x_img=x_img0,          # (1,T,C,H,W)
                x_meta=x_meta0,        # (1,T,M)
                legal_mask=mask0,      # (1,A) bool
            )


            obs0, obs1 = obs0_next, obs1_next
            h0.push(obs0)
            h1.push(obs1)

            if done:
                ep_count += 1

                # win count (only meaningful on terminated)
                if res.terminated and res.info.get("winner", None) == int(Owner.P0):
                    win_count += 1

                # ✅ terminal-only stats (exclude truncated episodes)
                if res.terminated or res.truncated:
                    p0a, p1a, p0l, p1l, p0c, p1c = _terminal_stats_from_env(env)
                    term_n += 1
                    term_sum_p0_army += p0a
                    term_sum_p1_army += p1a
                    term_sum_p0_land += p0l
                    term_sum_p1_land += p1l
                    term_sum_p0_cities += p0c
                    term_sum_p1_cities += p1c

                env, obs0, obs1 = reset_episode(env, cfg.env, seeder)
                h0.reset(obs0)
                h1.reset(obs1)

        viz_end_update(vs, upd)

        # ---- bootstrap V(s_T) if rollout ended mid-episode ----
        last_done = bool(buf.done[-1] > 0.5) if len(buf.done) > 0 else True
        if last_done:
            last_v = 0.0
        else:
            seq0 = h0.get_padded_seq()
            x_img0_seq, x_meta0_seq = encode_obs_sequence(seq0, player_id=0)
            x_img0 = x_img0_seq.unsqueeze(0).to(device)
            x_meta0 = x_meta0_seq.unsqueeze(0).to(device)
            mask0_np = env.legal_action_mask(Owner.P0)
            mask0 = torch.from_numpy(mask0_np).unsqueeze(0).to(device).to(torch.bool)
            _, _, v_last, _ = policy.act(x_img0, x_meta0, mask0)
            last_v = float(v_last.item())

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
                opponent_pool=opponent_pool,
                rng_state=rng_state,
                config=config_for_ckpt,
            )
            print(f"[ckpt] saved {ckpt_path} (and latest.pt)")

        # ---- logging ----
        if int(cfg.log_every) > 0 and upd % int(cfg.log_every) == 0:
            wr = win_count / max(1, ep_count)
            msg = f"[upd {upd:4d}] episodes={ep_count:6d} win_rate={wr:.3f} pool={len(opponent_pool)}"

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
