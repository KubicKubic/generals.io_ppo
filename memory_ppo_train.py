# memory_ppo_train.py
from __future__ import annotations

import os
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from generals_env_memory import GeneralsEnvWithMemory
from generals_env import Owner
from memory_rl_model import ObsHistory, encode_obs_sequence, SeqPPOPolicyRoPEFactorized

from checkpoint_video import make_video_for_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# RNG state helpers
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_rng_state() -> Dict:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state: Dict):
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and "torch_cuda_random" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


# -----------------------------
# PPO batch/buffer
# -----------------------------

@dataclass
class Batch:
    x_img: torch.Tensor   # (N,T,C,H,W)
    x_meta: torch.Tensor  # (N,T,M)
    maskA: torch.Tensor   # (N,A) bool
    a: torch.Tensor       # (N,)
    logp: torch.Tensor    # (N,)
    v: torch.Tensor       # (N,)
    r: torch.Tensor       # (N,)
    done: torch.Tensor    # (N,)
    adv: torch.Tensor     # (N,)
    ret: torch.Tensor     # (N,)

class RolloutBuffer:
    def __init__(self):
        self.x_img: List[torch.Tensor] = []
        self.x_meta: List[torch.Tensor] = []
        self.maskA: List[torch.Tensor] = []
        self.a: List[torch.Tensor] = []
        self.logp: List[torch.Tensor] = []
        self.v: List[torch.Tensor] = []
        self.r: List[float] = []
        self.done: List[float] = []

    def add(
        self,
        x_img_seq: torch.Tensor,
        x_meta_seq: torch.Tensor,
        maskA: torch.Tensor,
        a: torch.Tensor,
        logp: torch.Tensor,
        v: torch.Tensor,
        r: float,
        done: bool,
    ):
        self.x_img.append(x_img_seq)
        self.x_meta.append(x_meta_seq)
        self.maskA.append(maskA)
        self.a.append(a)
        self.logp.append(logp)
        self.v.append(v)
        self.r.append(float(r))
        self.done.append(1.0 if done else 0.0)

    def build(self, gamma=0.99, lam=0.95) -> Batch:
        x_img = torch.stack(self.x_img).to(DEVICE)
        x_meta = torch.stack(self.x_meta).to(DEVICE)

        maskA = torch.stack(self.maskA).to(DEVICE)
        if maskA.dtype != torch.bool:
            maskA = maskA.to(torch.bool)

        a = torch.stack(self.a).to(DEVICE).long().view(-1)
        logp = torch.stack(self.logp).to(DEVICE).view(-1)
        v = torch.stack(self.v).to(DEVICE).view(-1)

        r = torch.tensor(self.r, dtype=torch.float32, device=DEVICE)
        done = torch.tensor(self.done, dtype=torch.float32, device=DEVICE)

        # GAE
        N = r.shape[0]
        adv = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        last_gae = 0.0
        next_v = 0.0
        for t in reversed(range(N)):
            nonterminal = 1.0 - done[t]
            delta = r[t] + gamma * next_v * nonterminal - v[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
            next_v = v[t]
        ret = adv + v

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return Batch(x_img, x_meta, maskA, a, logp, v, r, done, adv, ret)


# -----------------------------
# Potential-based shaping
# -----------------------------

def phi_from_scores(obs0: Dict[str, np.ndarray]) -> float:
    """
    Potential function Phi(s), based on global scoreboard in obs.
    Keep same magnitudes as your previous shaping weights.
    """
    total_army = obs0["total_army"].astype(np.float32)  # [P0,P1]
    total_land = obs0["total_land"].astype(np.float32)  # [P0,P1]
    army_adv = float(total_army[0] - total_army[1])
    land_adv = float(total_land[0] - total_land[1])
    return 0.00025 * army_adv + 0.0015 * land_adv


def potential_shaped_reward(
    env_r0: float,
    phi_s: float,
    phi_sp: float,
    gamma: float,
    done: bool,
) -> float:
    """
    r' = r_env + gamma * Phi(s') - Phi(s)
    If done, treat Phi(s') as 0 to avoid leaking terminal shaping beyond episode.
    """
    nonterminal = 0.0 if done else 1.0
    return float(env_r0) + gamma * phi_sp * nonterminal - phi_s


# -----------------------------
# Opponent action
# -----------------------------

@torch.no_grad()
def choose_opponent_action(
    env: GeneralsEnvWithMemory,
    opp_policy: Optional[SeqPPOPolicyRoPEFactorized],
    opp_hist: ObsHistory,
    random_prob: float = 0.25,
    T: int = 100,
) -> int:
    mask_np = env.legal_action_mask(Owner.P1)
    legal = np.flatnonzero(mask_np)
    if opp_policy is None or random.random() < random_prob:
        return int(np.random.choice(legal))

    seq = opp_hist.get_padded_seq()
    x_img_seq, x_meta_seq = encode_obs_sequence(seq, player_id=1)
    x_img = x_img_seq.unsqueeze(0).to(DEVICE)
    x_meta = x_meta_seq.unsqueeze(0).to(DEVICE)
    maskA = torch.from_numpy(mask_np).unsqueeze(0).to(DEVICE).to(torch.bool)

    a, _, _, _ = opp_policy.act(x_img, x_meta, maskA)
    return int(a.item())


# -----------------------------
# PPO update
# -----------------------------

def ppo_update(
    policy: SeqPPOPolicyRoPEFactorized,
    optimizer: optim.Optimizer,
    batch: Batch,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    epochs=4,
    minibatch=16,
):
    N = batch.a.shape[0]
    idx = torch.arange(N, device=DEVICE)

    for _ in range(epochs):
        perm = idx[torch.randperm(N)]
        for start in range(0, N, minibatch):
            j = perm[start : start + minibatch]

            logp_new, ent, v_new = policy.evaluate_actions(
                batch.x_img[j],
                batch.x_meta[j],
                batch.maskA[j],
                batch.a[j],
            )

            ratio = torch.exp(logp_new - batch.logp[j])
            surr1 = ratio * batch.adv[j]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch.adv[j]
            pi_loss = -torch.min(surr1, surr2).mean()

            v_loss = ((v_new - batch.ret[j]) ** 2).mean()
            ent_loss = -ent.mean()

            loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


# -----------------------------
# Checkpointing
# -----------------------------

def save_checkpoint(
    path: str,
    policy: SeqPPOPolicyRoPEFactorized,
    optimizer: optim.Optimizer,
    update: int,
    ep_count: int,
    win_count: int,
    opponent_pool: List[SeqPPOPolicyRoPEFactorized],
    rng_state: Dict,
    config: Dict,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pool_state = [p.state_dict() for p in opponent_pool]
    ckpt = {
        "update": update,
        "ep_count": ep_count,
        "win_count": win_count,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "opponent_pool": pool_state,
        "rng_state": rng_state,
        "config": config,
    }
    torch.save(ckpt, path)

def load_checkpoint(
    path: str,
    policy: SeqPPOPolicyRoPEFactorized,
    optimizer: optim.Optimizer,
    opponent_pool_max: int,
) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    policy.load_state_dict(ckpt["policy"])
    optimizer.load_state_dict(ckpt["optimizer"])

    pool_state = ckpt.get("opponent_pool", [])
    opponent_pool: List[SeqPPOPolicyRoPEFactorized] = []
    for sd in pool_state[-opponent_pool_max:]:
        p = copy.deepcopy(policy).to(DEVICE)
        p.load_state_dict(sd)
        p.eval()
        opponent_pool.append(p)

    if "rng_state" in ckpt:
        set_rng_state(ckpt["rng_state"])

    return {
        "update": int(ckpt.get("update", 0)),
        "ep_count": int(ckpt.get("ep_count", 0)),
        "win_count": int(ckpt.get("win_count", 0)),
        "opponent_pool": opponent_pool,
        "config": ckpt.get("config", {}),
    }


# -----------------------------
# Train loop
# -----------------------------

def train(
    seed: int = 0,
    total_updates: int = 1200,
    rollout_len: int = 128,
    T: int = 100,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    snapshot_every: int = 25,
    opponent_pool_max: int = 12,
    random_opp_prob: float = 0.25,
    ckpt_dir: str = "checkpoints",
    save_every: int = 50,
    resume_path: Optional[str] = None,
    make_video: bool = False,
    videos_dir: str = "videos",
    video_fps: int = 15,
    video_seed: int = 123,
    video_max_halfturns: int = 800,
    video_cell: int = 20,
    video_draw_text: bool = True,
):
    set_seed(seed)

    env = GeneralsEnvWithMemory(seed=seed)
    (obs0, obs1), _ = env.reset()

    h0 = ObsHistory(max_len=T)
    h1 = ObsHistory(max_len=T)
    h0.reset(obs0)
    h1.reset(obs1)

    policy = SeqPPOPolicyRoPEFactorized(action_size=env.action_size, H=env.H, W=env.W, T=T).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    opponent_pool: List[SeqPPOPolicyRoPEFactorized] = []
    ep_count = 0
    win_count = 0
    start_update = 1

    if resume_path is not None:
        ck = load_checkpoint(resume_path, policy, optimizer, opponent_pool_max=opponent_pool_max)
        opponent_pool = ck["opponent_pool"]
        ep_count = ck["ep_count"]
        win_count = ck["win_count"]
        start_update = ck["update"] + 1

        (obs0, obs1), _ = env.reset()
        h0.reset(obs0)
        h1.reset(obs1)
        print(f"[resume] loaded {resume_path} at update={ck['update']} pool={len(opponent_pool)}")

    config = dict(
        seed=seed, total_updates=total_updates, rollout_len=rollout_len, T=T, lr=lr,
        gamma=gamma, lam=lam, snapshot_every=snapshot_every, opponent_pool_max=opponent_pool_max,
        random_opp_prob=random_opp_prob,
        model="RoPE+factorized(src-conditioned)",
        reward="potential-based",
    )

    os.makedirs(ckpt_dir, exist_ok=True)

    for upd in range(start_update, total_updates + 1):
        buf = RolloutBuffer()

        opp = None
        if opponent_pool and random.random() < 0.8:
            opp = random.choice(opponent_pool)

        for _ in range(rollout_len):
            # current potential
            phi_s = phi_from_scores(obs0)

            # build P0 sequence
            seq0 = h0.get_padded_seq()
            x_img0_seq, x_meta0_seq = encode_obs_sequence(seq0, player_id=0)

            x_img0 = x_img0_seq.unsqueeze(0).to(DEVICE)
            x_meta0 = x_meta0_seq.unsqueeze(0).to(DEVICE)

            # legal mask
            mask0_np = env.legal_action_mask(Owner.P0)
            mask0 = torch.from_numpy(mask0_np).unsqueeze(0).to(DEVICE).to(torch.bool)

            # act
            a0, logp0, v0, _ = policy.act(x_img0, x_meta0, mask0)

            # opponent act
            a1 = choose_opponent_action(env, opp, h1, random_prob=random_opp_prob, T=T)

            # step
            res = env.step(int(a0.item()), int(a1))
            (obs0_next, obs1_next) = res.obs
            done = bool(res.terminated or res.truncated)

            # next potential
            phi_sp = phi_from_scores(obs0_next)

            # potential-based reward
            env_r0, _ = res.reward
            r = potential_shaped_reward(env_r0, phi_s, phi_sp, gamma=gamma, done=done)

            # update histories to next obs
            obs0, obs1 = obs0_next, obs1_next
            h0.push(obs0)
            h1.push(obs1)

            # store sample
            buf.add(
                x_img_seq=x_img0_seq,
                x_meta_seq=x_meta0_seq,
                maskA=torch.from_numpy(mask0_np.astype(np.uint8)),
                a=a0.cpu(),
                logp=logp0.cpu(),
                v=v0.cpu(),
                r=r,
                done=done,
            )

            if done:
                ep_count += 1
                if res.terminated and res.info.get("winner", None) == int(Owner.P0):
                    win_count += 1

                (obs0, obs1), _ = env.reset()
                h0.reset(obs0)
                h1.reset(obs1)

        batch = buf.build(gamma=gamma, lam=lam)
        ppo_update(policy, optimizer, batch)

        # snapshot opponent
        if upd % snapshot_every == 0:
            snap = copy.deepcopy(policy).to(DEVICE)
            snap.eval()
            opponent_pool.append(snap)
            if len(opponent_pool) > opponent_pool_max:
                opponent_pool.pop(0)

        # checkpoint
        if upd % save_every == 0 or upd == total_updates:
            rng_state = get_rng_state()
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{upd:06d}.pt")
            save_checkpoint(
                ckpt_path, policy, optimizer,
                update=upd, ep_count=ep_count, win_count=win_count,
                opponent_pool=opponent_pool,
                rng_state=rng_state,
                config=config,
            )
            if make_video:
                try:
                    out_mp4 = make_video_for_checkpoint(
                        ckpt_path=ckpt_path,
                        videos_dir=videos_dir,
                        T=T,
                        fps=video_fps,
                        seed=video_seed,
                        max_halfturns=video_max_halfturns,
                        cell=video_cell,
                        draw_text=video_draw_text,
                    )
                    print(f"[video] saved {out_mp4}")
                except Exception as e:
                    print(f"[video] failed for {ckpt_path}: {e}")
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            save_checkpoint(
                latest_path, policy, optimizer,
                update=upd, ep_count=ep_count, win_count=win_count,
                opponent_pool=opponent_pool,
                rng_state=rng_state,
                config=config,
            )
            print(f"[ckpt] saved {ckpt_path} (and latest.pt)")

        if upd % 10 == 0:
            wr = win_count / max(1, ep_count)
            print(f"[upd {upd:4d}] episodes={ep_count:6d} win_rate={wr:.3f} pool={len(opponent_pool)}")

    torch.save(policy.state_dict(), "generals_seq_policy.pt")
    print("Saved to generals_seq_policy.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_updates", type=int, default=1200)
    p.add_argument("--rollout_len", type=int, default=128)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--snapshot_every", type=int, default=25)
    p.add_argument("--opponent_pool_max", type=int, default=12)
    p.add_argument("--random_opp_prob", type=float, default=0.25)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--make_video", action="store_true", help="Generate an mp4 for each saved checkpoint")
    p.add_argument("--videos_dir", type=str, default="videos")
    p.add_argument("--video_fps", type=int, default=15)
    p.add_argument("--video_seed", type=int, default=123)
    p.add_argument("--video_max_halfturns", type=int, default=800)
    p.add_argument("--video_cell", type=int, default=20)
    p.add_argument("--video_no_text", action="store_true", help="Disable drawing army numbers/text on video")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        seed=args.seed,
        total_updates=args.total_updates,
        rollout_len=args.rollout_len,
        T=args.T,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        snapshot_every=args.snapshot_every,
        opponent_pool_max=args.opponent_pool_max,
        random_opp_prob=args.random_opp_prob,
        ckpt_dir=args.ckpt_dir,
        save_every=args.save_every,
        resume_path=args.resume,
        make_video=args.make_video,
        videos_dir=args.videos_dir,
        video_fps=args.video_fps,
        video_seed=args.video_seed,
        video_max_halfturns=args.video_max_halfturns,
        video_cell=args.video_cell,
        video_draw_text=(not args.video_no_text),
    )
