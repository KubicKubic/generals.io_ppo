# memory_ppo_train.py
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from generals_env_with_memory import GeneralsEnvWithMemory
from generals_env import Owner
from model import ObsHistory, encode_obs_sequence, SeqPPOPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Batch:
    x_img: torch.Tensor   # (N,T,C,H,W)
    x_meta: torch.Tensor  # (N,T,M)
    mask: torch.Tensor    # (N,A) bool
    a: torch.Tensor       # (N,)
    logp: torch.Tensor    # (N,)
    v: torch.Tensor       # (N,)
    r: torch.Tensor       # (N,)
    done: torch.Tensor    # (N,)
    adv: torch.Tensor     # (N,)
    ret: torch.Tensor     # (N,)


class RolloutBuffer:
    """
    Stores PPO samples.
    We store the full encoded sequence tensors per step. This is heavier,
    so keep rollout_len moderate (e.g., 64~256) unless you optimize storage.
    """
    def __init__(self):
        self.x_img: List[torch.Tensor] = []
        self.x_meta: List[torch.Tensor] = []
        self.mask: List[torch.Tensor] = []
        self.a: List[torch.Tensor] = []
        self.logp: List[torch.Tensor] = []
        self.v: List[torch.Tensor] = []
        self.r: List[float] = []
        self.done: List[float] = []

    def add(
        self,
        x_img_seq: torch.Tensor,   # (T,C,H,W) CPU
        x_meta_seq: torch.Tensor,  # (T,M) CPU
        mask: torch.Tensor,        # (A,) CPU bool
        a: torch.Tensor,           # scalar
        logp: torch.Tensor,        # scalar
        v: torch.Tensor,           # scalar
        r: float,
        done: bool,
    ):
        self.x_img.append(x_img_seq)
        self.x_meta.append(x_meta_seq)
        self.mask.append(mask)
        self.a.append(a)
        self.logp.append(logp)
        self.v.append(v)
        self.r.append(float(r))
        self.done.append(1.0 if done else 0.0)

    def build(self, gamma=0.99, lam=0.95) -> Batch:
        x_img = torch.stack(self.x_img).to(DEVICE)     # (N,T,C,H,W)
        x_meta = torch.stack(self.x_meta).to(DEVICE)   # (N,T,M)
        mask = torch.stack(self.mask).to(DEVICE)       # (N,A)
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

        return Batch(x_img, x_meta, mask, a, logp, v, r, done, adv, ret)


def shaping_from_scores(obs0: Dict[str, np.ndarray]) -> float:
    """
    Small dense shaping (kept tiny; win/loss dominates):
      + land advantage, + army advantage
    Uses scoreboard which is global in obs.
    """
    total_army = obs0["total_army"].astype(np.float32)  # [P0,P1]
    total_land = obs0["total_land"].astype(np.float32)  # [P0,P1]
    army_adv = float(total_army[0] - total_army[1])
    land_adv = float(total_land[0] - total_land[1])
    return 0.00025 * army_adv + 0.0015 * land_adv


@torch.no_grad()
def choose_opponent_action(
    env: GeneralsEnvWithMemory,
    opp_policy: Optional[SeqPPOPolicy],
    opp_hist: ObsHistory,
    random_prob: float = 0.2,
    T: int = 100,
) -> int:
    """
    Choose P1 action.
    - With prob random_prob: random legal action.
    - Else use opp_policy on last T obs sequence for P1.
    """
    mask_np = env.legal_action_mask(Owner.P1)
    legal = np.flatnonzero(mask_np)
    if opp_policy is None or random.random() < random_prob:
        return int(np.random.choice(legal))

    seq = opp_hist.get_padded_seq()
    x_img_seq, x_meta_seq = encode_obs_sequence(seq, player_id=1)
    x_img = x_img_seq.unsqueeze(0).to(DEVICE)   # (1,T,C,H,W)
    x_meta = x_meta_seq.unsqueeze(0).to(DEVICE) # (1,T,M)
    mask = torch.from_numpy(mask_np).unsqueeze(0).to(DEVICE)

    a, _, _, _ = opp_policy.act(x_img, x_meta, mask)
    return int(a.item())


def ppo_update(
    policy: SeqPPOPolicy,
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
                batch.mask[j],
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


def train(
    seed: int = 0,
    total_updates: int = 1200,
    rollout_len: int = 128,   # keep moderate due to large sequence inputs
    T: int = 100,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    snapshot_every: int = 25,
    opponent_pool_max: int = 12,
    random_opp_prob: float = 0.25,
):
    set_seed(seed)

    env = GeneralsEnvWithMemory(seed=seed)
    (obs0, obs1), _ = env.reset()

    # histories
    h0 = ObsHistory(max_len=T)
    h1 = ObsHistory(max_len=T)
    h0.reset(obs0)
    h1.reset(obs1)

    policy = SeqPPOPolicy(action_size=env.action_size, T=T).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    opponent_pool: List[SeqPPOPolicy] = []

    ep_count = 0
    win_count = 0

    for upd in range(1, total_updates + 1):
        buf = RolloutBuffer()

        # pick opponent snapshot with 80% prob
        opp = None
        if opponent_pool and random.random() < 0.8:
            opp = random.choice(opponent_pool)

        for _ in range(rollout_len):
            # build P0 sequence input
            seq0 = h0.get_padded_seq()
            x_img0_seq, x_meta0_seq = encode_obs_sequence(seq0, player_id=0)
            # store CPU versions in buffer, but policy needs GPU now
            x_img0 = x_img0_seq.unsqueeze(0).to(DEVICE)   # (1,T,C,H,W)
            x_meta0 = x_meta0_seq.unsqueeze(0).to(DEVICE) # (1,T,M)

            # action mask
            mask0_np = env.legal_action_mask(Owner.P0)
            mask0 = torch.from_numpy(mask0_np).unsqueeze(0).to(DEVICE)

            # act
            a0, logp0, v0, _ = policy.act(x_img0, x_meta0, mask0)

            # opponent act
            a1 = choose_opponent_action(env, opp, h1, random_prob=random_opp_prob, T=T)

            # step one half-turn
            res = env.step(int(a0.item()), int(a1))
            (obs0, obs1) = res.obs
            done = bool(res.terminated or res.truncated)

            # update histories with NEW obs
            h0.push(obs0)
            h1.push(obs1)

            # reward
            r0, _ = res.reward
            r = float(r0) + shaping_from_scores(obs0)

            # store sample (store sequence tensors on CPU to reduce GPU memory)
            buf.add(
                x_img_seq=x_img0_seq,                      # (T,C,H,W) CPU
                x_meta_seq=x_meta0_seq,                    # (T,M) CPU
                mask=torch.from_numpy(mask0_np),           # (A,) CPU
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

        # snapshot
        if upd % snapshot_every == 0:
            snap = copy.deepcopy(policy).to(DEVICE)
            snap.eval()
            opponent_pool.append(snap)
            if len(opponent_pool) > opponent_pool_max:
                opponent_pool.pop(0)

        if upd % 10 == 0:
            wr = win_count / max(1, ep_count)
            print(f"[upd {upd:4d}] episodes={ep_count:6d} win_rate={wr:.3f} pool={len(opponent_pool)}")

    torch.save(policy.state_dict(), "generals_seq_policy.pt")
    print("Saved to generals_seq_policy.pt")


if __name__ == "__main__":
    train()
