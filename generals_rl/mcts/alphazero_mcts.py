
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import copy
import hashlib
import math
import random

import numpy as np
import torch

from ..env.generals_env import Owner
from ..data.encoding import encode_obs_sequence
from ..data.history import ObsHistory


@dataclass
class MCTSConfig:
    """
    AlphaZero-style (PUCT) MCTS for this codebase.

    We model one MCTS "ply" as:
        our_action (for `player`) + opponent_action (via opponent_model) -> env.step -> next state

    That matches the current env.step(a0, a1) API.

    WARNING (important for PPO training):
      If you set actor_mode="mcts" during PPO rollouts, collected actions are NOT sampled from pi_theta.
      This makes PPO updates off-policy. It's usually fine for evaluation / distillation,
      but do this knowingly.
    """
    enabled: bool = False
    actor_mode: str = "ppo"            # "ppo" | "mcts"
    num_simulations: int = 100
    max_depth: int = 40
    c_puct: float = 1.5

    # root action selection
    tau: float = 1.0                   # temperature for pi_mcts from visit counts (train); set ~0 for eval
    deterministic: bool = False        # True => pick argmax N at root (ignores tau)

    # exploration noise (root only)
    dirichlet_alpha: float = 0.0       # 0 => disabled
    dirichlet_eps: float = 0.0         # 0 => disabled

    # reduce branching factor
    topk_actions: int = 0              # 0 => no restriction; else consider only topK priors per node

    # opponent model used INSIDE search
    opponent_model: str = "policy"     # "policy" | "random"
    opponent_sample: str = "sample"    # "sample" | "argmax"


class _Node:
    __slots__ = ("P", "N", "W", "Q", "expanded", "legal_actions")

    def __init__(self):
        self.P: Dict[int, float] = {}
        self.N: Dict[int, int] = {}
        self.W: Dict[int, float] = {}
        self.Q: Dict[int, float] = {}
        self.expanded: bool = False
        self.legal_actions: List[int] = []


def _softmax_masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = logits.clone()
    x[~mask] = -1e9
    return torch.softmax(x, dim=-1)


def _clone_env(env):
    if hasattr(env, "clone") and callable(getattr(env, "clone")):
        return env.clone()
    return copy.deepcopy(env)


def _clone_hist(h: ObsHistory) -> ObsHistory:
    h2 = ObsHistory(max_len=h.max_len)
    h2.buf = list(h.buf)
    return h2


def _state_key(env) -> bytes:
    """
    Hash the underlying true env state to key transposition table.
    This ignores memory wrapper fields (they don't affect dynamics).
    """
    e = env.env if hasattr(env, "env") else env
    h = hashlib.blake2b(digest_size=16)
    h.update(int(e.half_t).to_bytes(4, "little", signed=False))
    # arrays
    h.update(np.ascontiguousarray(e.tile_type).tobytes())
    h.update(np.ascontiguousarray(e.owner).tobytes())
    h.update(np.ascontiguousarray(e.army).tobytes())
    # also include real area offsets if present
    if hasattr(e, "real_r0") and hasattr(e, "real_c0"):
        h.update(int(e.real_r0).to_bytes(2, "little", signed=True))
        h.update(int(e.real_c0).to_bytes(2, "little", signed=True))
    return h.digest()


class AlphaZeroMCTSPolicy:
    """
    AlphaZero-style (PUCT) MCTS action selector using the existing PPO policy net as (prior, value).

    Usage:
        mcts = AlphaZeroMCTSPolicy(policy, cfg.mcts, device=device, T=cfg.T)
        a, pi = mcts.select_action(env, h0, h1, player=Owner.P0)
    """
    def __init__(self, base_policy, cfg: MCTSConfig, *, device: torch.device, T: int):
        self.base_policy = base_policy
        self.cfg = cfg
        self.device = device
        self.T = int(T)
        self._nodes: Dict[bytes, _Node] = {}

    @torch.no_grad()
    def _prior_and_value(self, hist: ObsHistory, *, player_id: int, legal_mask_np: np.ndarray) -> Tuple[np.ndarray, float]:
        seq = hist.get_padded_seq()
        x_img_seq, x_meta_seq = encode_obs_sequence(seq, player_id=player_id)
        x_img = x_img_seq.unsqueeze(0).to(self.device)
        x_meta = x_meta_seq.unsqueeze(0).to(self.device)
        mask = torch.from_numpy(legal_mask_np).to(self.device).to(torch.bool).unsqueeze(0)

        logits, v = self.base_policy.logits_and_value(x_img, x_meta)
        logits = logits[0]
        v = float(v[0].item())

        probs = _softmax_masked(logits, mask[0]).detach().cpu().numpy()

        # optional top-k restriction
        k = int(self.cfg.topk_actions)
        if k > 0:
            legal_idx = np.flatnonzero(legal_mask_np)
            if len(legal_idx) > k:
                legal_probs = probs[legal_idx]
                topk_local = np.argpartition(-legal_probs, k - 1)[:k]
                keep = set(int(legal_idx[i]) for i in topk_local)
                probs2 = np.zeros_like(probs)
                for a in keep:
                    probs2[a] = probs[a]
                s = float(probs2.sum())
                if s > 0:
                    probs = probs2 / s
                else:
                    probs = np.zeros_like(probs)
                    probs[legal_idx] = 1.0 / len(legal_idx)

        return probs, v

    def _apply_root_noise(self, priors: np.ndarray, legal_mask_np: np.ndarray) -> np.ndarray:
        alpha = float(self.cfg.dirichlet_alpha)
        eps = float(self.cfg.dirichlet_eps)
        if alpha <= 0.0 or eps <= 0.0:
            return priors
        legal_idx = np.flatnonzero(legal_mask_np)
        if len(legal_idx) == 0:
            return priors
        noise = np.random.dirichlet([alpha] * len(legal_idx)).astype(np.float32)
        out = priors.copy()
        out[legal_idx] = (1.0 - eps) * out[legal_idx] + eps * noise
        s = float(out[legal_idx].sum())
        if s > 0:
            out[legal_idx] /= s
        return out

    @torch.no_grad()
    def _opponent_action(self, env_sim, h0: ObsHistory, h1: ObsHistory, *, player: Owner) -> int:
        opp_player = Owner.P1 if player == Owner.P0 else Owner.P0
        mask_np = env_sim.legal_action_mask(opp_player)
        legal = np.flatnonzero(mask_np)
        if len(legal) == 0:
            return 0

        if str(self.cfg.opponent_model).lower() == "random":
            return int(random.choice(list(legal)))

        opp_hist = h1 if opp_player == Owner.P1 else h0
        priors, _ = self._prior_and_value(opp_hist, player_id=int(opp_player), legal_mask_np=mask_np)

        if str(self.cfg.opponent_sample).lower() == "argmax":
            return int(np.argmax(priors))
        else:
            return int(np.random.choice(np.arange(len(priors)), p=priors))

    def _get_node(self, key: bytes) -> _Node:
        n = self._nodes.get(key)
        if n is None:
            n = _Node()
            self._nodes[key] = n
        return n

    def _expand_node(self, node: _Node, env_sim, h0: ObsHistory, h1: ObsHistory, *, player: Owner, root: bool) -> float:
        mask_np = env_sim.legal_action_mask(player)
        legal = np.flatnonzero(mask_np)
        node.legal_actions = [int(a) for a in legal]
        if len(node.legal_actions) == 0:
            node.expanded = True
            return 0.0

        hist = h0 if player == Owner.P0 else h1
        priors, v = self._prior_and_value(hist, player_id=int(player), legal_mask_np=mask_np)
        if root:
            priors = self._apply_root_noise(priors, mask_np)

        node.P = {a: float(priors[a]) for a in node.legal_actions}
        node.N = {a: 0 for a in node.legal_actions}
        node.W = {a: 0.0 for a in node.legal_actions}
        node.Q = {a: 0.0 for a in node.legal_actions}
        node.expanded = True
        return v

    def _select_puct(self, node: _Node) -> int:
        total_N = sum(node.N.values()) + 1
        sqrt_total = math.sqrt(total_N)
        c = float(self.cfg.c_puct)

        best_a = node.legal_actions[0]
        best_score = -1e18
        for a in node.legal_actions:
            q = node.Q[a]
            n = node.N[a]
            p = node.P.get(a, 0.0)
            u = c * p * (sqrt_total / (1.0 + n))
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return int(best_a)

    def _backup(self, path: List[Tuple[_Node, int]], value: float):
        v = float(value)
        for node, a in reversed(path):
            node.N[a] += 1
            node.W[a] += v
            node.Q[a] = node.W[a] / node.N[a]

    def _root_pi(self, root: _Node, action_size: int) -> np.ndarray:
        pi = np.zeros((action_size,), dtype=np.float32)
        if len(root.legal_actions) == 0:
            return pi

        tau = float(self.cfg.tau)
        if self.cfg.deterministic or tau <= 1e-6:
            best_a = max(root.legal_actions, key=lambda a: root.N[a])
            pi[best_a] = 1.0
            return pi

        counts = np.array([root.N[a] for a in root.legal_actions], dtype=np.float32)
        counts = np.power(np.maximum(counts, 1e-8), 1.0 / tau)
        s = float(counts.sum())
        if s <= 0:
            for a in root.legal_actions:
                pi[a] = 1.0 / len(root.legal_actions)
        else:
            for a, c in zip(root.legal_actions, counts):
                pi[a] = float(c / s)
        return pi

    def clear_tree(self):
        """Call this if you don't want to reuse nodes across moves."""
        self._nodes.clear()

    def select_action(self, env, h0: ObsHistory, h1: ObsHistory, *, player: Owner) -> Tuple[int, np.ndarray]:
        """
        Returns (action, pi_mcts) for the given player at the current root state.
        """
        root_env = env
        root_key = _state_key(root_env)
        root_node = self._get_node(root_key)

        # expand root if needed
        if not root_node.expanded:
            _ = self._expand_node(root_node, root_env, h0, h1, player=player, root=True)

        action_size = int(root_env.action_size)

        # simulations
        for _ in range(int(self.cfg.num_simulations)):
            env_sim = _clone_env(env)
            h0_sim = _clone_hist(h0)
            h1_sim = _clone_hist(h1)

            path: List[Tuple[_Node, int]] = []
            cur_key = _state_key(env_sim)
            cur_node = self._get_node(cur_key)

            for depth in range(int(self.cfg.max_depth)):
                # terminal?
                # (need to detect terminal by stepping; env doesn't expose done flag directly, so we check via legal mask + max_halfturns is trunc)
                if not cur_node.expanded:
                    leaf_v = self._expand_node(cur_node, env_sim, h0_sim, h1_sim, player=player, root=False)
                    self._backup(path, leaf_v)
                    break

                if len(cur_node.legal_actions) == 0:
                    self._backup(path, 0.0)
                    break

                a = self._select_puct(cur_node)
                path.append((cur_node, a))

                opp_a = self._opponent_action(env_sim, h0_sim, h1_sim, player=player)
                if player == Owner.P0:
                    res = env_sim.step(a, opp_a)
                else:
                    res = env_sim.step(opp_a, a)

                # terminal reward (sparse win/lose)
                done = bool(res.terminated or res.truncated)
                r0, r1 = res.reward
                if done:
                    leaf_v = float(r0 if player == Owner.P0 else r1)
                    self._backup(path, leaf_v)
                    break

                o0, o1 = res.obs
                h0_sim.push(o0)
                h1_sim.push(o1)

                cur_key = _state_key(env_sim)
                cur_node = self._get_node(cur_key)
            else:
                # reached max_depth; bootstrap value
                hist = h0_sim if player == Owner.P0 else h1_sim
                mask_np = env_sim.legal_action_mask(player)
                _, leaf_v = self._prior_and_value(hist, player_id=int(player), legal_mask_np=mask_np)
                self._backup(path, leaf_v)

        pi = self._root_pi(root_node, action_size=action_size)
        if self.cfg.deterministic or float(self.cfg.tau) <= 1e-6:
            return int(np.argmax(pi)), pi

        a = int(np.random.choice(np.arange(action_size), p=pi))
        return a, pi
