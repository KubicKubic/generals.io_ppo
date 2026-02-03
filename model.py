# memory_rl_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# History buffer (stores last N obs dicts per player)
# -----------------------------

class ObsHistory:
    """
    Stores last N observations (dicts) and can return a padded sequence.
    """
    def __init__(self, max_len: int = 100):
        self.max_len = int(max_len)
        self.buf: List[Dict[str, np.ndarray]] = []

    def reset(self, first_obs: Dict[str, np.ndarray]):
        self.buf = [first_obs]

    def push(self, obs: Dict[str, np.ndarray]):
        self.buf.append(obs)
        if len(self.buf) > self.max_len:
            self.buf = self.buf[-self.max_len :]

    def get_seq(self) -> List[Dict[str, np.ndarray]]:
        return self.buf

    def get_padded_seq(self) -> List[Dict[str, np.ndarray]]:
        """
        If fewer than max_len, left-pad by repeating the earliest obs.
        This keeps shapes consistent and avoids having to invent a blank obs.
        """
        if not self.buf:
            raise RuntimeError("ObsHistory is empty. Call reset() first.")
        if len(self.buf) >= self.max_len:
            return self.buf[-self.max_len :]
        pad = [self.buf[0]] * (self.max_len - len(self.buf))
        return pad + self.buf


# -----------------------------
# Encoding: sequence of obs -> tensors
# -----------------------------

def encode_obs_sequence(
    obs_seq: List[Dict[str, np.ndarray]],
    player_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a sequence of observations into:
      x_img: (T, C, H, W)
      x_meta: (T, M)

    Per-frame image channels (C=20):
      - tile one-hot: EMPTY/MOUNTAIN/CITY/GENERAL (4)  [tile_type already fog-rendered]
      - owner one-hot on visible cells: self/enemy/neutral (3)
      - armies on visible cells (log1p): self_army, enemy_army (2)
      - visible mask (1)
      - memory_tag one-hot (0..7) (8)
      - exposed_my_city_mask (1)
      - (optional) you can add more channels later

      Total C = 4 + 3 + 2 + 1 + 8 + 1 = 19
      We add 1 extra "bias" channel (all ones) to help CNN (so C=20).

    Per-frame meta (M=10):
      - half_in_turn
      - turn_frac_in_round (turn % 25)/25
      - log1p(my_total_army)/8, log1p(opp_total_army)/8
      - my_total_land/(25*25), opp_total_land/(25*25)
      - real_H/25, real_W/25
      - exposed_my_general_seen (0/1)
      - exposed_my_city_count/(25*25)
    """
    T = len(obs_seq)
    if T == 0:
        raise ValueError("obs_seq is empty")

    # shapes
    H, W = obs_seq[0]["tile_type"].shape
    self_id = int(player_id)
    opp_id = 1 - self_id

    # allocate
    C = 20
    M = 10
    x_img = np.zeros((T, C, H, W), dtype=np.float32)
    x_meta = np.zeros((T, M), dtype=np.float32)

    for t in range(T):
        obs = obs_seq[t]
        tile = obs["tile_type"].astype(np.int32)     # fog-rendered
        owner = obs["owner"].astype(np.int32)        # -2 in fog
        army = obs["army"].astype(np.int32)          # -1 in fog
        vis = obs["visible"].astype(np.float32)

        mem = obs.get("memory_tag", np.zeros((H, W), dtype=np.int8)).astype(np.int32)  # 0..7
        exposed_city = obs.get("exposed_my_city_mask", np.zeros((H, W), dtype=np.bool_)).astype(np.float32)

        # tile one-hot (4)
        for tt in range(4):
            x_img[t, tt] = (tile == tt).astype(np.float32)

        # owner one-hot on visible (3)
        x_img[t, 4] = ((owner == self_id) & (vis > 0)).astype(np.float32)
        x_img[t, 5] = ((owner == opp_id) & (vis > 0)).astype(np.float32)
        x_img[t, 6] = ((owner == -1) & (vis > 0)).astype(np.float32)

        # armies on visible (2), log1p
        army_safe = np.maximum(army, 0)
        self_army = np.where(owner == self_id, army_safe, 0)
        opp_army = np.where(owner == opp_id, army_safe, 0)
        x_img[t, 7] = np.log1p(self_army).astype(np.float32)
        x_img[t, 8] = np.log1p(opp_army).astype(np.float32)

        # visible mask (1)
        x_img[t, 9] = vis

        # memory_tag one-hot (8): channels 10..17 correspond to tag 0..7
        for k in range(8):
            x_img[t, 10 + k] = (mem == k).astype(np.float32)

        # exposed_my_city_mask (1)
        x_img[t, 18] = exposed_city

        # bias channel (all ones)
        x_img[t, 19] = 1.0

        # meta (10)
        half_in_turn = float(obs["half_in_turn"][0])
        turn = int(obs["turn"][0])
        turn_frac = float(turn % 25) / 25.0

        total_army = obs["total_army"].astype(np.float32)  # [P0,P1]
        total_land = obs["total_land"].astype(np.float32)  # [P0,P1]
        my_army_total = float(total_army[self_id])
        opp_army_total = float(total_army[opp_id])
        my_land_total = float(total_land[self_id])
        opp_land_total = float(total_land[opp_id])

        real_shape = obs["real_shape"].astype(np.float32)  # [real_H, real_W]
        real_h = float(real_shape[0]) / 25.0
        real_w = float(real_shape[1]) / 25.0

        exposed_my_general_seen = float(obs.get("exposed_my_general_seen", np.array([0], dtype=np.bool_))[0])
        exposed_my_city_count = float(obs.get("exposed_my_city_count", np.array([0], dtype=np.int32))[0]) / (25.0 * 25.0)

        x_meta[t] = np.array(
            [
                half_in_turn,
                turn_frac,
                np.log1p(my_army_total) / 8.0,
                np.log1p(opp_army_total) / 8.0,
                my_land_total / (25.0 * 25.0),
                opp_land_total / (25.0 * 25.0),
                real_h,
                real_w,
                exposed_my_general_seen,
                exposed_my_city_count,
            ],
            dtype=np.float32,
        )

    return torch.from_numpy(x_img), torch.from_numpy(x_meta)


# -----------------------------
# Model: per-frame CNN -> sequence Transformer -> policy/value
# -----------------------------

class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int = 20, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*T, C, H, W)
        returns: (B*T, out_dim)
        """
        h = self.conv(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        return self.fc(h)


class SeqPPOPolicy(nn.Module):
    """
    Input is a window of last T=100 half-turn observations:
      x_img: (B, T, C, H, W)
      x_meta: (B, T, M)

    We encode each frame -> embedding, then process sequence via Transformer.
    We use the LAST token representation for policy/value.
    """
    def __init__(
        self,
        action_size: int,
        T: int = 100,
        img_channels: int = 20,
        meta_dim: int = 10,
        frame_dim: int = 256,
        model_dim: int = 320,
        nhead: int = 8,
        nlayers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_size = int(action_size)
        self.T = int(T)

        self.frame_encoder = FrameEncoder(img_channels, frame_dim)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.proj = nn.Linear(frame_dim + 64, model_dim)

        # positional embedding for T steps
        self.pos = nn.Parameter(torch.zeros(1, T, model_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.head = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, self.action_size)
        self.v = nn.Linear(256, 1)

    def forward(self, x_img: torch.Tensor, x_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_img: (B,T,C,H,W)
        x_meta: (B,T,M)
        returns:
          logits: (B, action_size)
          value: (B,)
        """
        B, T, C, H, W = x_img.shape
        assert T == self.T, f"Expected T={self.T}, got T={T}"

        # encode frames
        x_img2 = x_img.reshape(B * T, C, H, W)
        f = self.frame_encoder(x_img2).reshape(B, T, -1)  # (B,T,frame_dim)

        m = self.meta_mlp(x_meta.reshape(B * T, -1)).reshape(B, T, -1)  # (B,T,64)

        z = self.proj(torch.cat([f, m], dim=-1))  # (B,T,model_dim)
        z = z + self.pos  # add learned positions

        z = self.tr(z)  # (B,T,model_dim)
        last = z[:, -1, :]  # last time step representation

        h = self.head(last)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value

    def _masked_dist(self, logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.distributions.Categorical:
        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        return torch.distributions.Categorical(logits=masked_logits)

    @torch.no_grad()
    def act(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_mask: torch.Tensor):
        logits, v = self.forward(x_img, x_meta)
        dist = self._masked_dist(logits, legal_mask)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, v, ent

    def evaluate_actions(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_mask: torch.Tensor, actions: torch.Tensor):
        logits, v = self.forward(x_img, x_meta)
        dist = self._masked_dist(logits, legal_mask)
        logp = dist.log_prob(actions)
        ent = dist.entropy()
        return logp, ent, v
