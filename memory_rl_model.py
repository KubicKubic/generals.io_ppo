# memory_rl_model.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# History buffer
# -----------------------------

class ObsHistory:
    """Stores last N observations (dicts) and returns padded sequences."""
    def __init__(self, max_len: int = 100):
        self.max_len = int(max_len)
        self.buf: List[Dict[str, np.ndarray]] = []

    def reset(self, first_obs: Dict[str, np.ndarray]):
        self.buf = [first_obs]

    def push(self, obs: Dict[str, np.ndarray]):
        self.buf.append(obs)
        if len(self.buf) > self.max_len:
            self.buf = self.buf[-self.max_len :]

    def get_padded_seq(self) -> List[Dict[str, np.ndarray]]:
        if not self.buf:
            raise RuntimeError("ObsHistory empty; call reset() first.")
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
      - bias channel (1)

      Total C = 20

    Per-frame meta (M=10):
      - half_in_turn
      - turn_frac_in_round
      - log1p(my_total_army)/8, log1p(opp_total_army)/8
      - my_total_land/(25*25), opp_total_land/(25*25)
      - real_H/25, real_W/25
      - exposed_my_general_seen (0/1)
      - exposed_my_city_count/(25*25)
    """
    T = len(obs_seq)
    if T == 0:
        raise ValueError("obs_seq is empty")

    H, W = obs_seq[0]["tile_type"].shape
    self_id = int(player_id)
    opp_id = 1 - self_id

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

        # owner one-hot (visible only) (3)
        x_img[t, 4] = ((owner == self_id) & (vis > 0)).astype(np.float32)
        x_img[t, 5] = ((owner == opp_id) & (vis > 0)).astype(np.float32)
        x_img[t, 6] = ((owner == -1) & (vis > 0)).astype(np.float32)

        # armies visible-only (2)
        army_safe = np.maximum(army, 0)
        self_army = np.where(owner == self_id, army_safe, 0)
        opp_army = np.where(owner == opp_id, army_safe, 0)
        x_img[t, 7] = np.log1p(self_army).astype(np.float32)
        x_img[t, 8] = np.log1p(opp_army).astype(np.float32)

        # visible (1)
        x_img[t, 9] = vis

        # memory_tag one-hot (8): 10..17 are tags 0..7
        for k in range(8):
            x_img[t, 10 + k] = (mem == k).astype(np.float32)

        # exposed_my_city_mask (1)
        x_img[t, 18] = exposed_city

        # bias (1)
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
# RoPE blocks
# -----------------------------

def _rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    assert head_dim % 2 == 0, "RoPE needs even head_dim"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", pos, inv_freq)  # (T, half)
    return freqs.cos(), freqs.sin()

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B,nH,T,D), cos/sin: (T, D//2)
    B, nH, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos.view(1, 1, T, half)
    sin = sin.view(1, 1, T, half)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # (B,nH,T,hd)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = _rope_cache(T, self.head_dim, device=x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        scale = (self.head_dim ** -0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,nH,T,T)
        att = att.softmax(dim=-1)
        att = self.drop(att)

        y = torch.matmul(att, v)  # (B,nH,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, d)
        return self.out(y)

class RoPETransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.ln1(x))
        x = x + self.drop1(h)
        h = self.ff(self.ln2(x))
        x = x + self.drop2(h)
        return x


# -----------------------------
# Per-frame encoder
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
        h = self.conv(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        return self.fc(h)


# -----------------------------
# Policy: RoPE Transformer + Factorized action heads (dir/mode conditioned on src)
# -----------------------------

class SeqPPOPolicyRoPEFactorized(nn.Module):
    """
    Factorized policy:
      - source logits over (H*W)
      - dir logits over 4
      - mode logits over 2

    NEW: dir/mode logits are conditioned on selected source via an embedding.
    """
    def __init__(
        self,
        action_size: int,
        H: int = 25,
        W: int = 25,
        T: int = 100,
        img_channels: int = 20,
        meta_dim: int = 10,
        frame_dim: int = 256,
        d_model: int = 320,
        nhead: int = 8,
        nlayers: int = 4,
        dropout: float = 0.1,
        src_emb_dim: int = 64,
    ):
        super().__init__()
        self.action_size = int(action_size)
        self.H = int(H)
        self.W = int(W)
        self.T = int(T)
        self.src_size = self.H * self.W  # 625

        self.frame_encoder = FrameEncoder(img_channels, frame_dim)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.in_proj = nn.Linear(frame_dim + 64, d_model)
        self.blocks = nn.ModuleList([RoPETransformerBlock(d_model, nhead, dropout=dropout) for _ in range(nlayers)])

        # global head feature
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
        )

        # source head (global)
        self.pi_src = nn.Linear(256, self.src_size)

        # src embedding for conditioning dir/mode
        self.src_emb = nn.Embedding(self.src_size, src_emb_dim)

        # conditional context MLP
        self.dm_ctx = nn.Sequential(
            nn.Linear(256 + src_emb_dim, 256),
            nn.ReLU(),
        )

        # dir/mode heads (conditioned)
        self.pi_dir = nn.Linear(256, 4)
        self.pi_mode = nn.Linear(256, 2)

        self.v = nn.Linear(256, 1)

    def _global_features(self, x_img: torch.Tensor, x_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          g: (B,256) global features for policy/value
          value: (B,)
        """
        B, T, C, H, W = x_img.shape
        assert T == self.T, f"Expected T={self.T}, got T={T}"

        x2 = x_img.reshape(B * T, C, H, W)
        f = self.frame_encoder(x2).reshape(B, T, -1)  # (B,T,frame_dim)
        m = self.meta_mlp(x_meta.reshape(B * T, -1)).reshape(B, T, -1)  # (B,T,64)
        z = self.in_proj(torch.cat([f, m], dim=-1))  # (B,T,d_model)

        for blk in self.blocks:
            z = blk(z)

        last = z[:, -1, :]          # (B,d_model)
        g = self.head(last)         # (B,256)
        value = self.v(g).squeeze(-1)
        return g, value

    def _split_masks(self, legal_action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        legal_action_mask: (B, A) bool, A = H*W*8
        Returns:
          src_mask: (B, H*W) bool
          dm_mask:  (B, H*W, 8) bool  where 8 = 4*2 (dir,mode flattened)
        """
        B, A = legal_action_mask.shape
        dm_mask = legal_action_mask.view(B, self.src_size, 8)
        src_mask = dm_mask.any(dim=-1)
        return src_mask, dm_mask

    def _masked_categorical(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        masked_logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=masked_logits)

    def _dir_mode_logits(self, g: torch.Tensor, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g: (B,256), src: (B,)
        returns logits_dir (B,4), logits_mode (B,2) conditioned on src.
        """
        e = self.src_emb(src)  # (B,src_emb_dim)
        ctx = self.dm_ctx(torch.cat([g, e], dim=-1))  # (B,256)
        return self.pi_dir(ctx), self.pi_mode(ctx)

    @torch.no_grad()
    def act(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_action_mask: torch.Tensor):
        """
        Returns:
          action_flat: (B,)
          logp: (B,)
          value: (B,)
          entropy: (B,)
        """
        g, value = self._global_features(x_img, x_meta)
        logits_src = self.pi_src(g)

        src_mask, dm_mask = self._split_masks(legal_action_mask)

        # sample source
        dist_src = self._masked_categorical(logits_src, src_mask)
        src = dist_src.sample()
        logp_src = dist_src.log_prob(src)
        ent_src = dist_src.entropy()

        # conditioned dir/mode logits
        logits_dir, logits_mode = self._dir_mode_logits(g, src)

        # masks conditioned on src
        B = src.shape[0]
        row = dm_mask[torch.arange(B, device=src.device), src]  # (B,8)
        dm = row.view(B, 4, 2)
        dir_mask = dm.any(dim=-1)   # (B,4)
        mode_mask = dm.any(dim=-2)  # (B,2)

        dist_dir = self._masked_categorical(logits_dir, dir_mask)
        dist_mode = self._masked_categorical(logits_mode, mode_mask)

        d = dist_dir.sample()
        m = dist_mode.sample()
        logp = logp_src + dist_dir.log_prob(d) + dist_mode.log_prob(m)
        ent = ent_src + dist_dir.entropy() + dist_mode.entropy()

        action_flat = ((src * 4 + d) * 2 + m).long()
        return action_flat, logp, value, ent

    def evaluate_actions(
        self,
        x_img: torch.Tensor,
        x_meta: torch.Tensor,
        legal_action_mask: torch.Tensor,
        actions_flat: torch.Tensor,
    ):
        """
        Evaluate logp, entropy, value for given actions.
        """
        g, value = self._global_features(x_img, x_meta)
        logits_src = self.pi_src(g)
        src_mask, dm_mask = self._split_masks(legal_action_mask)

        actions_flat = actions_flat.long()
        src = actions_flat // 8
        dm = actions_flat % 8
        d = dm // 2
        m = dm % 2

        dist_src = self._masked_categorical(logits_src, src_mask)

        # conditioned logits for the given src
        logits_dir, logits_mode = self._dir_mode_logits(g, src)

        # masks conditioned on src
        B = actions_flat.shape[0]
        row = dm_mask[torch.arange(B, device=actions_flat.device), src]  # (B,8)
        dm2 = row.view(B, 4, 2)
        dir_mask = dm2.any(dim=-1)
        mode_mask = dm2.any(dim=-2)

        dist_dir = self._masked_categorical(logits_dir, dir_mask)
        dist_mode = self._masked_categorical(logits_mode, mode_mask)

        logp = dist_src.log_prob(src) + dist_dir.log_prob(d) + dist_mode.log_prob(m)
        ent = dist_src.entropy() + dist_dir.entropy() + dist_mode.entropy()
        return logp, ent, value
