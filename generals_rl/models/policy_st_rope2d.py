from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    masked_logits = logits.masked_fill(~mask, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)


# ---------------------------
# RoPE
# ---------------------------
def _rope_angles(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0):
    assert dim % 2 == 0
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # (T, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, nh, T, dh)
    cos/sin: (T, dh/2)
    """
    dh = x.shape[-1]
    assert dh % 2 == 0
    half = dh // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    while cos.ndim < x1.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B,T,3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.d_head).transpose(1, 2)  # (B,nh,T,dh)
        k = k.view(B, T, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_head).transpose(1, 2)

        cos, sin = _rope_angles(T, self.d_head, device=x.device, dtype=x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,nh,T,T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # (B,nh,T,dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out(y)
        y = self.proj_drop(y)
        return y


class RoPETransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, nhead, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------
# Conv blocks
# ---------------------------
class ConvNet2D(nn.Module):
    """(B,Cin,H,W) -> (B,Cout,H,W)"""
    def __init__(self, cin: int, cout: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        c = cin
        for _ in range(max(1, depth - 1)):
            layers += [nn.Conv2d(c, hidden, 3, padding=1), nn.ReLU()]
            c = hidden
        layers += [nn.Conv2d(c, cout, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------
# Main policy
# ---------------------------
class SeqPPOPolicySTRoPE2D(nn.Module):
    """
    Per your request (revised):
      - frame encoder outputs f[t] as (B, D, H, W) directly.
      - time RoPE transformer runs on per-cell sequences (T, D).
      - g_last is (B, D, H, W).
      - head uses obs_last + g_last + meta_last_map -> (B,8,H,W) -> flatten.

    Input:
      x_img: (B,T,L,H,W)
      x_meta:(B,T,M)
    """
    def __init__(
        self,
        H: int = 25,
        W: int = 25,
        img_channels: int = 20,
        meta_dim: int = 10,
        meta_proj: int = 16,
        d_model: int = 64,          # <-- this is D
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.0,
        # conv encoder/head capacity
        enc_hidden: int = 128,
        enc_depth: int = 3,
        head_hidden: int = 128,
        head_depth: int = 3
    ):
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.L = int(img_channels)
        self.M = int(meta_dim)
        self.meta_proj = int(meta_proj)
        self.D = int(d_model)

        self.action_size = self.H * self.W * 8

        self.meta_mlp = nn.Sequential(
            nn.Linear(self.M, self.meta_proj),
            nn.ReLU(),
        )

        # per-frame encoder: (L + meta_proj) -> D
        self.frame_encoder = ConvNet2D(
            cin=self.L + self.meta_proj,
            cout=self.D,
            hidden=enc_hidden,
            depth=enc_depth,
        )

        # time transformer on (T, D)
        self.time_blocks = nn.ModuleList([
            RoPETransformerBlock(d_model=self.D, nhead=nhead, mlp_ratio=4, dropout=dropout)
            for _ in range(nlayers)
        ])

        # head: (L + D + meta_proj) -> 8
        self.head = ConvNet2D(
            cin=self.L + self.D + self.meta_proj,
            cout=8,
            hidden=head_hidden,
            depth=head_depth,
        )

        # value head from g_last
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.D, 1),
        )

    def _meta_map(self, meta_proj: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # meta_proj: (B, meta_proj)
        B = meta_proj.shape[0]
        return meta_proj[:, :, None, None].expand(B, meta_proj.shape[1], H, W)

    def _encode_frames(self, x_img: torch.Tensor, x_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
          f: (B,T,D,H,W)
          meta_all_proj: (B,T,meta_proj)
          meta_last_map: (B,meta_proj,H,W)
        """
        B, T, L, H, W = x_img.shape
        assert (H, W) == (self.H, self.W)
        assert L == self.L
        assert x_meta.shape[:2] == (B, T) and x_meta.shape[2] == self.M

        meta_all = self.meta_mlp(x_meta.reshape(B * T, self.M)).view(B, T, self.meta_proj)
        meta_maps = meta_all[:, :, :, None, None].expand(B, T, self.meta_proj, H, W)

        x_cat = torch.cat([x_img, meta_maps], dim=2)  # (B,T,L+meta_proj,H,W)

        x_flat = x_cat.reshape(B * T, L + self.meta_proj, H, W)
        f_flat = self.frame_encoder(x_flat)           # (B*T, D, H, W)
        f = f_flat.view(B, T, self.D, H, W)          # (B,T,D,H,W)

        meta_last_map = self._meta_map(meta_all[:, -1], H, W)
        return f, meta_all, meta_last_map

    def _time_transform_last(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: (B,T,D,H,W)
        returns g_last: (B,D,H,W)
        """
        B, T, D, H, W = f.shape
        # (B,T,D,H,W) -> (B,H,W,T,D) -> (B*H*W, T, D)
        seq = f.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, D)

        x = seq
        for blk in self.time_blocks:
            x = blk(x)   # (B*H*W, T, D)

        g_last = x[:, -1, :].view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)
        return g_last

    def logits_and_value(self, x_img: torch.Tensor, x_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          logits_flat: (B, H*W*8)
          value: (B,)
        """
        B, T, L, H, W = x_img.shape
        f, _meta_all, meta_last_map = self._encode_frames(x_img, x_meta)
        g_last = self._time_transform_last(f)              # (B,D,H,W)

        obs_last = x_img[:, -1]                            # (B,L,H,W)
        head_in = torch.cat([obs_last, g_last, meta_last_map], dim=1)  # (B,L+D+meta_proj,H,W)

        logits_dm = self.head(head_in)                     # (B,8,H,W)
        logits_flat = logits_dm.permute(0, 2, 3, 1).contiguous().view(B, H * W * 8)

        v = self.value_head(g_last).squeeze(-1)            # (B,)
        return logits_flat, v

    @torch.no_grad()
    def act(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_action_mask: torch.Tensor):
        logits, v = self.logits_and_value(x_img, x_meta)
        dist = masked_categorical(logits, legal_action_mask)
        a = dist.sample()
        return a.long(), dist.log_prob(a), v, dist.entropy()

    def evaluate_actions(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_action_mask: torch.Tensor, actions_flat: torch.Tensor):
        logits, v = self.logits_and_value(x_img, x_meta)
        dist = masked_categorical(logits, legal_action_mask)
        a = actions_flat.long()
        return dist.log_prob(a), dist.entropy(), v
