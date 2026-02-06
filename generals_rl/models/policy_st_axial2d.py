from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    masked_logits = logits.masked_fill(~mask, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)


# ---------------------------
# RoPE for time transformer
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
# Axial Attention (2D)
# ---------------------------
class AxialMHA(nn.Module):
    """
    Multi-head self-attn on a 1D sequence: (B, N, D) -> (B, N, D)
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.nhead, self.d_head).transpose(1, 2)  # (B,nh,N,dh)
        k = k.view(B, N, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.nhead, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,nh,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # (B,nh,N,dh)
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


class AxialBlock2D(nn.Module):
    """
    x: (B, D, H, W)
    Do:
      - row attention: for each row, attend over W
      - col attention: for each col, attend over H
    After one block, information can propagate globally (row+col).
    """
    def __init__(self, d_model: int, nhead: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_row = nn.LayerNorm(d_model)
        self.row_attn = AxialMHA(d_model, nhead, dropout=dropout)

        self.ln_col = nn.LayerNorm(d_model)
        self.col_attn = AxialMHA(d_model, nhead, dropout=dropout)

        hidden = d_model * mlp_ratio
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape

        # ---- row attn ----
        # (B, D, H, W) -> (B*H, W, D)
        xr = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, D)
        xr = xr + self.row_attn(self.ln_row(xr))
        # back: (B,H,W,D)->(B,D,H,W)
        x = xr.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # ---- col attn ----
        # (B, D, H, W) -> (B*W, H, D)
        xc = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, D)
        xc = xc + self.col_attn(self.ln_col(xc))
        x = xc.view(B, W, H, D).permute(0, 3, 2, 1).contiguous()

        # ---- per-position MLP ----
        # (B,D,H,W) -> (B*H*W, D)
        xf = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, D)
        xf = xf + self.mlp(self.ln_mlp(xf))
        x = xf.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        return x


class AxialEncoder2D(nn.Module):
    """
    (B, Cin, H, W) -> (B, D, H, W)
    Use 1x1 projection + N axial blocks.
    """
    def __init__(self, cin: int, d_model: int, nhead: int, nlayers: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Conv2d(cin, d_model, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList([
            AxialBlock2D(d_model=d_model, nhead=nhead, mlp_ratio=4, dropout=dropout)
            for _ in range(nlayers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class AxialHead2D(nn.Module):
    """
    (B, Cin, H, W) -> (B, Cout, H, W)
    """
    def __init__(self, cin: int, cout: int, d_model: int, nhead: int, nlayers: int, dropout: float = 0.0):
        super().__init__()
        self.proj_in = nn.Conv2d(cin, d_model, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList([
            AxialBlock2D(d_model=d_model, nhead=nhead, mlp_ratio=4, dropout=dropout)
            for _ in range(nlayers)
        ])
        self.proj_out = nn.Conv2d(d_model, cout, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj_out(x)


# ---------------------------
# Main policy (Axial spatial, RoPE temporal)
# ---------------------------
class SeqPPOPolicySTAxial2D(nn.Module):
    """
    Drop-in replacement for st_rope2d style policy:
      - frame encoder: axial transformer on 2D, outputs (B,T,D,H,W)
      - time transformer: per-cell RoPE transformer on sequences (T,D)
      - head: axial transformer on 2D producing logits (B,8,H,W)
      - value: from g_last (B,D,H,W) global avg
    """
    def __init__(
        self,
        H: int = 25,
        W: int = 25,
        img_channels: int = 20,
        meta_dim: int = 10,
        meta_proj: int = 16,

        d_model: int = 64,          # temporal model dim (and g_last channels)
        nhead_time: int = 4,
        nlayers_time: int = 2,
        dropout: float = 0.0,

        # axial spatial encoder/head
        nhead_axial: int = 8,
        nlayers_axial_enc: int = 2,
        nlayers_axial_head: int = 2,
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

        # spatial frame encoder: (L + meta_proj) -> D
        # we choose axial internal dim then project to D
        self.frame_encoder_axial = AxialEncoder2D(
            cin=self.L + self.meta_proj,
            d_model=d_model,
            nhead=nhead_axial,
            nlayers=nlayers_axial_enc,
            dropout=dropout,
        )

        # time transformer on (T, D)
        self.time_blocks = nn.ModuleList([
            RoPETransformerBlock(d_model=self.D, nhead=nhead_time, mlp_ratio=4, dropout=dropout)
            for _ in range(nlayers_time)
        ])

        # spatial head: (L + D + meta_proj) -> 8
        self.head = AxialHead2D(
            cin=self.L + self.D + self.meta_proj,
            cout=8,
            d_model=d_model,
            nhead=nhead_axial,
            nlayers=nlayers_axial_head,
            dropout=dropout,
        )

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.D, 1),
        )

    def _meta_map(self, meta_proj: torch.Tensor, H: int, W: int) -> torch.Tensor:
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
        z = self.frame_encoder_axial(x_flat)           # (B*T, axial_d, H, W)
        f = z.view(B, T, self.D, H, W)            # (B,T,D,H,W)

        meta_last_map = self._meta_map(meta_all[:, -1], H, W)
        return f, meta_all, meta_last_map

    def _time_transform_last(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: (B,T,D,H,W)
        returns g_last: (B,D,H,W)
        """
        B, T, D, H, W = f.shape
        seq = f.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, D)

        x = seq
        for blk in self.time_blocks:
            x = blk(x)

        g_last = x[:, -1, :].view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
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
