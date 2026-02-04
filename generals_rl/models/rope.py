from __future__ import annotations
import torch
import torch.nn as nn

def rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    assert head_dim % 2 == 0, "RoPE needs even head_dim"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", pos, inv_freq)
    return freqs.cos(), freqs.sin()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = rope_cache(T, self.head_dim, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scale = (self.head_dim ** -0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        att = att.softmax(dim=-1)
        att = self.drop(att)

        y = torch.matmul(att, v)
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
