from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from .frame_encoder import FrameEncoder
from .rope import RoPETransformerBlock

class SeqPPOPolicyRoPEFactorized(nn.Module):
    """RoPE Transformer policy with factorized (src,dir,mode) heads."""
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
        self.src_size = self.H * self.W

        self.frame_encoder = FrameEncoder(img_channels, frame_dim)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.in_proj = nn.Linear(frame_dim + 64, d_model)
        self.blocks = nn.ModuleList([RoPETransformerBlock(d_model, nhead, dropout=dropout) for _ in range(nlayers)])

        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
        )

        self.pi_src = nn.Linear(256, self.src_size)

        self.src_emb = nn.Embedding(self.src_size, src_emb_dim)
        self.dm_ctx = nn.Sequential(
            nn.Linear(256 + src_emb_dim, 256),
            nn.ReLU(),
        )
        self.pi_dir = nn.Linear(256, 4)
        self.pi_mode = nn.Linear(256, 2)

        self.v = nn.Linear(256, 1)

    # ---- key extension points ----
    def global_features(self, x_img: torch.Tensor, x_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override/extend if you want a different backbone."""
        B, T, C, H, W = x_img.shape
        assert T == self.T, f"Expected T={self.T}, got T={T}"

        x2 = x_img.reshape(B * T, C, H, W)
        f = self.frame_encoder(x2).reshape(B, T, -1)
        m = self.meta_mlp(x_meta.reshape(B * T, -1)).reshape(B, T, -1)
        z = self.in_proj(torch.cat([f, m], dim=-1))

        for blk in self.blocks:
            z = blk(z)

        last = z[:, -1, :]
        g = self.head(last)
        value = self.v(g).squeeze(-1)
        return g, value

    def split_masks(self, legal_action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _ = legal_action_mask.shape
        dm_mask = legal_action_mask.view(B, self.src_size, 8)
        src_mask = dm_mask.any(dim=-1)
        return src_mask, dm_mask

    def masked_categorical(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        masked_logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=masked_logits)

    def dir_mode_logits(self, g: torch.Tensor, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.src_emb(src)
        ctx = self.dm_ctx(torch.cat([g, e], dim=-1))
        return self.pi_dir(ctx), self.pi_mode(ctx)

    @torch.no_grad()
    def act(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_action_mask: torch.Tensor):
        g, value = self.global_features(x_img, x_meta)
        logits_src = self.pi_src(g)

        src_mask, dm_mask = self.split_masks(legal_action_mask)

        dist_src = self.masked_categorical(logits_src, src_mask)
        src = dist_src.sample()
        logp_src = dist_src.log_prob(src)
        ent_src = dist_src.entropy()

        logits_dir, logits_mode = self.dir_mode_logits(g, src)

        B = src.shape[0]
        row = dm_mask[torch.arange(B, device=src.device), src]
        dm = row.view(B, 4, 2)
        dir_mask = dm.any(dim=-1)
        mode_mask = dm.any(dim=-2)

        dist_dir = self.masked_categorical(logits_dir, dir_mask)
        dist_mode = self.masked_categorical(logits_mode, mode_mask)

        d = dist_dir.sample()
        m = dist_mode.sample()
        logp = logp_src + dist_dir.log_prob(d) + dist_mode.log_prob(m)
        ent = ent_src + dist_dir.entropy() + dist_mode.entropy()

        action_flat = ((src * 4 + d) * 2 + m).long()
        return action_flat, logp, value, ent

    def evaluate_actions(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_action_mask: torch.Tensor, actions_flat: torch.Tensor):
        g, value = self.global_features(x_img, x_meta)
        logits_src = self.pi_src(g)
        src_mask, dm_mask = self.split_masks(legal_action_mask)

        actions_flat = actions_flat.long()
        src = actions_flat // 8
        dm = actions_flat % 8
        d = dm // 2
        m = dm % 2

        dist_src = self.masked_categorical(logits_src, src_mask)

        logits_dir, logits_mode = self.dir_mode_logits(g, src)

        B = actions_flat.shape[0]
        row = dm_mask[torch.arange(B, device=actions_flat.device), src]
        dm2 = row.view(B, 4, 2)
        dir_mask = dm2.any(dim=-1)
        mode_mask = dm2.any(dim=-2)

        dist_dir = self.masked_categorical(logits_dir, dir_mask)
        dist_mode = self.masked_categorical(logits_mode, mode_mask)

        logp = dist_src.log_prob(src) + dist_dir.log_prob(d) + dist_mode.log_prob(m)
        ent = dist_src.entropy() + dist_dir.entropy() + dist_mode.entropy()
        return logp, ent, value
