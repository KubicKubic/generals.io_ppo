from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class SpatialBackbone(nn.Module):
    """Matrix->Matrix backbone: (B,C,H,W) -> (B,D,H,W)."""
    def __init__(self, in_ch: int, d: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d, d, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SeqPPOPolicySpatialFactorized(nn.Module):
    """Spatial policy (matrix->matrix) + factorized (src,dir,mode) heads.

    - Spatial features F: (B,D,H,W) from last frame (T must match, but can be 1).
    - src logits are a map: (B,H,W) -> flatten to (B,H*W)
    - dir/mode logits are predicted from gathered feature at chosen src.
    - value is global pooled scalar.

    Flat action encoding matches the existing env:
        action_flat = ((src * 4 + dir) * 2 + mode)  where src in [0, H*W).
    """
    def __init__(
        self,
        action_size: int,
        H: int = 25,
        W: int = 25,
        T: int = 1,
        img_channels: int = 20,
        meta_dim: int = 10,
        d: int = 128,
        meta_proj: int = 16,
    ):
        super().__init__()
        self.action_size = int(action_size)
        self.H = int(H)
        self.W = int(W)
        self.T = int(T)
        self.src_size = self.H * self.W

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_proj),
            nn.ReLU(),
        )
        self.backbone = SpatialBackbone(img_channels + meta_proj, d)

        self.src_head = nn.Conv2d(d, 1, kernel_size=1)

        self.dm_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.dir_head = nn.Linear(d, 4)
        self.mode_head = nn.Linear(d, 2)

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(d, 1),
        )

    # ---- key extension point ----
    def spatial_features(self, x_img: torch.Tensor, x_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_img: (B,T,C,H,W), x_meta: (B,T,M)
        Returns: F (B,D,H,W), value (B,)
        """
        B, T, C, H, W = x_img.shape
        assert H == self.H and W == self.W, f"Expected H,W=({self.H},{self.W}), got ({H},{W})"
        assert T == self.T, f"Expected T={self.T}, got T={T}"

        x_last = x_img[:, -1]          # (B,C,H,W)
        m_last = x_meta[:, -1]         # (B,M)

        m = self.meta_mlp(m_last)      # (B,meta_proj)
        m = m[:, :, None, None].expand(B, m.shape[1], H, W)
        x = torch.cat([x_last, m], dim=1)  # (B,C+meta_proj,H,W)

        F = self.backbone(x)               # (B,D,H,W)
        v = self.value_head(F).squeeze(-1) # (B,)
        return F, v

    def _split_masks(self, legal_action_mask: torch.Tensor):
        B, _ = legal_action_mask.shape
        dm_mask = legal_action_mask.view(B, self.src_size, 8)
        src_mask = dm_mask.any(dim=-1)
        return src_mask, dm_mask

    def _masked_categorical(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        masked_logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=masked_logits)

    def _gather_cell(self, F: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Gather per-cell embedding from F at flattened src indices."""
        B, D, H, W = F.shape
        r = (src // W).long()
        c = (src % W).long()
        return F[torch.arange(B, device=F.device), :, r, c]  # (B,D)

    @torch.no_grad()
    def act(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_action_mask: torch.Tensor):
        F, value = self.spatial_features(x_img, x_meta)

        logits_src_map = self.src_head(F).squeeze(1)           # (B,H,W)
        logits_src = logits_src_map.view(F.shape[0], -1)       # (B,H*W)

        src_mask, dm_mask = self._split_masks(legal_action_mask)

        dist_src = self._masked_categorical(logits_src, src_mask)
        src = dist_src.sample()
        logp_src = dist_src.log_prob(src)
        ent_src = dist_src.entropy()

        cell = self._gather_cell(F, src)                       # (B,D)
        h = self.dm_mlp(cell)
        logits_dir = self.dir_head(h)                          # (B,4)
        logits_mode = self.mode_head(h)                        # (B,2)

        B = src.shape[0]
        row = dm_mask[torch.arange(B, device=src.device), src]  # (B,8)
        dm = row.view(B, 4, 2)
        dir_mask = dm.any(dim=-1)    # (B,4)
        mode_mask = dm.any(dim=-2)   # (B,2)

        dist_dir = self._masked_categorical(logits_dir, dir_mask)
        dist_mode = self._masked_categorical(logits_mode, mode_mask)

        d = dist_dir.sample()
        m = dist_mode.sample()

        logp = logp_src + dist_dir.log_prob(d) + dist_mode.log_prob(m)
        ent = ent_src + dist_dir.entropy() + dist_mode.entropy()

        action_flat = ((src * 4 + d) * 2 + m).long()
        return action_flat, logp, value, ent

    def evaluate_actions(self, x_img: torch.Tensor, x_meta: torch.Tensor, legal_action_mask: torch.Tensor, actions_flat: torch.Tensor):
        F, value = self.spatial_features(x_img, x_meta)

        logits_src_map = self.src_head(F).squeeze(1)
        logits_src = logits_src_map.view(F.shape[0], -1)

        src_mask, dm_mask = self._split_masks(legal_action_mask)

        actions_flat = actions_flat.long()
        src = actions_flat // 8
        dm = actions_flat % 8
        d = dm // 2
        m = dm % 2

        dist_src = self._masked_categorical(logits_src, src_mask)

        cell = self._gather_cell(F, src)
        h = self.dm_mlp(cell)
        logits_dir = self.dir_head(h)
        logits_mode = self.mode_head(h)

        B = actions_flat.shape[0]
        row = dm_mask[torch.arange(B, device=actions_flat.device), src]
        dm2 = row.view(B, 4, 2)
        dir_mask = dm2.any(dim=-1)
        mode_mask = dm2.any(dim=-2)

        dist_dir = self._masked_categorical(logits_dir, dir_mask)
        dist_mode = self._masked_categorical(logits_mode, mode_mask)

        logp = dist_src.log_prob(src) + dist_dir.log_prob(d) + dist_mode.log_prob(m)
        ent = dist_src.entropy() + dist_dir.entropy() + dist_mode.entropy()
        return logp, ent, value
