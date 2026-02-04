from __future__ import annotations

from typing import Any, Dict

from .policy_rope_factorized import SeqPPOPolicyRoPEFactorized
from .policy_spatial_factorized import SeqPPOPolicySpatialFactorized


def make_policy(
    name: str,
    *,
    action_size: int,
    H: int,
    W: int,
    T: int,
    img_channels: int = 20,
    meta_dim: int = 10,
    rope: Dict[str, Any] | None = None,
    spatial: Dict[str, Any] | None = None,
):
    """Factory for selecting policy architecture by name.

    name:
      - rope_factorized (default): SeqPPOPolicyRoPEFactorized
      - spatial_factorized: SeqPPOPolicySpatialFactorized (matrix->matrix src map)

    Important knobs:
      - Modify env->tensor channels/meta in `generals_rl/data/encoding.py`
      - For RoPE model: pass keys like d_model/nhead/nlayers/dropout/frame_dim/src_emb_dim
      - For Spatial model: pass keys like d/meta_proj
    """
    n = (name or "rope_factorized").lower()
    rope = rope or {}
    spatial = spatial or {}

    if n in ("rope", "rope_factorized", "rope-factorized"):
        return SeqPPOPolicyRoPEFactorized(
            action_size=action_size,
            H=H, W=W, T=T,
            img_channels=img_channels,
            meta_dim=meta_dim,
            **rope,
        )

    if n in ("spatial", "spatial_factorized", "spatial-factorized"):
        return SeqPPOPolicySpatialFactorized(
            action_size=action_size,
            H=H, W=W, T=T,
            img_channels=img_channels,
            meta_dim=meta_dim,
            **spatial,
        )

    raise ValueError(f"Unknown model.name '{name}'. Supported: rope_factorized, spatial_factorized")
