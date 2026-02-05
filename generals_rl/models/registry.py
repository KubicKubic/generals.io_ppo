from __future__ import annotations

from typing import Any, Dict

from .policy_st_rope2d import SeqPPOPolicySTRoPE2D
from .policy_st_axial2d import SeqPPOPolicySTAxial2D


def make_policy(
    name: str,
    *,
    action_size: int,
    H: int,
    W: int,
    T: int,
    img_channels: int = 20,
    meta_dim: int = 10,
    st_rope2d: Dict[str, Any] | None = None,
    st_axial2d: Dict[str, Any] | None = None,
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
    print(name)
    n = (name or "st_axial2d").lower()
    st_rope2d = st_rope2d or {}

    if n in ("st_rope2d", "st-rope2d", "spatiotemporal_rope2d", "spatiotemporal-rope2d"):
        return SeqPPOPolicySTRoPE2D(
            H=H, W=W,
            img_channels=img_channels,
            meta_dim=meta_dim,
            **st_rope2d,
        )

    if n in ("st_axial2d", "st-axial2d", "spatiotemporal_axial2d", "spatiotemporal-axial2d"):
        return SeqPPOPolicySTAxial2D(
            H=H, W=W,
            img_channels=img_channels,
            meta_dim=meta_dim,
            **st_axial2d,
        )

    raise ValueError(
        f"Unknown model.name '{name}'. Supported: rope_factorized, spatial_factorized, st_rope2d"
    )
