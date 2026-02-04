from __future__ import annotations

from typing import Any, Dict

from .policy_st_rope2d import SeqPPOPolicySTRoPE2D


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
    st_rope2d: Dict[str, Any] | None = None,
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
    n = (name or "rope_factorized").lower()
    rope = rope or {}
    spatial = spatial or {}
    st_rope2d = st_rope2d or {}

    if n in ("st_rope2d", "st-rope2d", "spatiotemporal_rope2d", "spatiotemporal-rope2d"):
        return SeqPPOPolicySTRoPE2D(
            H=H, W=W,
            img_channels=img_channels,
            meta_dim=meta_dim,
            **st_rope2d,
        )

    raise ValueError(
        f"Unknown model.name '{name}'. Supported: rope_factorized, spatial_factorized, st_rope2d"
    )
