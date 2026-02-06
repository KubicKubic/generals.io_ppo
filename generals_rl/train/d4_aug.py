# generals_rl/train/d4_aug.py
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch


# ---- action encoding (MUST match env/generals_env.py) ----
# a = (((r * W + c) * 4 + d) * 2 + mode)
def _encode_action(r: int, c: int, d: int, mode: int, W: int) -> int:
    return (((r * W + c) * 4 + d) * 2 + mode)


def _decode_action(a: int, W: int) -> Tuple[int, int, int, int]:
    mode = a % 2
    a //= 2
    d = a % 4
    a //= 4
    idx = a
    r = idx // W
    c = idx % W
    return r, c, d, mode


# =========================
# D4 (rot/flip) 部分：你已有的话可保留原实现
# =========================
def _apply_flip_rc(r: int, c: int, n: int) -> Tuple[int, int]:
    return r, (n - 1 - c)


def _apply_rot90_ccw_rc(r: int, c: int, n: int) -> Tuple[int, int]:
    return (n - 1 - c), r


def _transform_rc(r: int, c: int, n: int, k: int, flip: bool) -> Tuple[int, int]:
    if flip:
        r, c = _apply_flip_rc(r, c, n)
    k = k % 4
    for _ in range(k):
        r, c = _apply_rot90_ccw_rc(r, c, n)
    return r, c


def _flip_dir(d: int) -> int:
    return (-d) % 4


def _rot_ccw_dir(d: int, k: int) -> int:
    return (d - (k % 4)) % 4


def _transform_dir(d: int, k: int, flip: bool) -> int:
    if flip:
        d = _flip_dir(d)
    d = _rot_ccw_dir(d, k)
    return d


def _apply_d4_tensor(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=(-1,))  # horizontal flip on W
    if (k % 4) != 0:
        x = torch.rot90(x, k % 4, dims=(-2, -1))  # CCW
    return x


@lru_cache(maxsize=64)
def _action_perm_cpu(n: int, aug_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    D4:
      aug_id 0..3  : rot k
      aug_id 4..7  : flip + rot k
    Returns:
      perm[a] = a'      old->new
      inv[a'] = a       for mask: new_mask = old_mask[..., inv]
    """
    k = aug_id % 4
    flip = (aug_id // 4) == 1

    action_size = n * n * 4 * 2
    perm = torch.empty(action_size, dtype=torch.long)
    inv = torch.empty(action_size, dtype=torch.long)

    for a in range(action_size):
        r, c, d, mode = _decode_action(a, n)
        r2, c2 = _transform_rc(r, c, n, k=k, flip=flip)
        d2 = _transform_dir(d, k=k, flip=flip)
        a2 = _encode_action(r2, c2, d2, mode, n)
        perm[a] = a2
        inv[a2] = a

    return perm, inv


def d4_augment_minibatch(
    x_img: torch.Tensor,
    maskA: torch.Tensor,
    a: torch.Tensor,
    aug_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H = int(x_img.shape[-2])
    W = int(x_img.shape[-1])
    assert H == W
    n = H

    k = aug_id % 4
    flip = (aug_id // 4) == 1
    x_img2 = _apply_d4_tensor(x_img, k=k, flip=flip)

    perm_cpu, inv_cpu = _action_perm_cpu(n, int(aug_id))
    perm = perm_cpu.to(device=a.device, non_blocking=True)
    inv = inv_cpu.to(device=a.device, non_blocking=True)

    a2 = perm[a]
    maskA2 = maskA.index_select(dim=-1, index=inv)
    return x_img2, maskA2, a2


# =========================
# NEW: 平移（translation）增强
# 约束：移出 25x25 的部分必须是山
# -> 我们只做 (dy>=0, dx>=0) 的平移，并且只对 padding 足够的样本应用
# =========================
@lru_cache(maxsize=128)
def _action_translate_maps_cpu(n: int, dy: int, dx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      map_old_to_new: (A,) long, -1 for out-of-bounds
      valid_old: (V,) long
      new_idx: (V,) long
    So that:
      new_mask[:, new_idx] = old_mask[:, valid_old]
      new_a = map_old_to_new[old_a]
    """
    assert n >= 3
    dy = int(dy)
    dx = int(dx)

    action_size = n * n * 4 * 2
    m = torch.full((action_size,), -1, dtype=torch.long)
    valid_old = []
    new_idx = []

    for a in range(action_size):
        r, c, d, mode = _decode_action(a, n)
        r2 = r + dy
        c2 = c + dx
        if 0 <= r2 < n and 0 <= c2 < n:
            a2 = _encode_action(r2, c2, d, mode, n)
            m[a] = a2
            valid_old.append(a)
            new_idx.append(a2)

    return m, torch.tensor(valid_old, dtype=torch.long), torch.tensor(new_idx, dtype=torch.long)


def _translate_x_img_fill_mountain(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    Shift x on (H,W) with mountain-padding.
    x: (B,T,C,H,W)  (训练里就是这个)
    Mountain fill pattern (C=20):
      - tile one-hot: mountain channel=1 -> 1
      - mem one-hot: mem==0 channel=10 -> 1
      - bias channel=19 -> 1
      others -> 0
    """
    assert x.ndim == 5, "expect (B,T,C,H,W)"
    B, T, C, H, W = x.shape
    dy = int(dy)
    dx = int(dx)

    y = x.new_zeros(x.shape)
    # mountain fill
    if C > 1:
        y[:, :, 1, :, :] = 1.0
    if C > 10:
        y[:, :, 10, :, :] = 1.0
    if C > 19:
        y[:, :, 19, :, :] = 1.0

    # overlap copy
    if dy >= 0:
        ys0, yd0, hh = 0, dy, H - dy
    else:
        ys0, yd0, hh = -dy, 0, H + dy

    if dx >= 0:
        xs0, xd0, ww = 0, dx, W - dx
    else:
        xs0, xd0, ww = -dx, 0, W + dx

    if hh > 0 and ww > 0:
        y[:, :, :, yd0:yd0 + hh, xd0:xd0 + ww] = x[:, :, :, ys0:ys0 + hh, xs0:xs0 + ww]

    return y


def translate_augment_minibatch_mountain(
    x_img: torch.Tensor,   # (B,T,C,H,W)
    x_meta: torch.Tensor,  # (B,T,M)
    maskA: torch.Tensor,   # (B,A)
    a: torch.Tensor,       # (B,)
    dy: int,
    dx: int,
    real_h_meta_idx: int = 6,
    real_w_meta_idx: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply translation (dy,dx) but ONLY on samples whose real_shape padding is enough:
      real_H + dy <= H and real_W + dx <= W
    so that cropped-out part is guaranteed to be mountains (padding region).
    """
    dy = int(dy)
    dx = int(dx)
    if dy == 0 and dx == 0:
        return x_img, maskA, a

    H = int(x_img.shape[-2])
    W = int(x_img.shape[-1])
    assert H == W
    n = H

    # eligibility from meta (real_shape normalized by 25 in encoding.py)
    real_h = torch.round(x_meta[:, -1, real_h_meta_idx] * float(H)).long().clamp(0, H)
    real_w = torch.round(x_meta[:, -1, real_w_meta_idx] * float(W)).long().clamp(0, W)
    eligible = (real_h + dy <= H) & (real_w + dx <= W)

    if not bool(eligible.any()):
        return x_img, maskA, a

    # precompute action/mask mapping
    map_cpu, valid_old_cpu, new_idx_cpu = _action_translate_maps_cpu(n, dy, dx)
    map_t = map_cpu.to(device=a.device, non_blocking=True)
    valid_old = valid_old_cpu.to(device=a.device, non_blocking=True)
    new_idx = new_idx_cpu.to(device=a.device, non_blocking=True)

    # translate obs (all samples) then select eligible
    x_shift = _translate_x_img_fill_mountain(x_img, dy, dx)

    # translate mask (all samples)
    new_mask = maskA.new_zeros(maskA.shape)
    new_mask[:, new_idx] = maskA[:, valid_old]

    # translate action (all samples)
    a_shift = map_t[a]

    # select per-sample
    sel = eligible.view(-1, 1, 1, 1, 1)
    x_out = torch.where(sel, x_shift, x_img)
    mask_out = torch.where(eligible.view(-1, 1), new_mask, maskA)
    a_out = torch.where(eligible, a_shift, a)

    return x_out, mask_out, a_out


def d4_translate_augment_minibatch(
    x_img: torch.Tensor,
    x_meta: torch.Tensor,
    maskA: torch.Tensor,
    a: torch.Tensor,
    d4_id: int,
    dy: int,
    dx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Order matters:
      1) translate (down/right) with mountain padding, only on eligible samples
      2) apply D4 (rot/flip)
    This guarantees "cropped-out part are mountains" is preserved under D4.
    """
    x_img, maskA, a = translate_augment_minibatch_mountain(x_img, x_meta, maskA, a, dy=dy, dx=dx)
    x_img, maskA, a = d4_augment_minibatch(x_img, maskA, a, aug_id=d4_id)
    return x_img, maskA, a
