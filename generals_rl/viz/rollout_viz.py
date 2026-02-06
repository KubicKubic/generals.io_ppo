from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Optional, TextIO, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

from ..env.generals_env import Owner, TileType
from ..env.generals_env_memory import GeneralsEnvWithMemory


# ----------------------------
# Viz state
# ----------------------------
@dataclass
class VizState:
    do_viz: bool
    out_dir: str
    frames_per_update: int   # NOW: global cap for the whole mp4
    stride: int
    saved: int               # number of frames actually written (or planned)
    writer: Any
    trace_f: Optional[TextIO]
    cell: int
    draw_text: bool
    pov_player: int

    # top-k action viz (optional)
    topk_actions: int
    draw_arrows: bool
    print_topk: bool

    # concat buffer: frame_buf[env_id] = [frame0, frame1, ...]
    frame_buf: List[List[np.ndarray]] = field(default_factory=list)


# ---------- rendering helpers ----------
def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _mix(a, b, w=0.5):
    return tuple(int(_clamp(a[i] * (1 - w) + b[i] * w, 0, 255)) for i in range(3))


# ---------- palette ----------
C_BG = (16, 16, 20)
C_GRID = (35, 35, 45)

C_NEUTRAL = (92, 92, 105)

C_SELF = (70, 120, 210)
C_ENEMY = (210, 90, 90)

C_MOUNTAIN = (30, 30, 35)
C_CITY_TINT = (150, 120, 70)
C_GENERAL_TINT = (240, 210, 80)


# ---------- robust enum helpers ----------
def _enum_payload(e):
    outs = [e]
    v = getattr(e, "value", None)
    if v is not None:
        outs.append(v)
    try:
        outs.append(int(e))
    except Exception:
        pass
    if v is not None:
        try:
            outs.append(int(v))
        except Exception:
            pass
    return outs


def _to_int_enumish(x, fallback: int = 0) -> int:
    for a in _enum_payload(x):
        try:
            return int(a)
        except Exception:
            pass
    return int(fallback)


def render_full_frame(
    env_mem: GeneralsEnvWithMemory,
    cell: int = 20,
    draw_text: bool = True,
    pov_player: int = 0,
    hud_height: int = 14,
    draw_hud: bool = True,
) -> np.ndarray:
    env = env_mem.env
    H, W = env.H, env.W

    O_NEU = _to_int_enumish(Owner.NEUTRAL)
    O_P0  = _to_int_enumish(Owner.P0)
    O_P1  = _to_int_enumish(Owner.P1)

    T_MOUNTAIN = _to_int_enumish(TileType.MOUNTAIN)
    T_CITY     = _to_int_enumish(TileType.CITY)
    T_GENERAL  = _to_int_enumish(TileType.GENERAL)

    pov_owner = O_P0 if pov_player == 0 else O_P1
    opp_owner = O_P1 if pov_player == 0 else O_P0

    def owner_base_color(o_int: int):
        if o_int == O_NEU:
            return C_NEUTRAL
        if o_int == pov_owner:
            return C_SELF
        if o_int == opp_owner:
            return C_ENEMY
        return C_NEUTRAL

    pad_top = hud_height if (draw_text and draw_hud) else 0

    img = np.zeros((pad_top + H * cell, W * cell, 3), dtype=np.uint8)
    img[:] = C_BG

    for r in range(H):
        for c in range(W):
            t = _to_int_enumish(env.tile_type[r, c])
            o = _to_int_enumish(env.owner[r, c])

            if t == T_MOUNTAIN:
                col = C_MOUNTAIN
            else:
                base = owner_base_color(o)
                if t == T_CITY:
                    col = _mix(base, C_CITY_TINT, 0.65)
                elif t == T_GENERAL:
                    col = _mix(base, C_GENERAL_TINT, 0.72)
                else:
                    col = base

            y0 = pad_top + r * cell
            x0 = c * cell
            img[y0:y0 + cell, x0:x0 + cell] = col

            img[y0:y0 + 1, x0:x0 + cell] = C_GRID
            img[y0:y0 + cell, x0:x0 + 1] = C_GRID

    img[pad_top + H * cell - 1: pad_top + H * cell, :, :] = C_GRID
    img[pad_top: pad_top + H * cell, W * cell - 1: W * cell, :] = C_GRID

    if draw_text and Image is not None and ImageDraw is not None and ImageFont is not None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        if font is not None:
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)

            for r in range(H):
                for c in range(W):
                    t = _to_int_enumish(env.tile_type[r, c])
                    if t == T_MOUNTAIN:
                        continue

                    a = int(env.army[r, c])
                    if a <= 0:
                        continue

                    o = _to_int_enumish(env.owner[r, c])
                    y0 = pad_top + r * cell
                    x0 = c * cell

                    if o == pov_owner:
                        tc = (235, 245, 255)
                    elif o == opp_owner:
                        tc = (255, 235, 235)
                    else:
                        tc = (220, 220, 230)

                    draw.text((x0 + 2, y0 + 1), str(a), fill=tc, font=font)

            if draw_hud and pad_top > 0:
                turn = env.half_t // 2
                half_in_turn = env.half_t % 2

                flat_owner = env.owner.reshape(-1).tolist()
                flat_army = env.army.reshape(-1).tolist()
                army0 = army1 = land0 = land1 = 0
                for oo, aa in zip(flat_owner, flat_army):
                    oi = _to_int_enumish(oo)
                    if oi == O_P0:
                        land0 += 1
                        army0 += int(aa)
                    elif oi == O_P1:
                        land1 += 1
                        army1 += int(aa)

                hud = (
                    f"half_t={env.half_t}  turn={turn}  half={half_in_turn} | "
                    f"P0 army/land={army0}/{land0}  P1 army/land={army1}/{land1}"
                )
                draw.rectangle((0, 0, min(W * cell, 980), pad_top), fill=(0, 0, 0))
                draw.text((3, 0), hud, fill=(255, 255, 255), font=font)

            img = np.asarray(pil_img)

    return img


def viz_begin_update(
    *,
    do_viz: bool,
    out_dir: str,
    frames_per_update: int,
    rollout_len: int,
    save_mp4: bool,
    mp4_fps: int,
    save_trace_jsonl: bool,
    cell: int,
    draw_text: bool,
    pov_player: int,
    upd: int,
    topk_actions: int = 10,
    draw_arrows: bool = True,
    print_topk: bool = False,
    tag: str = "",
) -> VizState:
    if not do_viz:
        return VizState(
            False,
            out_dir,
            int(frames_per_update),
            10**9,
            0,
            None,
            None,
            cell,
            draw_text,
            pov_player,
            topk_actions,
            draw_arrows,
            print_topk,
            frame_buf=[],
        )

    if imageio is None:
        raise RuntimeError("Visualization requires imageio. pip install imageio imageio-ffmpeg")

    os.makedirs(out_dir, exist_ok=True)
    stride = 1

    suffix = f"_{tag}" if tag else ""
    writer = None
    if save_mp4:
        mp4_path = os.path.join(out_dir, f"upd_{upd:06d}{suffix}.mp4")
        writer = imageio.get_writer(
            mp4_path,
            fps=int(mp4_fps),
            codec="libx264",
            quality=8,
            ffmpeg_log_level="error",
            macro_block_size=None,
        )

    trace_f = None
    if save_trace_jsonl:
        trace_path = os.path.join(out_dir, f"upd_{upd:06d}{suffix}.jsonl")
        trace_f = open(trace_path, "w", encoding="utf-8")

    return VizState(
        True,
        out_dir,
        int(frames_per_update),  # global cap
        stride,
        0,
        writer,
        trace_f,
        cell,
        draw_text,
        pov_player,
        topk_actions,
        draw_arrows,
        print_topk,
        frame_buf=[],
    )


def viz_end_update(vs: VizState, upd: int, tag: str = ""):
    if not vs.do_viz:
        return

    suffix = f"_{tag}" if tag else ""

    written = 0
    cap = int(vs.frames_per_update)

    # IMPORTANT: write at most `cap` frames TOTAL, in env_id order.
    if vs.writer is not None and vs.frame_buf:
        for env_id in range(len(vs.frame_buf)):
            if written >= cap:
                break
            for fr in vs.frame_buf[env_id]:
                if written >= cap:
                    break
                vs.writer.append_data(fr)
                written += 1

    vs.saved = written

    if vs.writer is not None:
        vs.writer.close()
        print(f"[viz] saved mp4: {os.path.join(vs.out_dir, f'upd_{upd:06d}{suffix}.mp4')}")
    if vs.trace_f is not None:
        vs.trace_f.close()
        print(f"[viz] saved trace: {os.path.join(vs.out_dir, f'upd_{upd:06d}{suffix}.jsonl')}")
    print(f"[viz] saved {vs.saved} frames (cap={cap}) to {vs.out_dir} for upd={upd}{suffix}")


# ----------------------------
# Optional: arrows/top-k
# ----------------------------
def _decode_action_flat(a: int) -> Tuple[int, int, int]:
    src = a // 8
    dm = a % 8
    d = dm // 2
    m = dm % 2
    return src, d, m


def _dir_to_delta(d: int) -> Tuple[int, int]:
    if d == 0:
        return -1, 0
    if d == 1:
        return 0, 1
    if d == 2:
        return 1, 0
    return 0, -1


def _draw_arrows_on_frame(
    frame: np.ndarray,
    H: int,
    W: int,
    cell: int,
    actions: List[int],
    probs: List[float],
    hud_height_guess: int = 14,
) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame

    base = Image.fromarray(frame)
    draw_base = ImageDraw.Draw(base)

    pad_top = hud_height_guess if (frame.shape[0] >= H * cell + hud_height_guess) else 0

    def rank_color(i: int, n: int) -> tuple[int, int, int]:
        t = 0.0 if n <= 1 else i / (n - 1)
        dark = (0, 140, 0)
        light = (180, 255, 180)
        r = int(dark[0] * (1 - t) + light[0] * t)
        g = int(dark[1] * (1 - t) + light[1] * t)
        b = int(dark[2] * (1 - t) + light[2] * t)
        return (r, g, b)

    head_len = 6
    head_w = 4
    shrink = 0.70

    for i, (a, p) in enumerate(zip(actions, probs)):
        src, d, m = _decode_action_flat(int(a))
        r = src // W
        c = src % W
        dr, dc = _dir_to_delta(int(d))
        r2 = max(0, min(H - 1, r + dr))
        c2 = max(0, min(W - 1, c + dc))

        x1 = c * cell + cell // 2
        y1 = pad_top + r * cell + cell // 2
        x2 = c2 * cell + cell // 2
        y2 = pad_top + r2 * cell + cell // 2

        x2s = int(x1 + (x2 - x1) * shrink)
        y2s = int(y1 + (y2 - y1) * shrink)

        col = rank_color(i, len(actions))
        width = 3 if i == 0 else (2 if i < 3 else 1)

        draw_base.line((x1, y1, x2s, y2s), fill=col, width=width)

        hx = x2s - x1
        hy = y2s - y1
        L = (hx * hx + hy * hy) ** 0.5 + 1e-6
        ux, uy = hx / L, hy / L
        px, py = -uy, ux

        ax = x2s - int(ux * head_len)
        ay = y2s - int(uy * head_len)
        bx = ax + int(px * head_w)
        by = ay + int(py * head_w)
        cx = ax - int(px * head_w)
        cy = ay - int(py * head_w)

        draw_base.line((x2s, y2s, bx, by), fill=col, width=width)
        draw_base.line((x2s, y2s, cx, cy), fill=col, width=width)

    return np.asarray(base)


@torch.no_grad()
def _topk_legal_actions(
    policy,
    x_img: torch.Tensor,
    x_meta: torch.Tensor,
    legal_action_mask: torch.Tensor,
    k: int = 10,
):
    assert x_img.shape[0] == 1
    legal_flat = legal_action_mask[0].to(torch.bool)

    if hasattr(policy, "logits_and_value"):
        logits_flat, _v = policy.logits_and_value(x_img, x_meta)
        logits_flat = logits_flat.view(-1)

        masked_logits = logits_flat.masked_fill(~legal_flat, -1e9)
        p_flat = F.softmax(masked_logits, dim=-1)

        kk = min(int(k), int(legal_flat.sum().item()))
        top_p, top_a = torch.topk(p_flat, k=kk, largest=True, sorted=True)
        return top_a.tolist(), top_p.tolist()

    raise RuntimeError("Policy does not support logits_and_value for topk visualization.")


def make_viz_frame(
    vs: VizState,
    env: GeneralsEnvWithMemory,
    policy=None,
    x_img: Optional[torch.Tensor] = None,
    x_meta: Optional[torch.Tensor] = None,
    legal_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    frame = render_full_frame(env, cell=vs.cell, draw_text=vs.draw_text, pov_player=vs.pov_player)

    if vs.topk_actions > 0 and policy is not None and x_img is not None and x_meta is not None and legal_mask is not None:
        try:
            top_actions, top_probs = _topk_legal_actions(policy, x_img, x_meta, legal_mask, k=int(vs.topk_actions))
            if vs.draw_arrows:
                frame = _draw_arrows_on_frame(
                    frame,
                    H=env.env.H,
                    W=env.env.W,
                    cell=vs.cell,
                    actions=top_actions,
                    probs=top_probs,
                    hud_height_guess=14,
                )
        except Exception as e:
            print(f"[viz] topk/arrow failed: {e}")

    return frame


# ----------------------------
# Compatibility: keep old single-env function
# ----------------------------
def maybe_visualize_rollout_step(
    vs: VizState,
    env: GeneralsEnvWithMemory,
    upd: int,
    step: int,
    a0: int,
    a1: int,
    r: float,
    v: float,
    logp: float,
    done: bool,
    policy=None,
    x_img: Optional[torch.Tensor] = None,
    x_meta: Optional[torch.Tensor] = None,
    legal_mask: Optional[torch.Tensor] = None,
):
    if not vs.do_viz:
        return

    if vs.trace_f is not None:
        vs.trace_f.write(json.dumps({
            "upd": int(upd),
            "step": int(step),
            "env_id": 0,
            "half_t": int(env.env.half_t),
            "a0": int(a0),
            "a1": int(a1),
            "r": float(r),
            "v": float(v),
            "logp": float(logp),
            "done": bool(done),
        }, ensure_ascii=False) + "\n")

    if (step % vs.stride != 0):
        return
    if vs.saved >= int(vs.frames_per_update):
        return

    fr = make_viz_frame(vs, env, policy=policy, x_img=x_img, x_meta=x_meta, legal_mask=legal_mask)
    if vs.writer is not None:
        vs.writer.append_data(fr)
        vs.saved += 1


# ----------------------------
# Concat mode collector:
# buffer per env, then `viz_end_update` writes in env order with GLOBAL cap.
# ----------------------------
def maybe_visualize_rollout_step_concat(
    vs: VizState,
    envs: List[GeneralsEnvWithMemory],
    upd: int,
    step: int,
    a0_list: List[int],
    a1_list: List[int],
    r_list: List[float],
    v_list: List[float],
    logp_list: List[float],
    done_list: List[bool],
    policy=None,
    x_img_batch: Optional[torch.Tensor] = None,      # (E,T,C,H,W)
    x_meta_batch: Optional[torch.Tensor] = None,     # (E,T,M)
    legal_mask_batch: Optional[torch.Tensor] = None, # (E,A)
):
    if not vs.do_viz:
        return

    E = len(envs)
    if not vs.frame_buf:
        vs.frame_buf = [[] for _ in range(E)]

    if vs.trace_f is not None:
        for i in range(E):
            vs.trace_f.write(json.dumps({
                "upd": int(upd),
                "step": int(step),
                "env_id": int(i),
                "half_t": int(envs[i].env.half_t),
                "a0": int(a0_list[i]),
                "a1": int(a1_list[i]),
                "r": float(r_list[i]),
                "v": float(v_list[i]),
                "logp": float(logp_list[i]),
                "done": bool(done_list[i]),
            }, ensure_ascii=False) + "\n")

    if (step % vs.stride) != 0:
        return

    # NOTE: we don't try to enforce the GLOBAL cap here (would require prioritization policy during collection).
    # We only do a reasonable per-env bound to avoid runaway memory,
    # and the final GLOBAL cap + env-order truncation is enforced in viz_end_update.
    per_env_soft_cap = int(vs.frames_per_update)

    for i in range(E):
        if len(vs.frame_buf[i]) >= per_env_soft_cap:
            continue

        xi = xm = lm = None
        if policy is not None and x_img_batch is not None and x_meta_batch is not None and legal_mask_batch is not None:
            xi = x_img_batch[i:i+1]
            xm = x_meta_batch[i:i+1]
            lm = legal_mask_batch[i:i+1]

        fr = make_viz_frame(vs, envs[i], policy=policy, x_img=xi, x_meta=xm, legal_mask=lm)
        vs.frame_buf[i].append(fr)

    # saved here is "collected", actual "written" is decided at viz_end_update with global cap
    vs.saved = min(int(vs.frames_per_update), sum(len(x) for x in vs.frame_buf))
