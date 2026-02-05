from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Optional, TextIO, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from ..env.generals_env import Owner, TileType
from ..env.generals_env_memory import GeneralsEnvWithMemory

# ----------------------------
# Viz state
# ----------------------------
@dataclass
class VizState:
    do_viz: bool
    out_dir: str
    frames_per_update: int
    stride: int
    saved: int
    writer: Any
    trace_f: Optional[TextIO]
    cell: int
    draw_text: bool
    pov_player: int

    # NEW: top-k action viz
    topk_actions: int
    draw_arrows: bool
    print_topk: bool

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
    """Return candidate representations for enum-like object e."""
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


def _eq_enumish(x, const) -> bool:
    """Robust equality: expand payload for BOTH sides and compare."""
    xs = _enum_payload(x)
    cs = _enum_payload(const)

    for a in xs:
        for b in cs:
            try:
                if a == b:
                    return True
            except Exception:
                pass

    for a in xs:
        for b in cs:
            try:
                if int(a) == int(b):
                    return True
            except Exception:
                pass

    return False


def _to_int_enumish(x, fallback: int = 0) -> int:
    """Best-effort convert enum-ish / numpy scalar / int to int."""
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
    hud_height: int = 14,     # top padding height
    draw_hud: bool = True,    # set False to disable HUD entirely
) -> np.ndarray:
    """
    Render full true state.

    Fixes:
      - "black background + white numbers" (numpy/PIL desync) by drawing tiles/grid in numpy first,
        then converting to PIL to draw text/HUD.
      - HUD covering the board by adding TOP PADDING (hud_height) and shifting the board down.
    """
    env = env_mem.env
    H, W = env.H, env.W

    # ---------- robust enum/int helpers ----------
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

    # robust constants
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

    # ---------- top padding ----------
    pad_top = hud_height if (draw_text and draw_hud) else 0

    # ---------- Phase 1: draw tiles + grid into numpy ----------
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

            # grid
            img[y0:y0 + 1, x0:x0 + cell] = C_GRID
            img[y0:y0 + cell, x0:x0 + 1] = C_GRID

    # outer border for the board area only
    img[pad_top + H * cell - 1: pad_top + H * cell, :, :] = C_GRID
    img[pad_top: pad_top + H * cell, W * cell - 1: W * cell, :] = C_GRID

    # ---------- Phase 2: draw numbers + HUD on PIL ----------
    if draw_text and Image is not None and ImageDraw is not None and ImageFont is not None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        if font is not None:
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)

            # numbers
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

            # HUD in top padding area
            if draw_hud and pad_top > 0:
                turn = env.half_t // 2
                half_in_turn = env.half_t % 2

                # robust scan for army/land
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
                # background bar (only in padding, not on board)
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
    # NEW:
    topk_actions: int = 10,
    draw_arrows: bool = True,
    print_topk: bool = False,
) -> VizState:
    if not do_viz:
        return VizState(False, out_dir, frames_per_update, 10**9, 0, None, None, cell, draw_text, pov_player, topk_actions, draw_arrows, print_topk)

    if imageio is None:
        raise RuntimeError("Visualization requires imageio. pip install imageio imageio-ffmpeg")

    os.makedirs(out_dir, exist_ok=True)
    stride = 1

    writer = None
    if save_mp4:
        mp4_path = os.path.join(out_dir, f"upd_{upd:06d}.mp4")
        writer = imageio.get_writer(mp4_path, fps=int(mp4_fps), codec="libx264", quality=8)

    trace_f = None
    if save_trace_jsonl:
        trace_path = os.path.join(out_dir, f"upd_{upd:06d}.jsonl")
        trace_f = open(trace_path, "w", encoding="utf-8")

    return VizState(True, out_dir, frames_per_update, stride, 0, writer, trace_f, cell, draw_text, pov_player, topk_actions, draw_arrows, print_topk)


def viz_end_update(vs: VizState, upd: int):
    if not vs.do_viz:
        return
    if vs.writer is not None:
        vs.writer.close()
        print(f"[viz] saved mp4: {os.path.join(vs.out_dir, f'upd_{upd:06d}.mp4')}")
    if vs.trace_f is not None:
        vs.trace_f.close()
        print(f"[viz] saved trace: {os.path.join(vs.out_dir, f'upd_{upd:06d}.jsonl')}")
    print(f"[viz] saved {vs.saved} frames to {vs.out_dir} for upd={upd}")


# ----------------------------
# Helpers: decode/arrow draw
# ----------------------------
def _decode_action_flat(a: int) -> Tuple[int, int, int]:
    src = a // 8
    dm = a % 8
    d = dm // 2
    m = dm % 2
    return src, d, m


def _dir_to_delta(d: int) -> Tuple[int, int]:
    # 约定：0=up,1=right,2=down,3=left（若你 env 不同可在这里改）
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
    """
    Draw arrows for top actions using green shades (dark->light),
    and draw legend OUTSIDE the board (right margin).
    """
    if Image is None or ImageDraw is None:
        return frame

    base = Image.fromarray(frame)
    draw_base = ImageDraw.Draw(base)

    pad_top = hud_height_guess if (frame.shape[0] >= H * cell + hud_height_guess) else 0

    # ---------- arrow style ----------
    # smaller arrows than before
    def rank_color(i: int, n: int) -> tuple[int, int, int]:
        # i: 0..n-1, 0 is best (darkest)
        if n <= 1:
            t = 0.0
        else:
            t = i / (n - 1)  # 0..1
        # dark green -> light green
        dark = (0, 140, 0)
        light = (180, 255, 180)
        r = int(dark[0] * (1 - t) + light[0] * t)
        g = int(dark[1] * (1 - t) + light[1] * t)
        b = int(dark[2] * (1 - t) + light[2] * t)
        return (r, g, b)

    # arrow head parameters (smaller)
    head_len = 6
    head_w = 4
    shrink = 0.70

    # ---------- draw arrows on board ----------
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

        # shorten body
        x2s = int(x1 + (x2 - x1) * shrink)
        y2s = int(y1 + (y2 - y1) * shrink)

        col = rank_color(i, len(actions))
        width = 3 if i == 0 else (2 if i < 3 else 1)

        draw_base.line((x1, y1, x2s, y2s), fill=col, width=width)

        # arrow head
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

    # ---------- legend outside (right margin, auto columns) ----------
    out_h = base.size[1]

    # layout params
    title_h = 18          # title text block height
    line_h = 14           # one entry line height
    sw = 10               # color swatch size
    left_pad = 8
    top_pad = 8
    col_gap = 14          # gap between legend columns

    # how many entries fit per column?
    usable_h = out_h - (top_pad + title_h + 6)
    lines_per_col = max(1, usable_h // line_h)

    # a safe column width (no font metrics; keep generous)
    col_w = max(240, cell * 9)

    n = len(actions)
    ncols = (n + lines_per_col - 1) // lines_per_col

    legend_w = left_pad * 2 + ncols * col_w + max(0, (ncols - 1) * col_gap)
    out_w = base.size[0] + legend_w

    # create new canvas and paste base image
    out = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    out.paste(base, (0, 0))
    draw = ImageDraw.Draw(out)

    # title (only once, first column)
    x_start = base.size[0] + left_pad
    y_start = top_pad
    draw.text((x_start, y_start), f"Top actions (k={n})", fill=(255, 255, 255))

    # draw entries column by column
    y0 = y_start + title_h
    for i, (a, p) in enumerate(zip(actions, probs)):
        col_idx = i // lines_per_col
        row_idx = i % lines_per_col

        x = x_start + col_idx * (col_w + col_gap)
        y = y0 + row_idx * line_h

        col = rank_color(i, n)

        # swatch
        draw.rectangle((x, y + 3, x + sw, y + 3 + sw), fill=col)

        # decode
        src, d, m = _decode_action_flat(int(a))
        rr = src // W
        cc = src % W

        txt = f"#{i+1:02d} p={p:.3f} src=({rr},{cc}) d={d} m={m}"
        draw.text((x + sw + 6, y), txt, fill=(230, 230, 230))

    return np.asarray(out)

# ----------------------------
# Top-k probability computation
# ----------------------------
@torch.no_grad()
def _topk_legal_actions(
    policy,
    x_img: torch.Tensor,              # (1,T,C,H,W)
    x_meta: torch.Tensor,             # (1,T,M)
    legal_action_mask: torch.Tensor,  # (1,A) bool
    k: int = 10,
):
    """
    Return top-k legal actions and probs.

    Supports:
      1) Joint-logits policies (e.g. SeqPPOPolicySTRoPE2D): logits_flat = (1, A)
         -> p = softmax(masked_logits)
      2) Factorized policies (old): p(a)=p(src)*p(dir|src)*p(mode|src)
    """
    assert x_img.shape[0] == 1, "viz assumes batch=1 for topk"
    B = 1

    A = legal_action_mask.shape[1]
    legal_flat = legal_action_mask[0].to(torch.bool)  # (A,)

    # -------------------------
    # (1) NEW: Joint logits policy (policy_st_rope2d.py)
    # -------------------------
    if hasattr(policy, "logits_and_value"):
        # logits_flat: (1, A)
        logits_flat, _v = policy.logits_and_value(x_img, x_meta)
        logits_flat = logits_flat.view(-1)  # (A,)

        # masked softmax over legal actions
        masked_logits = logits_flat.masked_fill(~legal_flat, -1e9)
        p_flat = F.softmax(masked_logits, dim=-1)  # (A,)

        # if all zeros (shouldn't), fallback
        if float(p_flat.sum().item()) <= 0:
            legal_idx = torch.nonzero(legal_flat).view(-1)
            kk = min(int(k), int(legal_idx.numel()))
            top_a = legal_idx[:kk]
            top_p = torch.ones_like(top_a, dtype=torch.float32) / max(1, kk)
            return top_a.tolist(), top_p.tolist()

        kk = min(int(k), int(legal_flat.sum().item()))
        top_p, top_a = torch.topk(p_flat, k=kk, largest=True, sorted=True)
        return top_a.tolist(), top_p.tolist()

    assert False

# ----------------------------
# Main viz hook
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
    # NEW: for top-k prediction visualization (P0)
    policy=None,
    x_img: Optional[torch.Tensor] = None,    # (1,T,C,H,W)
    x_meta: Optional[torch.Tensor] = None,   # (1,T,M)
    legal_mask: Optional[torch.Tensor] = None,  # (1,A) bool
):
    if not vs.do_viz:
        return

    # trace each step
    if vs.trace_f is not None:
        vs.trace_f.write(json.dumps({
            "upd": int(upd),
            "step": int(step),
            "half_t": int(env.env.half_t),
            "a0": int(a0),
            "a1": int(a1),
            "r": float(r),
            "v": float(v),
            "logp": float(logp),
            "done": bool(done),
        }, ensure_ascii=False) + "\n")

    # only save frames sparsely
    if (step % vs.stride != 0) or (vs.saved >= vs.frames_per_update):
        return

    if imageio is None:
        raise RuntimeError("Visualization requires imageio. pip install imageio imageio-ffmpeg")

    # Render base frame
    frame = render_full_frame(env, cell=vs.cell, draw_text=vs.draw_text, pov_player=vs.pov_player)

    # Top-k legal actions (optional)
    if vs.topk_actions > 0 and policy is not None and x_img is not None and x_meta is not None and legal_mask is not None:
        try:
            top_actions, top_probs = _topk_legal_actions(policy, x_img, x_meta, legal_mask, k=int(vs.topk_actions))

            if vs.print_topk:
                # print as: rank prob src r,c dir mode flat
                H = env.env.H
                W = env.env.W
                print(f"[viz][upd={upd} step={step} half_t={int(env.env.half_t)}] top{len(top_actions)} legal actions:")
                for i, (aa, pp) in enumerate(zip(top_actions, top_probs), 1):
                    src, d, m = _decode_action_flat(int(aa))
                    rr = src // W
                    cc = src % W
                    print(f"  #{i:02d} p={pp:.5f}  flat={int(aa):4d}  src={src:3d}({rr:02d},{cc:02d})  dir={d}  mode={m}")

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

    # Save png
    # png_path = os.path.join(vs.out_dir, f"upd_{upd:06d}_step_{step:04d}_half_t_{int(env.env.half_t):05d}.png")
    # imageio.imwrite(png_path, frame)

    # Append to mp4
    if vs.writer is not None:
        vs.writer.append_data(frame)

    vs.saved += 1
