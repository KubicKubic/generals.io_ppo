from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Optional, TextIO, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..env.generals_env_memory import GeneralsEnvWithMemory
from ..video.checkpoint_video import render_full_frame  # 你已修好的 renderer（带 top padding 的版本）

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None


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
    stride = max(1, int(rollout_len) // max(1, int(frames_per_update)))

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
    x_img: torch.Tensor,        # (1,T,C,H,W)
    x_meta: torch.Tensor,       # (1,T,M)
    legal_action_mask: torch.Tensor,  # (1,A) bool
    k: int = 10,
):
    """
    Return top-k legal actions and probs under factorized model:
      p(a)=p(src)*p(dir|src)*p(mode|src)
    Supports:
      - RoPE factorized model (pi_src + dir_mode_logits)
      - Spatial factorized model (src_head + dm_mlp + dir_head/mode_head)
    """
    assert x_img.shape[0] == 1, "viz assumes batch=1 for topk"
    B = 1

    A = legal_action_mask.shape[1]
    src_size = A // 8
    H = policy.H if hasattr(policy, "H") else None
    W = policy.W if hasattr(policy, "W") else None
    if H is None or W is None:
        # fallback from env dimensions encoded in action_size
        W = int((src_size) ** 0.5)
        H = src_size // W

    # dm_mask: (1,src,8), src_mask: (1,src)
    dm_mask = legal_action_mask.view(B, src_size, 8)
    src_mask = dm_mask.any(dim=-1)  # (1,src)

    # ---- get src logits + per-src dir/mode logits (vectorized over src) ----
    if hasattr(policy, "pi_src") and hasattr(policy, "global_features"):
        # RoPE factorized
        g, _v = policy.global_features(x_img, x_meta)         # (1,256)
        logits_src = policy.pi_src(g)                         # (1,src)

        # vectorize dir/mode for all src
        src_all = torch.arange(src_size, device=logits_src.device, dtype=torch.long)  # (src,)
        g_rep = g.repeat(src_size, 1)                          # (src,256)
        logits_dir_all, logits_mode_all = policy.dir_mode_logits(g_rep, src_all)     # (src,4),(src,2)

    elif hasattr(policy, "src_head") and hasattr(policy, "spatial_features"):
        # Spatial factorized
        Fmap, _v = policy.spatial_features(x_img, x_meta)      # (1,D,H,W)
        logits_src_map = policy.src_head(Fmap).squeeze(1)      # (1,H,W)
        logits_src = logits_src_map.view(1, -1)                # (1,src)

        # all src cell embeddings: (src,D)
        D = Fmap.shape[1]
        flat = Fmap.view(1, D, H, W).reshape(D, H * W).transpose(0, 1).contiguous()  # (src,D)
        h = policy.dm_mlp(flat)                                # (src,D)
        logits_dir_all = policy.dir_head(h)                    # (src,4)
        logits_mode_all = policy.mode_head(h)                  # (src,2)
    else:
        raise RuntimeError("Unknown policy type for topk action viz.")

    # ---- masked softmax for src ----
    logits_src_masked = logits_src.masked_fill(~src_mask, -1e9)  # (1,src)
    p_src = F.softmax(logits_src_masked, dim=-1).view(-1)        # (src,)

    # ---- per-src masked softmax for dir/mode ----
    dm = dm_mask[0].view(src_size, 4, 2)                         # (src,4,2)
    dir_mask = dm.any(dim=-1)                                    # (src,4)
    mode_mask = dm.any(dim=-2)                                   # (src,2)

    # logits_dir_all: (src,4), logits_mode_all: (src,2)
    ld = logits_dir_all.masked_fill(~dir_mask, -1e9)
    lm = logits_mode_all.masked_fill(~mode_mask, -1e9)
    p_dir = F.softmax(ld, dim=-1)                                # (src,4)
    p_mode = F.softmax(lm, dim=-1)                               # (src,2)

    # ---- build probs for all flat actions (src,4,2) ----
    p_pair = p_dir.unsqueeze(-1) * p_mode.unsqueeze(-2)          # (src,4,2)
    # zero out illegal dm
    legal_dm = dm_mask[0].view(src_size, 4, 2).to(torch.float32)
    p_pair = p_pair * legal_dm
    # multiply by p_src
    p_pair = p_pair * p_src.view(src_size, 1, 1)

    p_flat = p_pair.reshape(src_size * 8)                        # (A,)
    legal_flat = legal_action_mask[0].to(torch.bool)             # (A,)
    p_flat = p_flat.masked_fill(~legal_flat, 0.0)

    # if all zeros (shouldn't), fallback
    if float(p_flat.sum().item()) <= 0:
        legal_idx = torch.nonzero(legal_flat).view(-1)
        top_idx = legal_idx[:k]
        top_p = torch.ones_like(top_idx, dtype=torch.float32) / max(1, len(top_idx))
        return top_idx.tolist(), top_p.tolist()

    # topk
    k = min(int(k), int(legal_flat.sum().item()))
    top_p, top_a = torch.topk(p_flat, k=k, largest=True, sorted=True)
    return top_a.tolist(), top_p.tolist()


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
