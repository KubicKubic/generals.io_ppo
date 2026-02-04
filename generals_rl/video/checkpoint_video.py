# checkpoint_video.py
from __future__ import annotations

import os
import numpy as np
import torch

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
from ..data import ObsHistory, encode_obs_sequence
from ..models import SeqPPOPolicyRoPEFactorized


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

# ---------- rollout for video ----------
@torch.no_grad()
def run_one_game_and_record(
    ckpt_path: str,
    out_mp4: str,
    T: int = 100,
    max_halfturns: int = 800,
    fps: int = 5,
    seed: int = 0,
    cell: int = 20,
    draw_text: bool = True,
    random_opp_prob: float = 1.0,  # compat; unused
    pov_player: int = 0,            # which player is "self"(blue) in rendering
):
    """
    Load checkpoint -> run 1 episode (P0 uses checkpoint policy) vs random bot -> save mp4.
    """
    if imageio is None:
        raise RuntimeError("imageio not installed. Please: pip install imageio imageio-ffmpeg")

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    # env
    env = GeneralsEnvWithMemory(seed=seed, max_halfturns=max_halfturns)
    (obs0, obs1), _ = env.reset()

    # policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SeqPPOPolicyRoPEFactorized(action_size=env.action_size, H=env.H, W=env.W, T=T).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    h0 = ObsHistory(max_len=T); h0.reset(obs0)
    h1 = ObsHistory(max_len=T); h1.reset(obs1)

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)

    frames_per_halfturn = 1

    def append_frame_repeated():
        frame = render_full_frame(env, cell=cell, draw_text=draw_text, pov_player=pov_player)
        for _ in range(frames_per_halfturn):
            writer.append_data(frame)

    try:
        append_frame_repeated()

        while True:
            # P0 action (checkpoint)
            seq0 = h0.get_padded_seq()
            x_img0_seq, x_meta0_seq = encode_obs_sequence(seq0, player_id=0)
            x_img0 = x_img0_seq.unsqueeze(0).to(device)
            x_meta0 = x_meta0_seq.unsqueeze(0).to(device)

            mask0_np = env.legal_action_mask(Owner.P0)
            mask0 = torch.from_numpy(mask0_np).unsqueeze(0).to(device).to(torch.bool)

            a0, _, _, _ = policy.act(x_img0, x_meta0, mask0)

            # P1 action: random legal
            mask1_np = env.legal_action_mask(Owner.P1)
            legal1 = np.flatnonzero(mask1_np)
            a1 = int(np.random.choice(legal1)) if len(legal1) > 0 else 0

            res = env.step(int(a0.item()), int(a1))
            obs0, obs1 = res.obs
            h0.push(obs0); h1.push(obs1)

            append_frame_repeated()

            if res.terminated or res.truncated:
                break
    finally:
        writer.close()


def make_video_for_checkpoint(
    ckpt_path: str,
    videos_dir: str,
    T: int = 100,
    fps: int = 5,
    seed: int = 0,
    max_halfturns: int = 800,
    cell: int = 20,
    draw_text: bool = True,
    pov_player: int = 0,
):
    """
    Convenience: ckpt_xxxxxx.pt -> videos/ckpt_xxxxxx.mp4
    """
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_mp4 = os.path.join(videos_dir, f"{base}.mp4")
    run_one_game_and_record(
        ckpt_path=ckpt_path,
        out_mp4=out_mp4,
        T=T,
        fps=fps,
        seed=seed,
        max_halfturns=max_halfturns,
        cell=cell,
        draw_text=draw_text,
        pov_player=pov_player,
    )
    return out_mp4
