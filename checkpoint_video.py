# checkpoint_video.py
from __future__ import annotations

import os
import numpy as np
import torch

try:
    import imageio.v2 as imageio
except Exception as e:
    imageio = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

from generals_env import Owner, TileType
from generals_env_memory import GeneralsEnvWithMemory
from memory_rl_model import ObsHistory, encode_obs_sequence, SeqPPOPolicyRoPEFactorized


# ---------- rendering helpers ----------
def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _mix(a, b, w=0.5):
    return tuple(int(_clamp(a[i] * (1 - w) + b[i] * w, 0, 255)) for i in range(3))

# basic palette
C_BG = (16, 16, 20)
C_GRID = (35, 35, 45)
C_NEUTRAL = (92, 92, 105)
C_P0 = (70, 120, 210)
C_P1 = (210, 90, 90)
C_MOUNTAIN = (30, 30, 35)
C_CITY = (150, 120, 70)
C_GENERAL = (240, 210, 80)

def render_full_frame(env_mem: GeneralsEnvWithMemory, cell: int = 20, draw_text: bool = True) -> np.ndarray:
    """
    Render full true state (not fog) as RGB image.
    """
    env = env_mem.env  # underlying GeneralsEnv
    H, W = env.H, env.W
    img = np.zeros((H * cell, W * cell, 3), dtype=np.uint8)
    img[:] = C_BG

    # try font
    font = None
    if draw_text and ImageFont is not None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    use_pil = (draw_text and Image is not None and ImageDraw is not None and font is not None)
    pil_img = Image.fromarray(img) if use_pil else None
    draw = ImageDraw.Draw(pil_img) if use_pil else None

    for r in range(H):
        for c in range(W):
            t = int(env.tile_type[r, c])
            o = int(env.owner[r, c])
            a = int(env.army[r, c])

            if t == TileType.MOUNTAIN:
                col = C_MOUNTAIN
            else:
                base = C_NEUTRAL if o == Owner.NEUTRAL else (C_P0 if o == Owner.P0 else C_P1)
                if t == TileType.CITY:
                    col = _mix(base, C_CITY, 0.55)
                elif t == TileType.GENERAL:
                    col = _mix(base, C_GENERAL, 0.6)
                else:
                    col = base

            y0 = r * cell
            x0 = c * cell
            img[y0:y0+cell, x0:x0+cell] = col

            # grid
            img[y0:y0+1, x0:x0+cell] = C_GRID
            img[y0:y0+cell, x0:x0+1] = C_GRID

            if use_pil and t != TileType.MOUNTAIN and a > 0:
                # small number
                txt = str(a)
                # choose text color
                tc = (250, 250, 250) if o != Owner.NEUTRAL else (210, 210, 220)
                draw.text((x0 + 2, y0 + 1), txt, fill=tc, font=font)

    # right/bottom border
    img[-1:, :, :] = C_GRID
    img[:, -1:, :] = C_GRID

    # overlay basic HUD on top-left
    if use_pil:
        turn = env.half_t // 2
        half_in_turn = env.half_t % 2
        # scoreboard from env (true)
        o0 = (env.owner == Owner.P0)
        o1 = (env.owner == Owner.P1)
        army0 = int(env.army[o0].sum())
        army1 = int(env.army[o1].sum())
        land0 = int(o0.sum())
        land1 = int(o1.sum())
        hud = f"half_t={env.half_t}  turn={turn}  half={half_in_turn} | P0 army/land={army0}/{land0}  P1 army/land={army1}/{land1}"
        draw.rectangle((0, 0, min(W * cell, 820), 14), fill=(0, 0, 0))
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
    fps: int = 15,
    seed: int = 0,
    cell: int = 20,
    draw_text: bool = True,
    random_opp_prob: float = 1.0,  # 1.0 => pure random opponent
):
    """
    Load checkpoint -> run 1 episode (P0 uses checkpoint policy) vs random bot -> save mp4.
    """
    if imageio is None:
        raise RuntimeError("imageio not installed. Please: pip install imageio imageio-ffmpeg")

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    # Create env with same seed (video reproducible)
    env = GeneralsEnvWithMemory(seed=seed, max_halfturns=max_halfturns)
    (obs0, obs1), _ = env.reset()

    # Load policy from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SeqPPOPolicyRoPEFactorized(action_size=env.action_size, H=env.H, W=env.W, T=T).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    h0 = ObsHistory(max_len=T); h0.reset(obs0)
    h1 = ObsHistory(max_len=T); h1.reset(obs1)

    # Writer
    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)

    # initial frame
    frame = render_full_frame(env, cell=cell, draw_text=draw_text)
    writer.append_data(frame)

    while True:
        # P0 action
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
        a1 = int(np.random.choice(legal1))

        # step
        res = env.step(int(a0.item()), int(a1))
        obs0, obs1 = res.obs
        h0.push(obs0); h1.push(obs1)

        # render after this half-turn
        frame = render_full_frame(env, cell=cell, draw_text=draw_text)
        writer.append_data(frame)

        if res.terminated or res.truncated:
            break

    writer.close()


def make_video_for_checkpoint(
    ckpt_path: str,
    videos_dir: str,
    T: int = 100,
    fps: int = 15,
    seed: int = 0,
    max_halfturns: int = 800,
    cell: int = 20,
    draw_text: bool = True,
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
    )
    return out_mp4
