from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Optional, TextIO
from ..env.generals_env_memory import GeneralsEnvWithMemory
from ..video.checkpoint_video import render_full_frame

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

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

def viz_begin_update(*, do_viz: bool, out_dir: str, frames_per_update: int, rollout_len: int,
                     save_mp4: bool, mp4_fps: int, save_trace_jsonl: bool,
                     cell: int, draw_text: bool, pov_player: int, upd: int) -> VizState:
    if not do_viz:
        return VizState(False, out_dir, frames_per_update, 10**9, 0, None, None, cell, draw_text, pov_player)
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
    return VizState(True, out_dir, frames_per_update, stride, 0, writer, trace_f, cell, draw_text, pov_player)

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

def maybe_visualize_rollout_step(vs: VizState, env: GeneralsEnvWithMemory, upd: int, step: int,
                                a0: int, a1: int, r: float, v: float, logp: float, done: bool):
    if not vs.do_viz:
        return
    if vs.trace_f is not None:
        vs.trace_f.write(json.dumps({
            "upd": int(upd), "step": int(step), "half_t": int(env.env.half_t),
            "a0": int(a0), "a1": int(a1), "r": float(r), "v": float(v), "logp": float(logp), "done": bool(done),
        }, ensure_ascii=False) + "\n")
    if (step % vs.stride == 0) and (vs.saved < vs.frames_per_update):
        if imageio is None:
            raise RuntimeError("Visualization requires imageio. pip install imageio imageio-ffmpeg")
        frame = render_full_frame(env, cell=vs.cell, draw_text=vs.draw_text, pov_player=vs.pov_player)
        png_path = os.path.join(vs.out_dir, f"upd_{upd:06d}_step_{step:04d}_half_t_{int(env.env.half_t):05d}.png")
        # imageio.imwrite(png_path, frame)
        if vs.writer is not None:
            vs.writer.append_data(frame)
        vs.saved += 1
