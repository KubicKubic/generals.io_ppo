# simulator.py
# Play GeneralsEnv (half-turn step) vs a random bot with visualization.
#
# Controls:
#   - Mouse left click: select a cell
#   - W/A/S/D: move from selected cell up/left/down/right
#   - Z: make NEXT move use HALF split (one-shot), otherwise LEAVE_ONE
#   - SPACE: pass (end half-turn with no-op action)
#   - R: reset game
#   - TAB: toggle view mode (fog view for P0 / full state)
#   - ESC / window close: quit
#
# Notes:
#   - A half-turn ends only after you make a VALID move (or SPACE pass).
#   - If you try an illegal move, nothing happens and the half-turn does NOT advance.

from __future__ import annotations

import sys
import numpy as np

import pygame

from generals_rl.env.generals_env import GeneralsEnv, Owner, TileType, MoveMode, Dir, encode_action


# ---------- UI config ----------
CELL = 24
MARGIN = 2
SIDEBAR_W = 260
TOPBAR_H = 40

FPS = 60

# Colors (R,G,B)
C_BG = (18, 18, 22)
C_GRID = (40, 40, 48)

C_NEUTRAL = (90, 90, 98)
C_P0 = (70, 120, 210)
C_P1 = (210, 90, 90)

C_MOUNTAIN = (35, 35, 40)
C_CITY = (150, 120, 70)
C_GENERAL = (240, 210, 80)

C_FOG_EMPTY = (60, 60, 70)      # fog-render empty
C_FOG_MOUNTAIN = (30, 30, 35)   # fog-render mountain (also cities under fog)

C_SEL = (255, 255, 255)
C_SEL_BAD = (255, 110, 110)

# Directions for WASD (W up, A left, S down, D right)
KEY_DIR = {
    pygame.K_w: Dir.UP,
    pygame.K_d: Dir.RIGHT,
    pygame.K_s: Dir.DOWN,
    pygame.K_a: Dir.LEFT,
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Simulator:
    def __init__(self):
        self.env = GeneralsEnv()  # defaults 25x25
        (self.obs0, self.obs1), _ = self.env.reset()

        self.selected = (0, 0)
        self.next_half_mode = MoveMode.LEAVE_ONE  # one-shot, can be set to HALF by Z
        self.show_full = False  # TAB toggles

        pygame.init()
        pygame.display.set_caption("GeneralsEnv Simulator (P0 vs Random Bot)")
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_big = pygame.font.SysFont("consolas", 20, bold=True)

        self.W = self.env.W
        self.H = self.env.H

        self.grid_w_px = self.W * CELL
        self.grid_h_px = self.H * CELL

        self.win_w = self.grid_w_px + SIDEBAR_W
        self.win_h = self.grid_h_px + TOPBAR_H

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()

    # ---------- Bot ----------
    def random_bot_action(self) -> int:
        mask = self.env.legal_action_mask(Owner.P1)
        legal = np.flatnonzero(mask)
        return int(np.random.choice(legal))

    # ---------- Action helpers ----------
    def is_legal_for_p0(self, action: int) -> bool:
        mask = self.env.legal_action_mask(Owner.P0)
        return bool(mask[action]) if 0 <= action < self.env.action_size else False

    def make_action_from_selected(self, direction: Dir, mode: MoveMode) -> int:
        r, c = self.selected
        return encode_action(r, c, int(direction), int(mode), self.env.W)

    def step_halfturn(self, a0: int):
        """Advance one half-turn using player action a0 and random bot a1."""
        a1 = self.random_bot_action()
        res = self.env.step(a0, a1)
        self.obs0, self.obs1 = res.obs
        return res

    # ---------- Rendering ----------
    def cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = c * CELL
        y = TOPBAR_H + r * CELL
        return pygame.Rect(x, y, CELL, CELL)

    def get_mouse_cell(self, mx: int, my: int):
        if my < TOPBAR_H:
            return None
        c = mx // CELL
        r = (my - TOPBAR_H) // CELL
        if 0 <= r < self.H and 0 <= c < self.W:
            return (r, c)
        return None

    def draw_text(self, text, x, y, color=(230, 230, 240), big=False):
        surf = (self.font_big if big else self.font).render(text, True, color)
        self.screen.blit(surf, (x, y))

    def tile_color_full(self, r, c):
        t = int(self.env.tile_type[r, c])
        o = int(self.env.owner[r, c])
        if t == TileType.MOUNTAIN:
            return C_MOUNTAIN
        # base ownership color
        if o == Owner.P0:
            base = C_P0
        elif o == Owner.P1:
            base = C_P1
        else:
            base = C_NEUTRAL

        if t == TileType.CITY:
            # tint cities
            return tuple(clamp((base[i] + C_CITY[i]) // 2, 0, 255) for i in range(3))
        if t == TileType.GENERAL:
            # generals stand out
            return tuple(clamp((base[i] + C_GENERAL[i]) // 2, 0, 255) for i in range(3))
        return base

    def tile_color_fog_view(self, r, c):
        # use obs0 rendering: tile_type already includes fog-rendering
        t = int(self.obs0["tile_type"][r, c])
        vis = bool(self.obs0["visible"][r, c])

        if not vis:
            # fog render: mountains & cities show as mountain, others as empty
            if t == TileType.MOUNTAIN:
                return C_FOG_MOUNTAIN
            else:
                return C_FOG_EMPTY

        # visible: show by owner (true owner visible only when vis)
        o = int(self.obs0["owner"][r, c])
        if t == TileType.MOUNTAIN:
            return C_MOUNTAIN

        if o == Owner.P0:
            base = C_P0
        elif o == Owner.P1:
            base = C_P1
        elif o == Owner.NEUTRAL:
            base = C_NEUTRAL
        else:
            base = C_NEUTRAL

        if t == TileType.CITY:
            return tuple(clamp((base[i] + C_CITY[i]) // 2, 0, 255) for i in range(3))
        if t == TileType.GENERAL:
            return tuple(clamp((base[i] + C_GENERAL[i]) // 2, 0, 255) for i in range(3))
        return base

    def draw_grid(self):
        for r in range(self.H):
            for c in range(self.W):
                rect = self.cell_rect(r, c)
                color = self.tile_color_full(r, c) if self.show_full else self.tile_color_fog_view(r, c)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, C_GRID, rect, 1)

                # Numbers: in full mode show all armies; in fog mode show only visible armies
                if self.show_full:
                    a = int(self.env.army[r, c])
                    o = int(self.env.owner[r, c])
                    t = int(self.env.tile_type[r, c])
                    if t != TileType.MOUNTAIN and (a > 0):
                        txt = str(a)
                        # Make neutral armies dimmer
                        col = (250, 250, 250) if o != Owner.NEUTRAL else (210, 210, 220)
                        self.draw_text(txt, rect.x + 3, rect.y + 3, col)
                else:
                    if bool(self.obs0["visible"][r, c]):
                        a = int(self.obs0["army"][r, c])
                        t = int(self.obs0["tile_type"][r, c])
                        if t != TileType.MOUNTAIN and a >= 0:
                            self.draw_text(str(a), rect.x + 3, rect.y + 3, (250, 250, 250))

        # selection outline
        sr, sc = self.selected
        srect = self.cell_rect(sr, sc)
        pygame.draw.rect(self.screen, C_SEL, srect, 3)

    def draw_sidebar(self, last_result=None):
        x0 = self.grid_w_px + 12
        y0 = 10

        mode_txt = "HALF (split)" if self.next_half_mode == MoveMode.HALF else "LEAVE_ONE"
        view_txt = "FULL STATE" if self.show_full else "P0 FOG VIEW"

        self.draw_text(f"View: {view_txt}", x0, y0, big=True)
        y0 += 30
        self.draw_text(f"Half-turn: {self.env.half_t}", x0, y0)
        y0 += 22
        self.draw_text(f"Turn: {self.env.half_t // 2}  HalfInTurn: {self.env.half_t % 2}", x0, y0)
        y0 += 22

        if hasattr(self.env, "real_H"):
            self.draw_text(f"Real map: {self.env.real_H} x {self.env.real_W}", x0, y0)
            y0 += 22

        sr, sc = self.selected
        self.draw_text(f"Selected: ({sr},{sc})", x0, y0)
        y0 += 22

        # show selected cell info (full or fog)
        if self.show_full:
            t = TileType(int(self.env.tile_type[sr, sc])).name
            o = int(self.env.owner[sr, sc])
            a = int(self.env.army[sr, sc])
            self.draw_text(f"Tile: {t}", x0, y0); y0 += 20
            self.draw_text(f"Owner: {o}", x0, y0); y0 += 20
            self.draw_text(f"Army: {a}", x0, y0); y0 += 20
        else:
            vis = bool(self.obs0["visible"][sr, sc])
            if vis:
                t = TileType(int(self.obs0["tile_type"][sr, sc])).name
                o = int(self.obs0["owner"][sr, sc])
                a = int(self.obs0["army"][sr, sc])
                self.draw_text(f"Visible: True", x0, y0); y0 += 20
                self.draw_text(f"Tile: {t}", x0, y0); y0 += 20
                self.draw_text(f"Owner: {o}", x0, y0); y0 += 20
                self.draw_text(f"Army: {a}", x0, y0); y0 += 20
            else:
                t = TileType(int(self.obs0["tile_type"][sr, sc])).name
                self.draw_text(f"Visible: False", x0, y0); y0 += 20
                self.draw_text(f"FogTile: {t}", x0, y0); y0 += 20
                self.draw_text(f"Owner: ?", x0, y0); y0 += 20
                self.draw_text(f"Army: ?", x0, y0); y0 += 20

        y0 += 10
        self.draw_text(f"Next mode (Z): {mode_txt}", x0, y0)
        y0 += 28

        self.draw_text("Controls:", x0, y0, big=True); y0 += 26
        self.draw_text("Click: select cell", x0, y0); y0 += 18
        self.draw_text("W/A/S/D: move", x0, y0); y0 += 18
        self.draw_text("Z: next move HALF", x0, y0); y0 += 18
        self.draw_text("SPACE: pass", x0, y0); y0 += 18
        self.draw_text("TAB: toggle view", x0, y0); y0 += 18
        self.draw_text("R: reset", x0, y0); y0 += 18
        self.draw_text("ESC: quit", x0, y0); y0 += 18

        if last_result is not None:
            y0 += 12
            if last_result.terminated:
                w = last_result.info.get("winner", None)
                msg = "You win!" if w == int(Owner.P0) else ("You lose!" if w == int(Owner.P1) else "Game over.")
                self.draw_text(f"Result: {msg}", x0, y0, (255, 230, 120), big=True)
            elif last_result.truncated:
                self.draw_text("Result: Truncated", x0, y0, (255, 230, 120), big=True)

    def draw_topbar(self):
        pygame.draw.rect(self.screen, (12, 12, 15), pygame.Rect(0, 0, self.win_w, TOPBAR_H))
        self.draw_text("GeneralsEnv Simulator — P0 vs Random Bot (half-turn step)", 10, 10, (230, 230, 240), big=True)

    # ---------- Main loop ----------
    def run(self):
        last_result = None

        running = True
        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    cell = self.get_mouse_cell(*event.pos)
                    if cell is not None:
                        self.selected = cell

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break

                    if event.key == pygame.K_TAB:
                        self.show_full = not self.show_full

                    if event.key == pygame.K_r:
                        (self.obs0, self.obs1), _ = self.env.reset()
                        last_result = None
                        self.next_half_mode = MoveMode.LEAVE_ONE
                        self.selected = (0, 0)

                    if event.key == pygame.K_z:
                        # one-shot: set next move to HALF
                        self.next_half_mode = MoveMode.HALF

                    if event.key == pygame.K_SPACE:
                        # pass ends half-turn
                        a0 = 0
                        last_result = self.step_halfturn(a0)
                        # mode resets after an action ends half-turn
                        self.next_half_mode = MoveMode.LEAVE_ONE

                    # movement keys
                    if event.key in KEY_DIR:
                        direction = KEY_DIR[event.key]
                        mode = self.next_half_mode

                        # current selection is the move source
                        sr, sc = self.selected
                        dr, dc = {
                            Dir.UP: (-1, 0),
                            Dir.RIGHT: (0, 1),
                            Dir.DOWN: (1, 0),
                            Dir.LEFT: (0, -1),
                        }[direction]
                        tr, tc = sr + dr, sc + dc  # target cell

                        a0 = self.make_action_from_selected(direction, mode)

                        if self.is_legal_for_p0(a0):
                            last_result = self.step_halfturn(a0)

                            # move selection to the target cell after the half-turn resolves
                            # clamp just in case (though legality guarantees in-bounds & not mountain)
                            self.selected = (clamp(tr, 0, self.H - 1), clamp(tc, 0, self.W - 1))

                            # reset one-shot mode after successful action
                            self.next_half_mode = MoveMode.LEAVE_ONE
                        else:
                            # illegal: do NOT advance time, keep selection
                            pass


            # Draw
            self.screen.fill(C_BG)
            self.draw_topbar()
            self.draw_grid()
            self.draw_sidebar(last_result=last_result)
            pygame.display.flip()

            # Auto-stop on terminal; allow reset with R
            if last_result is not None and (last_result.terminated or last_result.truncated):
                # keep rendering; just don't auto-advance
                pass

        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    Simulator().run()
