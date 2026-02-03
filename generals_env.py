# generals_env.py
# A minimal generals.io-like environment with:
#  - 2-player play
#  - Fog of war observations (FOG RENDERING):
#       * in fog: MOUNTAIN and CITY are rendered as MOUNTAIN
#       * everything else is rendered as EMPTY
#       * owner/army remain unknown
#  - Each env.step() = ONE half-turn (obs updates every half-turn)
#  - Each turn = 2 half-turns; growth happens ONCE per turn (after the 2nd half-turn)
#  - Move execution order inside a half-turn decided by a priority system (sequential execution)
#  - Variable "real" map size sampled uniformly from 15..25 (both H and W),
#    embedded in a fixed 25x25 grid; outside real area is filled with mountains.
#  - HARD constraints:
#       * Manhattan distance between the two generals >= 15
#       * The two generals must be connected via passable land (non-mountain tiles)

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Dict, List
import numpy as np


# -----------------------------
# Constants / enums
# -----------------------------

class TileType(IntEnum):
    EMPTY = 0
    MOUNTAIN = 1
    CITY = 2
    GENERAL = 3


class Owner(IntEnum):
    NEUTRAL = -1
    P0 = 0
    P1 = 1


class MoveMode(IntEnum):
    LEAVE_ONE = 0      # move all but 1
    HALF = 1           # move floor(army/2)


class Dir(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DIRS = {
    Dir.UP: (-1, 0),
    Dir.RIGHT: (0, 1),
    Dir.DOWN: (1, 0),
    Dir.LEFT: (0, -1),
}


# Action encoding:
# a = (((r * W + c) * 4 + dir) * 2 + mode)
def encode_action(r: int, c: int, d: int, mode: int, W: int) -> int:
    return (((r * W + c) * 4 + d) * 2 + mode)


def decode_action(a: int, W: int) -> Tuple[int, int, int, int]:
    mode = a % 2
    a //= 2
    d = a % 4
    a //= 4
    idx = a
    r = idx // W
    c = idx % W
    return r, c, d, mode


@dataclass(frozen=True)
class StepResult:
    # obs is (obs_p0, obs_p1)
    obs: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
    reward: Tuple[float, float]
    terminated: bool
    truncated: bool
    info: Dict


@dataclass
class _Intent:
    player: Owner
    action: int
    # tags for priority
    is_chase: bool
    is_defensive: bool
    tile_size: int
    is_attack_general: bool
    # deterministic tie-breaker
    tie: int


class GeneralsEnv:
    """
    Two-player generals-like env.

    Fixed grid is self.H x self.W (defaults 25x25).
    Each reset samples a "real" map size (real_H, real_W) uniformly from 15..25,
    and fills outside [0:real_H, 0:real_W] with mountains.

    State tensors (H, W):
      - tile_type: TileType (EMPTY/MOUNTAIN/CITY/GENERAL)
      - owner: Owner (NEUTRAL/P0/P1)
      - army: int >= 0

    Observation per player:
      - tile_type: int with fog rendering:
          * visible => true tile_type
          * fog     => (CITY/MOUNTAIN -> MOUNTAIN, else -> EMPTY)
      - owner: int, -2 for fog
      - army: int, -1 for fog
      - visible: bool mask (true visibility, not fog-render)
      - t: half-turn index as array([half_t])
      - turn: turn index as array([turn])
      - half_in_turn: array([0 or 1])
      - real_shape: array([real_H, real_W])
    """

    def __init__(
        self,
        H: int = 25,
        W: int = 25,
        seed: Optional[int] = None,
        round_len: int = 25,          # in TURNS
        city_initial_low: int = 40,
        city_initial_high: int = 50,
        general_initial: int = 1,
        vision_radius: int = 1,
        max_halfturns: int = 4000,    # truncation in half-turns

        # matching knobs
        round_includes_cities: bool = True,
        enable_chase_priority: bool = True,

        # map constraints
        min_real_size: int = 15,
        max_real_size: int = 25,
        min_general_dist: int = 15,
    ):
        self.H, self.W = int(H), int(W)
        if self.H != 25 or self.W != 25:
            # You can change this, but your requirement says default 25.
            pass

        self.rng = np.random.default_rng(seed)

        self.round_len = int(round_len)
        self.city_initial_low = int(city_initial_low)
        self.city_initial_high = int(city_initial_high)
        self.general_initial = int(general_initial)
        self.vision_radius = int(vision_radius)

        self.max_halfturns = int(max_halfturns)

        self.round_includes_cities = bool(round_includes_cities)
        self.enable_chase_priority = bool(enable_chase_priority)

        self.min_real_size = int(min_real_size)
        self.max_real_size = int(max_real_size)
        self.min_general_dist = int(min_general_dist)

        # state arrays are fixed size (25x25 by default)
        self.tile_type = np.zeros((self.H, self.W), dtype=np.int8)
        self.owner = np.full((self.H, self.W), Owner.NEUTRAL, dtype=np.int8)
        self.army = np.zeros((self.H, self.W), dtype=np.int32)

        self.general_pos: Dict[Owner, Optional[Tuple[int, int]]] = {Owner.P0: None, Owner.P1: None}

        # real map size for current episode
        self.real_H = self.H
        self.real_W = self.W

        # time
        self.half_t = 0  # each step() increments by 1
        self.t = 0       # kept for compatibility; equals half_t

        # action space size (fixed)
        self.action_size = self.H * self.W * 4 * 2

    # -----------------------------
    # Public API
    # -----------------------------
    def reset(self) -> Tuple[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]], Dict]:
        self.half_t = 0
        self.t = 0
        self._generate_map()

        obs0 = self._make_obs(Owner.P0)
        obs1 = self._make_obs(Owner.P1)
        return (obs0, obs1), {"half_t": self.half_t, "turn": self.half_t // 2, "real_shape": (self.real_H, self.real_W)}

    def step(self, action0: int, action1: int) -> StepResult:
        """
        One env step = ONE half-turn.
        Both players submit one action each; execution order decided by priority system.
        Growth happens ONCE per turn, i.e., after every 2 half-turns.
        """
        self.half_t += 1
        self.t = self.half_t

        # 1) moves in this half-turn
        self._apply_halfturn_priority(action0, action1)

        # 2) terminal check after moves
        term, winner = self._check_terminal()

        # 3) growth only at end of turn (after 2nd half-turn), and only if not terminated
        if (not term) and (self.half_t % 2 == 0):
            self._apply_growth_turn()
            term, winner = self._check_terminal()

        # 4) rewards (sparse win/lose)
        r0 = r1 = 0.0
        if term:
            if winner == Owner.P0:
                r0, r1 = 1.0, -1.0
            elif winner == Owner.P1:
                r0, r1 = -1.0, 1.0

        truncated = (self.half_t >= self.max_halfturns) and (not term)

        obs0 = self._make_obs(Owner.P0)
        obs1 = self._make_obs(Owner.P1)
        info = {
            "half_t": self.half_t,
            "turn": self.half_t // 2,
            "half_in_turn": self.half_t % 2,
            "winner": int(winner) if winner is not None else None,
            "real_shape": (self.real_H, self.real_W),
        }
        return StepResult((obs0, obs1), (r0, r1), term, truncated, info)

    # Convenience helpers for RL
    def obs_for_player(self, p: Owner) -> Dict[str, np.ndarray]:
        return self._make_obs(p)

    def legal_action_mask(self, p: Owner) -> np.ndarray:
        """
        Return (action_size,) bool mask for legal actions for player p,
        computed from TRUE internal state (not fog).
        """
        mask = np.zeros((self.action_size,), dtype=np.bool_)

        for r in range(self.H):
            for c in range(self.W):
                if self.owner[r, c] != p:
                    continue
                if self.tile_type[r, c] == TileType.MOUNTAIN:
                    continue
                if self.army[r, c] < 2:
                    continue
                for d in range(4):
                    dr, dc = DIRS[Dir(d)]
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < self.H and 0 <= cc < self.W):
                        continue
                    if self.tile_type[rr, cc] == TileType.MOUNTAIN:
                        continue
                    for mode in range(2):
                        a = encode_action(r, c, d, mode, self.W)
                        mask[a] = True

        if not mask.any():
            mask[0] = True
        return mask

    # -----------------------------
    # Map generation
    # -----------------------------
    def _generate_map(self):
        """
        Requirements implemented:
          - self.H/self.W fixed (default 25x25)
          - real_H, real_W sampled uniformly from [15, 25]
          - outside real area filled with mountains
          - generals Manhattan distance >= 15 (hard)
          - each city initial army uniform in [40, 50]
          - generals must be connected via passable land (non-mountain tiles)
        """
        # Sample real dimensions uniformly
        self.real_H = int(self.rng.integers(self.min_real_size, self.max_real_size + 1))
        self.real_W = int(self.rng.integers(self.min_real_size, self.max_real_size + 1))

        # We will retry whole map generation until constraints are satisfied.
        for _attempt in range(500):
            self._init_base_with_padding()

            # Place random internal mountains in the real area
            self._place_internal_mountains()

            # Place generals with hard distance constraint
            ok_generals = self._place_generals_with_constraints()
            if not ok_generals:
                continue

            # Connectivity constraint (passable = non-mountain)
            if not self._generals_connected():
                continue

            # Place cities (neutral) with random size 40..50
            self._place_neutral_cities()

            # Ensure generals still exist and remain on passable tiles
            # (cities don't affect passability; still safe to assert)
            if not self._generals_connected():
                continue

            return

        raise RuntimeError("Failed to generate a valid map with given constraints after many attempts.")

    def _init_base_with_padding(self):
        """
        Set outside [0:real_H,0:real_W] to mountains.
        Inside real area, start with empty neutral land.
        """
        # default everything to mountain
        self.tile_type.fill(TileType.MOUNTAIN)
        self.owner.fill(Owner.NEUTRAL)
        self.army.fill(0)

        # carve real area as empty
        self.tile_type[: self.real_H, : self.real_W] = TileType.EMPTY

        # clear general positions
        self.general_pos[Owner.P0] = None
        self.general_pos[Owner.P1] = None

    def _place_internal_mountains(self):
        """
        Place additional mountains within the real area.
        Uses simple random placement with collisions possible (so actual count <= target).
        """
        area = self.real_H * self.real_W
        n_m = area // 12
        for _ in range(n_m):
            r = int(self.rng.integers(0, self.real_H))
            c = int(self.rng.integers(0, self.real_W))
            self.tile_type[r, c] = TileType.MOUNTAIN
            # owner/army stay neutral/0

    def _random_empty_in_real(self) -> Tuple[int, int]:
        """
        Pick a cell in real area whose tile_type is EMPTY.
        """
        for _ in range(20000):
            r = int(self.rng.integers(0, self.real_H))
            c = int(self.rng.integers(0, self.real_W))
            if self.tile_type[r, c] == TileType.EMPTY:
                return r, c
        raise RuntimeError("No empty cell found in real area (too many mountains).")

    def _place_general(self, p: Owner, pos: Tuple[int, int]):
        r, c = pos
        self.tile_type[r, c] = TileType.GENERAL
        self.owner[r, c] = p
        self.army[r, c] = self.general_initial
        self.general_pos[p] = pos

    def _place_generals_with_constraints(self) -> bool:
        """
        Place two generals in the real area:
          - only on EMPTY cells
          - Manhattan distance >= min_general_dist (hard)
        """
        # Ensure we have enough empty cells
        # We'll attempt a number of samples.
        for _ in range(2000):
            g0 = self._random_empty_in_real()
            # Temporarily mark g0 as non-empty to avoid picking same cell for g1
            self.tile_type[g0] = TileType.GENERAL  # temporary mark; will set properly below

            success = False
            for _ in range(2000):
                g1 = self._random_empty_in_real()
                dist = abs(g0[0] - g1[0]) + abs(g0[1] - g1[1])
                if dist >= self.min_general_dist:
                    success = True
                    break

            # revert temporary mark
            self.tile_type[g0] = TileType.EMPTY

            if not success:
                continue

            # Place both generals properly
            self._place_general(Owner.P0, g0)
            self._place_general(Owner.P1, g1)
            return True

        return False

    def _generals_connected(self) -> bool:
        """
        Check if two generals are connected via passable land (non-mountain tiles),
        within the real area.
        """
        g0 = self.general_pos[Owner.P0]
        g1 = self.general_pos[Owner.P1]
        if g0 is None or g1 is None:
            return False

        (sr, sc) = g0
        (tr, tc) = g1

        # BFS on passable cells: tile_type != MOUNTAIN
        passable = (self.tile_type[: self.real_H, : self.real_W] != TileType.MOUNTAIN)

        if not passable[sr, sc] or not passable[tr, tc]:
            return False

        q_r = np.empty(self.real_H * self.real_W, dtype=np.int32)
        q_c = np.empty(self.real_H * self.real_W, dtype=np.int32)
        head = 0
        tail = 0

        visited = np.zeros((self.real_H, self.real_W), dtype=np.bool_)
        visited[sr, sc] = True
        q_r[tail] = sr
        q_c[tail] = sc
        tail += 1

        while head < tail:
            r = int(q_r[head])
            c = int(q_c[head])
            head += 1

            if (r, c) == (tr, tc):
                return True

            # 4-neighbors
            if r > 0 and (not visited[r - 1, c]) and passable[r - 1, c]:
                visited[r - 1, c] = True
                q_r[tail] = r - 1
                q_c[tail] = c
                tail += 1
            if r + 1 < self.real_H and (not visited[r + 1, c]) and passable[r + 1, c]:
                visited[r + 1, c] = True
                q_r[tail] = r + 1
                q_c[tail] = c
                tail += 1
            if c > 0 and (not visited[r, c - 1]) and passable[r, c - 1]:
                visited[r, c - 1] = True
                q_r[tail] = r
                q_c[tail] = c - 1
                tail += 1
            if c + 1 < self.real_W and (not visited[r, c + 1]) and passable[r, c + 1]:
                visited[r, c + 1] = True
                q_r[tail] = r
                q_c[tail] = c + 1
                tail += 1

        return False

    def _place_neutral_cities(self):
        """
        Place neutral cities in the real area on EMPTY cells.
        Each city initial army sampled uniformly from [city_initial_low, city_initial_high].
        """
        area = self.real_H * self.real_W
        n_c = max(2, area // 50)

        placed = 0
        for _ in range(20000):
            if placed >= n_c:
                break
            r = int(self.rng.integers(0, self.real_H))
            c = int(self.rng.integers(0, self.real_W))
            if self.tile_type[r, c] != TileType.EMPTY:
                continue
            # Place city
            self.tile_type[r, c] = TileType.CITY
            self.owner[r, c] = Owner.NEUTRAL
            self.army[r, c] = int(self.rng.integers(self.city_initial_low, self.city_initial_high + 1))
            placed += 1

        if placed < n_c:
            # Not fatal, but typically you want fixed city count; treat as error for reproducibility
            raise RuntimeError("Failed to place required number of cities (too few EMPTY cells).")

    # -----------------------------
    # Observation (fog of war)
    # -----------------------------
    def _visible_mask(self, p: Owner) -> np.ndarray:
        owned = (self.owner == p)
        vis = owned.copy()

        # expand Manhattan radius by 4-neighborhood
        expand = owned.copy()
        for _ in range(self.vision_radius):
            up = np.pad(expand[1:, :], ((0, 1), (0, 0)), constant_values=False)
            down = np.pad(expand[:-1, :], ((1, 0), (0, 0)), constant_values=False)
            left = np.pad(expand[:, 1:], ((0, 0), (0, 1)), constant_values=False)
            right = np.pad(expand[:, :-1], ((0, 0), (1, 0)), constant_values=False)
            expand = expand | up | down | left | right

        vis |= expand
        return vis
    
    def _compute_scores(self) -> Tuple[int, int, int, int]:
        """
        Returns:
        army0, army1, land0, land1
        land includes ALL tiles owned by player (including GENERAL and CITY tiles).
        Mountains are never owned.
        """
        o = self.owner
        a = self.army

        m0 = (o == Owner.P0)
        m1 = (o == Owner.P1)

        army0 = int(a[m0].sum())
        army1 = int(a[m1].sum())
        land0 = int(m0.sum())
        land1 = int(m1.sum())
        return army0, army1, land0, land1

    def _make_obs(self, p: Owner) -> Dict[str, np.ndarray]:
        vis = self._visible_mask(p)

        # Fog rendering for tile_type:
        #  - MOUNTAIN and CITY render as MOUNTAIN
        #  - everything else render as EMPTY
        fog_render = np.where(
            (self.tile_type == TileType.MOUNTAIN) | (self.tile_type == TileType.CITY),
            TileType.MOUNTAIN,
            TileType.EMPTY,
        ).astype(np.int16)

        tile_type_obs = np.where(vis, self.tile_type.astype(np.int16), fog_render)
        owner_obs = np.where(vis, self.owner, -2).astype(np.int16)  # -2 => unknown
        army_obs = np.where(vis, self.army, -1).astype(np.int32)    # -1 => unknown

        army0, army1, land0, land1 = self._compute_scores()

        return {
            "tile_type": tile_type_obs,
            "owner": owner_obs,
            "army": army_obs,
            "visible": vis.astype(np.bool_),

            # time
            "t": np.array([self.half_t], dtype=np.int32),              # existing, half-turn
            "half_t": np.array([self.half_t], dtype=np.int32),         # explicit alias
            "turn": np.array([self.half_t // 2], dtype=np.int32),
            "half_in_turn": np.array([self.half_t % 2], dtype=np.int32),

            # global scores (both players)
            "total_army": np.array([army0, army1], dtype=np.int32),    # [P0, P1]
            "total_land": np.array([land0, land1], dtype=np.int32),    # [P0, P1]

            # map size info (you already added)
            "real_shape": np.array([self.real_H, self.real_W], dtype=np.int32),
        }

    # -----------------------------
    # Move parsing / validation
    # -----------------------------
    def _parse_and_validate(self, action: int, p: Owner):
        if not (0 <= action < self.action_size):
            return None

        r, c, d, mode = decode_action(action, self.W)
        if not (0 <= r < self.H and 0 <= c < self.W):
            return None
        if self.owner[r, c] != p:
            return None
        if self.tile_type[r, c] == TileType.MOUNTAIN:
            return None
        if self.army[r, c] < 2:
            return None

        dr, dc = DIRS[Dir(d)]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < self.H and 0 <= cc < self.W):
            return None
        if self.tile_type[rr, cc] == TileType.MOUNTAIN:
            return None

        return (p, r, c, rr, cc, MoveMode(mode))

    def _compute_sent(self, r: int, c: int, mode: MoveMode) -> int:
        a = int(self.army[r, c])
        if mode == MoveMode.HALF:
            sent = a // 2
        else:
            sent = a - 1
        return max(0, sent)

    # -----------------------------
    # Combat / apply move (sequential)
    # -----------------------------
    def _resolve_arrival(self, attacker: Owner, r: int, c: int, sent: int):
        if sent <= 0:
            return

        ttype = TileType(int(self.tile_type[r, c]))
        if ttype == TileType.MOUNTAIN:
            return

        cur_owner = Owner(int(self.owner[r, c]))
        cur_army = int(self.army[r, c])

        if cur_owner == attacker:
            self.army[r, c] = cur_army + sent
            return

        if sent > cur_army:
            new_army = sent - cur_army
            self.owner[r, c] = attacker
            self.army[r, c] = new_army
        else:
            self.army[r, c] = cur_army - sent

    # -----------------------------
    # Priority system for a half-turn
    # -----------------------------
    def _is_enemy_general_target(self, attacker: Owner, rr: int, cc: int) -> bool:
        if self.tile_type[rr, cc] != TileType.GENERAL:
            return False
        return self.owner[rr, cc] != attacker

    def _classify_action(self, action: int, p: Owner, other_move) -> Optional[_Intent]:
        """
        Classify using the state BEFORE any move in this half-turn executes.
        `other_move` is the opponent's validated move tuple (or None), used for approximate chase.
        """
        move = self._parse_and_validate(action, p)
        if move is None:
            return None

        _, r, c, rr, cc, _mode = move
        tile_size = int(self.army[r, c])  # proxy for "largest tile"

        is_defensive = (self.owner[rr, cc] == p)
        is_attack_general = self._is_enemy_general_target(p, rr, cc)

        # Approx chase (needs calibration for a perfect match):
        # treat a swap as chase.
        is_chase = False
        if self.enable_chase_priority and other_move is not None:
            op, orr, occ, orrr, occc, _omode = other_move
            if (rr, cc) == (orr, occ) and (orrr, occc) == (r, c):
                is_chase = True

        tie = 0 if p == Owner.P0 else 1
        return _Intent(
            player=p,
            action=action,
            is_chase=bool(is_chase),
            is_defensive=bool(is_defensive),
            tile_size=tile_size,
            is_attack_general=bool(is_attack_general),
            tie=tie,
        )

    def _priority_key(self, it: _Intent) -> Tuple[int, int, int, int, int]:
        """
        Higher key executes earlier (sorted reverse=True).

        Priority structure:
          1) chase moves highest
          2) defensive (friendly->friendly)
          3) larger source tile earlier
          4) attack-general moves come after all other moves
        """
        chase = 1 if it.is_chase else 0
        defensive = 1 if it.is_defensive else 0
        size = it.tile_size
        non_attack_general = 0 if it.is_attack_general else 1

        # Deterministic tie-breaker: P0 before P1 (does not claim online correctness).
        tie = 1 if it.tie == 0 else 0

        return (chase, defensive, size, non_attack_general, tie)

    def _execute_one_action_sequential(self, action: int, p: Owner):
        """
        Execute a move sequentially on the CURRENT state.
        Re-validate because earlier moves might have changed legality/armies/ownership.
        """
        move = self._parse_and_validate(action, p)
        if move is None:
            return
        _, r, c, rr, cc, mode = move

        sent = self._compute_sent(r, c, mode)
        if sent <= 0:
            return

        self.army[r, c] -= sent
        self._resolve_arrival(p, rr, cc, sent)

    def _apply_halfturn_priority(self, action0: int, action1: int):
        """
        Apply one half-turn using priority ordering.
        We classify both moves on the pre-execution state, sort, then execute sequentially.
        """
        m0 = self._parse_and_validate(action0, Owner.P0)
        m1 = self._parse_and_validate(action1, Owner.P1)

        i0 = self._classify_action(action0, Owner.P0, m1)
        i1 = self._classify_action(action1, Owner.P1, m0)

        intents: List[_Intent] = [x for x in (i0, i1) if x is not None]
        intents.sort(key=self._priority_key, reverse=True)

        for it in intents:
            self._execute_one_action_sequential(it.action, it.player)

    # -----------------------------
    # Growth (TURN-level)
    # -----------------------------
    def _apply_growth_turn(self):
        """
        Apply growth once per TURN (after each pair of half-turns).
        """
        # generals +1 / turn
        for p in (Owner.P0, Owner.P1):
            gp = self.general_pos[p]
            if gp is not None:
                r, c = gp
                if self.owner[r, c] == p:
                    self.army[r, c] += 1

        # owned cities +1 / turn
        is_city = (self.tile_type == TileType.CITY)
        self.army[is_city & (self.owner != Owner.NEUTRAL)] += 1

        # every round_len TURNS: +1 on owned tiles (optionally includes cities)
        turn_idx = self.half_t // 2
        if turn_idx % self.round_len == 0:
            owned = (self.owner != Owner.NEUTRAL)
            if self.round_includes_cities:
                not_mountain = (self.tile_type != TileType.MOUNTAIN)
                self.army[owned & not_mountain] += 1
            else:
                is_normal = (self.tile_type == TileType.EMPTY) | (self.tile_type == TileType.GENERAL)
                self.army[owned & is_normal] += 1

    # -----------------------------
    # Terminal
    # -----------------------------
    def _check_terminal(self) -> Tuple[bool, Optional[Owner]]:
        g0 = self.general_pos[Owner.P0]
        g1 = self.general_pos[Owner.P1]
        if g0 is None or g1 is None:
            return True, None

        r0, c0 = g0
        r1, c1 = g1

        if self.owner[r0, c0] != Owner.P0:
            return True, Owner.P1
        if self.owner[r1, c1] != Owner.P1:
            return True, Owner.P0
        return False, None


# -----------------------------
# Quick sanity check (optional)
# -----------------------------
if __name__ == "__main__":
    env = GeneralsEnv(seed=0)
    (o0, o1), info = env.reset()
    print("reset:", info, "real_shape:", o0["real_shape"])
    for i in range(20):
        a0 = int(np.random.randint(env.action_size))
        a1 = int(np.random.randint(env.action_size))
        res = env.step(a0, a1)
        if i < 5:
            # show a small slice of tile_type obs for p0
            print("half_t:", res.info["half_t"], "turn:", res.info["turn"], "real_shape:", res.info["real_shape"])
        if res.terminated or res.truncated:
            print("done:", res.info)
            break
