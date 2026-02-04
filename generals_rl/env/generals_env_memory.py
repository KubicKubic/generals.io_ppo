# generals_env_memory.py
# Wrapper env that adds per-player "automatic memory" tags for fog cells,
# and also tracks which of your CITY ("tower") and GENERAL ("king")
# have ever been seen by the opponent.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from .generals_env import GeneralsEnv, Owner, TileType, StepResult


class GeneralsEnvWithMemory:
    """
    Wrapper around GeneralsEnv.

    Adds to each player's observation:
      - memory_tag: (H,W) int8, only non-zero on CURRENT fog cells, tags 1..7 per user's definition.
      - exposed_my_city_mask: (H,W) bool, True where "my city has been seen by opponent at least once"
      - exposed_my_general_seen: (1,) bool, whether "my general has been seen by opponent at least once"
      - exposed_my_city_count: (1,) int32, number of True in exposed_my_city_mask

    Memory definition:
      "曾经看到过" means: previously visible=True at some time for that player.
      We track LAST seen true (tile_type, owner) for each cell and each player.

    Memory tags (for a cell that is currently in fog):
      1: 曾经看到过这是山
      2: 曾经看到过这是不属于对方的城市（中立城市 or 自己的城市）
      3: 曾经看到过这是属于对方的王（对方general）
      4: 曾经看到过这是属于对方的地（对方拥有的普通地）
      5: 曾经看到过这是属于对方的城市
      6: 曾经是属于你的地（后来被对方占有）
      7: 曾经是属于你的城市（后来被对方占有）

    Exposure tracking:
      "被对手看见过" means: at some time, the opponent's visible mask included that cell
      AND at that time it was (tile_type==CITY and owner==me) for tower exposure,
      or (tile_type==GENERAL and owner==me) for king exposure.
      Once True, stays True for the rest of the episode.
    """

    def __init__(self, *args, **kwargs):
        self.env = GeneralsEnv(*args, **kwargs)

        H, W = self.env.H, self.env.W

        # last seen true tile_type for each player; init as -1 (unknown)
        self._last_tile = {
            Owner.P0: np.full((H, W), -1, dtype=np.int16),
            Owner.P1: np.full((H, W), -1, dtype=np.int16),
        }
        # last seen true owner for each player; init as -2 (unknown)
        self._last_owner = {
            Owner.P0: np.full((H, W), -2, dtype=np.int16),
            Owner.P1: np.full((H, W), -2, dtype=np.int16),
        }
        self._ever_seen = {
            Owner.P0: np.zeros((H, W), dtype=np.bool_),
            Owner.P1: np.zeros((H, W), dtype=np.bool_),
        }

        # Exposure records: my assets seen by opponent
        self._exposed_city = {
            Owner.P0: np.zeros((H, W), dtype=np.bool_),
            Owner.P1: np.zeros((H, W), dtype=np.bool_),
        }
        self._exposed_general = {
            Owner.P0: False,
            Owner.P1: False,
        }

    # ---- passthrough properties ----
    @property
    def H(self): return self.env.H

    @property
    def W(self): return self.env.W

    @property
    def action_size(self): return self.env.action_size

    @property
    def half_t(self): return self.env.half_t

    def legal_action_mask(self, p: Owner) -> np.ndarray:
        return self.env.legal_action_mask(p)

    # ---- core API ----
    def reset(self) -> Tuple[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]], Dict]:
        (obs0, obs1), info = self.env.reset()

        # clear memory + exposure
        for p in (Owner.P0, Owner.P1):
            self._last_tile[p].fill(-1)
            self._last_owner[p].fill(-2)
            self._ever_seen[p].fill(False)
            self._exposed_city[p].fill(False)
            self._exposed_general[p] = False

        # update memory
        self._update_memory_from_obs(Owner.P0, obs0)
        self._update_memory_from_obs(Owner.P1, obs1)

        # update exposure (who sees whose assets)
        self._update_exposure_from_obs_pair(obs0, obs1)

        # augment
        obs0 = self._augment_obs(Owner.P0, obs0)
        obs1 = self._augment_obs(Owner.P1, obs1)
        return (obs0, obs1), info

    def step(self, action0: int, action1: int) -> StepResult:
        res = self.env.step(action0, action1)
        obs0, obs1 = res.obs

        # update memory (newly visible)
        self._update_memory_from_obs(Owner.P0, obs0)
        self._update_memory_from_obs(Owner.P1, obs1)

        # update exposure (newly seen assets)
        self._update_exposure_from_obs_pair(obs0, obs1)

        # augment obs
        obs0 = self._augment_obs(Owner.P0, obs0)
        obs1 = self._augment_obs(Owner.P1, obs1)

        return StepResult((obs0, obs1), res.reward, res.terminated, res.truncated, res.info)

    def obs_for_player(self, p: Owner) -> Dict[str, np.ndarray]:
        """
        Convenience: get augmented observation for one player.
        (Does NOT advance env time.)
        """
        base = self.env.obs_for_player(p)
        self._update_memory_from_obs(p, base)
        # exposure depends on opponent obs, so we don't update exposure here.
        return self._augment_obs(p, base)

    # ---- internal: update memory ----
    def _update_memory_from_obs(self, p: Owner, obs: Dict[str, np.ndarray]) -> None:
        """
        For visible cells, record true tile_type & owner as last seen.
        """
        vis = obs["visible"].astype(np.bool_)
        tile = obs["tile_type"].astype(np.int16)   # visible cells are true tile_type in your env
        owner = obs["owner"].astype(np.int16)      # visible cells are true owner

        self._last_tile[p][vis] = tile[vis]
        self._last_owner[p][vis] = owner[vis]
        self._ever_seen[p][vis] = True

    # ---- internal: update exposure ----
    def _update_exposure_from_obs_pair(self, obs0: Dict[str, np.ndarray], obs1: Dict[str, np.ndarray]) -> None:
        """
        Use each player's *visible* cells to mark which opponent assets they have seen.
        """
        self._mark_seen_assets(viewer=Owner.P0, viewer_obs=obs0, target=Owner.P1)
        self._mark_seen_assets(viewer=Owner.P1, viewer_obs=obs1, target=Owner.P0)

    def _mark_seen_assets(self, viewer: Owner, viewer_obs: Dict[str, np.ndarray], target: Owner) -> None:
        """
        If viewer can see a cell that is (CITY owned by target) => mark target's exposed_city at that cell.
        If viewer can see a cell that is (GENERAL owned by target) => mark target's exposed_general True.
        """
        vis = viewer_obs["visible"].astype(np.bool_)
        if not vis.any():
            return

        # viewer_obs provides true tile/owner on visible cells
        tile = viewer_obs["tile_type"].astype(np.int16)
        owner = viewer_obs["owner"].astype(np.int16)

        # seen target cities
        seen_city = vis & (tile == TileType.CITY) & (owner == int(target))
        if seen_city.any():
            self._exposed_city[target][seen_city] = True

        # seen target general
        seen_gen = vis & (tile == TileType.GENERAL) & (owner == int(target))
        if seen_gen.any():
            self._exposed_general[target] = True

    # ---- internal: derive memory_tag and augment obs ----
    def _augment_obs(self, p: Owner, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add memory_tag + exposure fields to obs.
        """
        memory_tag = self._compute_memory_tag(p, obs)

        out = dict(obs)
        out["memory_tag"] = memory_tag

        # exposure fields for THIS player (i.e., "my assets seen by opponent")
        my_city_mask = self._exposed_city[p].copy()
        out["exposed_my_city_mask"] = my_city_mask.astype(np.bool_)
        out["exposed_my_city_count"] = np.array([int(my_city_mask.sum())], dtype=np.int32)
        out["exposed_my_general_seen"] = np.array([bool(self._exposed_general[p])], dtype=np.bool_)

        return out

    def _compute_memory_tag(self, p: Owner, obs: Dict[str, np.ndarray]) -> np.ndarray:
        vis = obs["visible"].astype(np.bool_)
        fog = ~vis

        lt = self._last_tile[p]
        lo = self._last_owner[p]
        seen = self._ever_seen[p]

        tag = np.zeros((self.H, self.W), dtype=np.int8)

        m = fog & seen
        if not m.any():
            return tag

        opp = Owner.P1 if p == Owner.P0 else Owner.P0
        self_id = p

        lt_m = lt[m]  # (N,)
        lo_m = lo[m]  # (N,)
        out = np.zeros(lt_m.shape[0], dtype=np.int8)

        # 1) mountain
        mountain_mask = (lt_m == TileType.MOUNTAIN)
        out[mountain_mask] = 1

        # 2/5/7) city
        city_mask = (lt_m == TileType.CITY)
        # default: non-opponent city (neutral or mine)
        out[city_mask] = 2
        # opponent city
        out[city_mask & (lo_m == int(opp))] = 5
        # my city (lost)
        out[city_mask & (lo_m == int(self_id))] = 7

        # 3) opponent general
        gen_mask = (lt_m == TileType.GENERAL)
        out[gen_mask & (lo_m == int(opp))] = 3

        # 4/6) empty tiles (land)
        empty_mask = (lt_m == TileType.EMPTY)
        out[empty_mask & (lo_m == int(opp))] = 4
        out[empty_mask & (lo_m == int(self_id))] = 6
        # neutral empty remains 0

        tag[m] = out
        return tag
