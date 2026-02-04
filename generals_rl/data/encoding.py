from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch

def encode_obs_sequence(
    obs_seq: List[Dict[str, np.ndarray]],
    player_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a sequence of observations into tensors.

    Returns:
      x_img: (T, C, H, W) float32
      x_meta: (T, M) float32
    """
    T = len(obs_seq)
    if T == 0:
        raise ValueError("obs_seq is empty")

    H, W = obs_seq[0]["tile_type"].shape
    self_id = int(player_id)
    opp_id = 1 - self_id

    C = 20
    M = 10
    x_img = np.zeros((T, C, H, W), dtype=np.float32)
    x_meta = np.zeros((T, M), dtype=np.float32)

    for t in range(T):
        obs = obs_seq[t]
        tile = obs["tile_type"].astype(np.int32)
        owner = obs["owner"].astype(np.int32)
        army = obs["army"].astype(np.int32)
        vis = obs["visible"].astype(np.float32)

        mem = obs.get("memory_tag", np.zeros((H, W), dtype=np.int8)).astype(np.int32)
        exposed_city = obs.get("exposed_my_city_mask", np.zeros((H, W), dtype=np.bool_)).astype(np.float32)

        for tt in range(4):
            x_img[t, tt] = (tile == tt).astype(np.float32)

        x_img[t, 4] = ((owner == self_id) & (vis > 0)).astype(np.float32)
        x_img[t, 5] = ((owner == opp_id) & (vis > 0)).astype(np.float32)
        x_img[t, 6] = ((owner == -1) & (vis > 0)).astype(np.float32)

        army_safe = np.maximum(army, 0)
        self_army = np.where(owner == self_id, army_safe, 0)
        opp_army = np.where(owner == opp_id, army_safe, 0)
        x_img[t, 7] = np.log1p(self_army).astype(np.float32)
        x_img[t, 8] = np.log1p(opp_army).astype(np.float32)

        x_img[t, 9] = vis

        for k in range(8):
            x_img[t, 10 + k] = (mem == k).astype(np.float32)

        x_img[t, 18] = exposed_city
        x_img[t, 19] = 1.0

        half_in_turn = float(obs["half_in_turn"][0])
        turn = int(obs["turn"][0])
        turn_frac = float(turn % 25) / 25.0

        total_army = obs["total_army"].astype(np.float32)
        total_land = obs["total_land"].astype(np.float32)
        my_army_total = float(total_army[self_id])
        opp_army_total = float(total_army[opp_id])
        my_land_total = float(total_land[self_id])
        opp_land_total = float(total_land[opp_id])

        real_shape = obs["real_shape"].astype(np.float32)
        real_h = float(real_shape[0]) / 25.0
        real_w = float(real_shape[1]) / 25.0

        exposed_my_general_seen = float(obs.get("exposed_my_general_seen", np.array([0], dtype=np.bool_))[0])
        exposed_my_city_count = float(obs.get("exposed_my_city_count", np.array([0], dtype=np.int32))[0]) / (25.0 * 25.0)

        x_meta[t] = np.array(
            [
                half_in_turn,
                turn_frac,
                np.log1p(my_army_total) / 8.0,
                np.log1p(opp_army_total) / 8.0,
                my_land_total / (25.0 * 25.0),
                opp_land_total / (25.0 * 25.0),
                real_h,
                real_w,
                exposed_my_general_seen,
                exposed_my_city_count,
            ],
            dtype=np.float32,
        )

    return torch.from_numpy(x_img), torch.from_numpy(x_meta)
