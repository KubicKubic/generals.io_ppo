from __future__ import annotations

from typing import Dict, List
import numpy as np

class ObsHistory:
    """Stores last N observations (dicts) and returns padded sequences."""
    def __init__(self, max_len: int = 100):
        self.max_len = int(max_len)
        self.buf: List[Dict[str, np.ndarray]] = []

    def reset(self, first_obs: Dict[str, np.ndarray]):
        self.buf = [first_obs]

    def push(self, obs: Dict[str, np.ndarray]):
        self.buf.append(obs)
        if len(self.buf) > self.max_len:
            self.buf = self.buf[-self.max_len:]

    def get_padded_seq(self) -> List[Dict[str, np.ndarray]]:
        if not self.buf:
            raise RuntimeError("ObsHistory empty; call reset() first.")
        if len(self.buf) >= self.max_len:
            return self.buf[-self.max_len:]
        pad = [self.buf[0]] * (self.max_len - len(self.buf))
        return pad + self.buf
