from __future__ import annotations
import random
import numpy as np
import torch
from typing import Dict

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_rng_state() -> Dict:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state: Dict):
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and "torch_cuda_random" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])
