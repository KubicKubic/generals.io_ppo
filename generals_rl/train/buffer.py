from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch

@dataclass
class Batch:
    x_img: torch.Tensor
    x_meta: torch.Tensor
    maskA: torch.Tensor
    a: torch.Tensor
    logp: torch.Tensor
    v: torch.Tensor
    r: torch.Tensor
    done: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor

class RolloutBuffer:
    def __init__(self, device):
        self.device = device
        self.x_img: List[torch.Tensor] = []
        self.x_meta: List[torch.Tensor] = []
        self.maskA: List[torch.Tensor] = []
        self.a: List[torch.Tensor] = []
        self.logp: List[torch.Tensor] = []
        self.v: List[torch.Tensor] = []
        self.r: List[float] = []
        self.done: List[float] = []

    def add(self, x_img_seq, x_meta_seq, maskA, a, logp, v, r: float, done: bool):
        self.x_img.append(x_img_seq)
        self.x_meta.append(x_meta_seq)
        self.maskA.append(maskA)
        self.a.append(a)
        self.logp.append(logp)
        self.v.append(v)
        self.r.append(float(r))
        self.done.append(1.0 if done else 0.0)

    def build(self, gamma: float, lam: float, last_v: float) -> Batch:
        x_img = torch.stack(self.x_img).to(self.device)
        x_meta = torch.stack(self.x_meta).to(self.device)
        maskA = torch.stack(self.maskA).to(self.device)
        if maskA.dtype != torch.bool:
            maskA = maskA.to(torch.bool)

        a = torch.stack(self.a).to(self.device).long().view(-1)
        logp = torch.stack(self.logp).to(self.device).view(-1)
        v = torch.stack(self.v).to(self.device).view(-1)

        r = torch.tensor(self.r, dtype=torch.float32, device=self.device)
        done = torch.tensor(self.done, dtype=torch.float32, device=self.device)

        N = r.shape[0]
        adv = torch.zeros(N, dtype=torch.float32, device=self.device)

        gae = 0.0
        for t in reversed(range(N)):
            nonterminal = 1.0 - done[t]
            next_value = float(last_v) if t == N - 1 else float(v[t + 1].item())
            delta = r[t] + gamma * next_value * nonterminal - v[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae

        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return Batch(x_img, x_meta, maskA, a, logp, v, r, done, adv, ret)
