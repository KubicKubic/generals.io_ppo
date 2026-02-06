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


def _left_pad_repeat_first(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Left-pad along time dim (dim=1) by repeating the first frame.

    x: (B, t, ...)
    returns: (B, T, ...)
    """
    t = int(x.shape[1])
    if t == T:
        return x
    if t > T:
        return x[:, -T:]
    pad_len = T - t
    pad = x[:, :1].expand(x.shape[0], pad_len, *x.shape[2:]).contiguous()
    return torch.cat([pad, x], dim=1)


class RolloutBuffer:
    """
    Vectorized rollout buffer:
      - store per-step batched tensors (E, ...)
      - compute GAE per-env along rollout axis (L, E)
      - flatten to (N=L*E, ...) for PPO
    """

    def __init__(self, device: torch.device):
        self.device = device

        self.x_img_steps: List[torch.Tensor] = []   # (E, Tt, C, H, W)   (Tt may vary)
        self.x_meta_steps: List[torch.Tensor] = []  # (E, Tt, M)         (Tt may vary)
        self.maskA_steps: List[torch.Tensor] = []   # (E, A)
        self.a_steps: List[torch.Tensor] = []       # (E,)
        self.logp_steps: List[torch.Tensor] = []    # (E,)
        self.v_steps: List[torch.Tensor] = []       # (E,)
        self.r_steps: List[torch.Tensor] = []       # (E,)
        self.done_steps: List[torch.Tensor] = []    # (E,)

    @property
    def num_steps(self) -> int:
        return len(self.r_steps)

    @property
    def num_envs(self) -> int:
        return int(self.r_steps[0].shape[0]) if self.r_steps else 0

    def add_step(
        self,
        *,
        x_img_seq: torch.Tensor,   # (E,T,C,H,W)
        x_meta_seq: torch.Tensor,  # (E,T,M)
        maskA: torch.Tensor,       # (E,A)
        a: torch.Tensor,           # (E,)
        logp: torch.Tensor,        # (E,)
        v: torch.Tensor,           # (E,)
        r: torch.Tensor,           # (E,)
        done: torch.Tensor,        # (E,) in {0,1}
    ) -> None:
        # store on CPU to save GPU memory
        self.x_img_steps.append(x_img_seq.detach().cpu())
        self.x_meta_steps.append(x_meta_seq.detach().cpu())
        self.maskA_steps.append(maskA.detach().cpu().to(torch.bool))
        self.a_steps.append(a.detach().cpu().long().view(-1))
        self.logp_steps.append(logp.detach().cpu().view(-1))
        self.v_steps.append(v.detach().cpu().view(-1))
        self.r_steps.append(r.detach().cpu().float().view(-1))
        self.done_steps.append(done.detach().cpu().float().view(-1))

    def build(self, *, gamma: float, lam: float, last_v: torch.Tensor | float) -> Batch:
        if self.num_steps == 0:
            raise RuntimeError("RolloutBuffer is empty")

        L = self.num_steps
        E = self.num_envs

        # pad variable T across steps to max_T so we can stack
        Ts = [int(x.shape[1]) for x in self.x_img_steps]
        T_max = max(Ts)

        x_img_padded = [_left_pad_repeat_first(x, T_max) for x in self.x_img_steps]   # each (E,T_max,C,H,W)
        x_meta_padded = [_left_pad_repeat_first(x, T_max) for x in self.x_meta_steps] # each (E,T_max,M)

        x_img = torch.stack(x_img_padded, dim=0).to(self.device)     # (L,E,T,C,H,W)
        x_meta = torch.stack(x_meta_padded, dim=0).to(self.device)   # (L,E,T,M)
        maskA = torch.stack(self.maskA_steps, dim=0).to(self.device) # (L,E,A)
        a = torch.stack(self.a_steps, dim=0).to(self.device).long()  # (L,E)
        logp = torch.stack(self.logp_steps, dim=0).to(self.device)   # (L,E)
        v = torch.stack(self.v_steps, dim=0).to(self.device)         # (L,E)
        r = torch.stack(self.r_steps, dim=0).to(self.device)         # (L,E)
        done = torch.stack(self.done_steps, dim=0).to(self.device)   # (L,E)

        if not torch.is_tensor(last_v):
            last_v = torch.tensor([float(last_v)] * E, dtype=torch.float32, device=self.device)
        else:
            last_v = last_v.to(self.device).float().view(E)

        # GAE: vectorized over envs
        adv = torch.zeros((L, E), dtype=torch.float32, device=self.device)
        gae = torch.zeros((E,), dtype=torch.float32, device=self.device)

        for t in reversed(range(L)):
            nonterminal = 1.0 - done[t]
            next_value = last_v if t == L - 1 else v[t + 1]
            delta = r[t] + float(gamma) * next_value * nonterminal - v[t]
            gae = delta + float(gamma) * float(lam) * nonterminal * gae
            adv[t] = gae

        ret = adv + v

        # flatten (L,E,...) -> (N=L*E,...)
        x_img_f = x_img.reshape(L * E, T_max, *x_img.shape[3:])      # (N,T,C,H,W)
        x_meta_f = x_meta.reshape(L * E, T_max, x_meta.shape[-1])    # (N,T,M)
        maskA_f = maskA.reshape(L * E, maskA.shape[-1])              # (N,A)

        a_f = a.reshape(L * E)
        logp_f = logp.reshape(L * E)
        v_f = v.reshape(L * E)
        r_f = r.reshape(L * E)
        done_f = done.reshape(L * E)
        adv_f = adv.reshape(L * E)
        ret_f = ret.reshape(L * E)

        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        return Batch(
            x_img=x_img_f,
            x_meta=x_meta_f,
            maskA=maskA_f,
            a=a_f.long(),
            logp=logp_f,
            v=v_f,
            r=r_f,
            done=done_f,
            adv=adv_f,
            ret=ret_f,
        )
