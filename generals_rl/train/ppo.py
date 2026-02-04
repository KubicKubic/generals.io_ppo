from __future__ import annotations
import torch
import torch.nn as nn
from .buffer import Batch
from .config import PPOConfig

def ppo_update(policy, optimizer, batch: Batch, cfg: PPOConfig, device):
    N = batch.a.shape[0]
    idx = torch.arange(N, device=device)
    for _ in range(int(cfg.epochs)):
        perm = idx[torch.randperm(N)]
        for start in range(0, N, int(cfg.minibatch)):
            j = perm[start:start + int(cfg.minibatch)]
            logp_new, ent, v_new = policy.evaluate_actions(batch.x_img[j], batch.x_meta[j], batch.maskA[j], batch.a[j])
            ratio = torch.exp(logp_new - batch.logp[j])
            surr1 = ratio * batch.adv[j]
            surr2 = torch.clamp(ratio, 1.0 - float(cfg.clip_eps), 1.0 + float(cfg.clip_eps)) * batch.adv[j]
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = ((v_new - batch.ret[j]) ** 2).mean()
            ent_loss = -ent.mean()
            loss = pi_loss + float(cfg.vf_coef) * v_loss + float(cfg.ent_coef) * ent_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm))
            optimizer.step()
