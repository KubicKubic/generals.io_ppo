# generals_rl/train/ppo.py
from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from .buffer import Batch
from .config import PPOConfig
from .d4_aug import d4_augment_minibatch


def ppo_update(policy, optimizer, batch: Batch, cfg: PPOConfig, device):
    N = batch.a.shape[0]
    idx = torch.arange(N, device=device)

    # 开关：想关掉增强就改 False
    use_d4_aug = True

    for _ in range(int(cfg.epochs)):
        perm = idx[torch.randperm(N)]
        for start in tqdm(range(0, N, int(cfg.minibatch)), desc="ppo optimize", dynamic_ncols=True):
            j = perm[start:start + int(cfg.minibatch)]

            x_img_j = batch.x_img[j]
            x_meta_j = batch.x_meta[j]
            maskA_j = batch.maskA[j]
            a_j = batch.a[j]

            # 这些 target/old policy 量固定不变（对应原始样本 j）
            logp_old_j = batch.logp[j]
            adv_j = batch.adv[j]
            ret_j = batch.ret[j]

            if use_d4_aug:
                # 不拼接：枚举 8 种 D4 变换，依次 forward/backward，最后 step 一次（梯度累积）
                optimizer.zero_grad(set_to_none=True)

                for aug_id in range(8):
                    x_img_aug, maskA_aug, a_aug = d4_augment_minibatch(
                        x_img=x_img_j,
                        maskA=maskA_j,
                        a=a_j,
                        aug_id=aug_id,
                    )

                    # 注意：这里 x_meta 默认不随 D4 变换；如果你的 x_meta 含坐标/朝向等，也需要一起变换
                    logp_new, ent, v_new = policy.evaluate_actions(
                        x_img_aug, x_meta_j, maskA_aug, a_aug
                    )

                    ratio = torch.exp(logp_new - logp_old_j)
                    surr1 = ratio * adv_j
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - float(cfg.clip_eps),
                        1.0 + float(cfg.clip_eps),
                    ) * adv_j

                    pi_loss = -torch.min(surr1, surr2).mean()
                    v_loss = ((v_new - ret_j) ** 2).mean()
                    ent_loss = -ent.mean()
                    loss = pi_loss + float(cfg.vf_coef) * v_loss + float(cfg.ent_coef) * ent_loss

                    # 8 次累积保持整体梯度尺度不变
                    (loss / 8.0).backward()

                nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm))
                optimizer.step()

            else:
                logp_new, ent, v_new = policy.evaluate_actions(x_img_j, x_meta_j, maskA_j, a_j)

                ratio = torch.exp(logp_new - logp_old_j)
                surr1 = ratio * adv_j
                surr2 = torch.clamp(
                    ratio,
                    1.0 - float(cfg.clip_eps),
                    1.0 + float(cfg.clip_eps),
                ) * adv_j

                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((v_new - ret_j) ** 2).mean()
                ent_loss = -ent.mean()
                loss = pi_loss + float(cfg.vf_coef) * v_loss + float(cfg.ent_coef) * ent_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm))
                optimizer.step()
