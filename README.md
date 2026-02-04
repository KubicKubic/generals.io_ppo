# Generals RL Refactor

## Run
```bash
python train.py --config train_config.yaml
```

## Extension points (few key interfaces)
- Obs -> tensor encoding: `generals_rl/data/encoding.py::encode_obs_sequence`
- Model architecture: `generals_rl/models/policy_rope_factorized.py::SeqPPOPolicyRoPEFactorized`
  - override `global_features()` to change backbone
  - override `dir_mode_logits()` / `split_masks()` to change factorization & legality
- Visualization: `generals_rl/viz/rollout_viz.py`
- Reward shaping: `generals_rl/train/reward.py`


## Switching models via config
Edit `train_config.yaml`:

```yaml
model:
  name: rope_factorized        # or spatial_factorized
  rope: {}
  spatial:
    d: 128
    meta_proj: 16
```
