"""Generals RL training code.

Exposed interfaces:
  - generals_rl.models.make_policy
  - generals_rl.models.SeqPPOPolicyRoPEFactorized
  - generals_rl.models.SeqPPOPolicySpatialFactorized
  - generals_rl.data.encode_obs_sequence
  - generals_rl.data.ObsHistory
  - generals_rl.train.train(cfg)
  - generals_rl.train.load_config(path)
"""
from .data import ObsHistory, encode_obs_sequence
from .models import make_policy, SeqPPOPolicyRoPEFactorized, SeqPPOPolicySpatialFactorized
from .train import TrainConfig, load_config, save_resolved_config, train
