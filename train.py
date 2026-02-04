from __future__ import annotations
import argparse
from generals_rl.train.config import load_config
from generals_rl.train.loop import train

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg)

if __name__ == "__main__":
    main()
