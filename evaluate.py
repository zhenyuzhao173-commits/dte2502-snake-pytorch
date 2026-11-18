"""Evaluate a PyTorch DQN checkpoint on the v17.1 Snake environment."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from agent import DeepQLearningAgent
from game_environment import SnakeNumpy
from utils import play_game2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=int, help="checkpoint number, e.g. 400")
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = Path(__file__).resolve().parent
    with open(root / "model_config" / "v17.1.json", encoding="utf-8") as f:
        cfg = json.load(f)
    agent = DeepQLearningAgent(
        board_size=cfg["board_size"], frames=cfg["frames"], n_actions=cfg["n_actions"],
        buffer_size=10, gamma=0.99, use_target_net=True, version="v17.1", device=args.device,
    )
    agent.load_model(root / "models" / "v17.1", args.checkpoint)
    env = SnakeNumpy(
        board_size=cfg["board_size"], frames=cfg["frames"], games=args.games,
        max_time_limit=cfg["max_time_limit"], frame_mode=True,
        obstacles=bool(cfg["obstacles"]), version="v17.1",
    )
    rewards, lengths, games = play_game2(
        env, agent, cfg["n_actions"], n_games=args.games, epsilon=-1,
        record=False, frame_mode=True, total_games=args.games,
    )
    n = max(int(games), 1)
    print(f"checkpoint={args.checkpoint:04d}")
    print(f"games={int(games)}")
    print(f"mean_reward={float(rewards) / n:.3f}")
    print(f"mean_length={float(lengths) / n:.3f}")

if __name__ == "__main__":
    main()
