"""Train the PyTorch DeepQLearningAgent on the v17.1 Snake environment."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agent import DeepQLearningAgent
from game_environment import SnakeNumpy
from utils import play_game2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v17.1")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--log-frequency", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init", default=None, help="Optional existing PyTorch checkpoint directory")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.97)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    with open(root / "model_config" / f"{args.version}.json", encoding="utf-8") as f:
        cfg = json.load(f)

    agent = DeepQLearningAgent(
        board_size=cfg["board_size"], frames=cfg["frames"],
        n_actions=cfg["n_actions"], buffer_size=cfg["buffer_size"],
        gamma=0.99, use_target_net=True, version=args.version, device=args.device,
    )
    if args.init:
        agent.load_model(args.init, 0)

    env = SnakeNumpy(
        board_size=cfg["board_size"], frames=cfg["frames"], games=args.games,
        max_time_limit=cfg["max_time_limit"], frame_mode=True,
        obstacles=bool(cfg["obstacles"]), version=args.version,
    )
    eval_env = SnakeNumpy(
        board_size=cfg["board_size"], frames=cfg["frames"], games=args.eval_games,
        max_time_limit=cfg["max_time_limit"], frame_mode=True,
        obstacles=bool(cfg["obstacles"]), version=args.version,
    )

    epsilon, epsilon_end, decay = args.epsilon_start, args.epsilon_end, args.epsilon_decay
    log = {"iteration": [], "reward_mean": [], "length_mean": [], "games": [], "loss": [], "epsilon": []}
    out_dir = root / "models" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "results"
    logs_dir.mkdir(exist_ok=True)

    start = time.time()
    for iteration in tqdm(range(1, args.episodes + 1), desc="training"):
        play_game2(
            env, agent, cfg["n_actions"], n_games=args.games, epsilon=epsilon,
            record=True, frame_mode=True, total_frames=args.games, stateful=True,
        )
        loss = agent.train_agent(batch_size=64, num_games=args.games, reward_clip=True)

        if iteration % args.log_frequency == 0:
            rewards, lengths, games = play_game2(
                eval_env, agent, cfg["n_actions"], n_games=args.eval_games,
                epsilon=-1, record=False, frame_mode=True, total_games=args.eval_games,
            )
            reward_mean = float(rewards / max(games, 1))
            length_mean = float(lengths / max(games, 1))
            log["iteration"].append(iteration)
            log["reward_mean"].append(round(reward_mean, 3))
            log["length_mean"].append(round(length_mean, 3))
            log["games"].append(int(games))
            log["loss"].append(float(loss))
            log["epsilon"].append(float(epsilon))
            pd.DataFrame(log).to_csv(logs_dir / f"training_{args.version}.csv", index=False)
            agent.update_target_net()
            agent.save_model(out_dir, iteration)

        epsilon = max(epsilon * decay, epsilon_end)

    # Always save a final checkpoint, even when the episode count is not a
    # multiple of log_frequency.
    agent.update_target_net()
    agent.save_model(out_dir, args.episodes)
    print(f"Training finished in {(time.time() - start) / 60:.2f} minutes")
    print(f"Saved final checkpoint: {out_dir / f'model_{args.episodes:04d}.pth'}")


if __name__ == "__main__":
    main()
