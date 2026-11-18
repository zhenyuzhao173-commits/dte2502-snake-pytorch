"""Create a GIF and PNG showing the final PyTorch DQN policy playing Snake."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from agent import DeepQLearningAgent
from game_environment import Snake


def run_episode(agent, env, max_steps):
    state = env.reset()
    frames, rewards = [], []
    for _ in range(max_steps):
        legal = env.get_legal_moves()
        action = int(agent.move(state, legal)[0])
        next_state, reward, done, info, _ = env.step(action)
        frames.append(next_state[:, :, -1].copy())
        rewards.append(int(reward))
        state = next_state
        if done:
            break
    return frames, rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=int)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--attempts", type=int, default=12)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    with open(root / "model_config/v17.1.json", encoding="utf-8") as f:
        cfg = json.load(f)
    agent = DeepQLearningAgent(
        board_size=cfg["board_size"], frames=cfg["frames"], n_actions=cfg["n_actions"],
        buffer_size=10, gamma=0.99, use_target_net=True, version="v17.1", device=args.device,
    )
    agent.load_model(root / "models" / "v17.1", args.checkpoint)
    max_steps = args.steps if args.steps is not None else cfg["max_time_limit"]
    np.random.seed(42)
    env = Snake(board_size=cfg["board_size"], frames=cfg["frames"],
                max_time_limit=max_steps, obstacles=bool(cfg["obstacles"]), version="v17.1")

    best_frames, best_rewards = [], []
    for _ in range(args.attempts):
        frames, rewards = run_episode(agent, env, max_steps)
        if sum(rewards) > sum(best_rewards) or not best_frames:
            best_frames, best_rewards = frames, rewards

    out = root / "results"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(best_frames[0], interpolation="nearest", vmin=0, vmax=4)
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        im.set_data(best_frames[i])
        ax.set_title(f"PyTorch DQN policy — step {i + 1}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(best_frames), interval=100, blit=True)
    anim.save(out / "final_policy.gif", writer=PillowWriter(fps=10))
    fig.savefig(out / "final_policy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / 'final_policy.gif'}")
    print(f"Saved {out / 'final_policy.png'}")
    print(f"best_reward={sum(best_rewards)} steps={len(best_frames)}")


if __name__ == "__main__":
    main()
