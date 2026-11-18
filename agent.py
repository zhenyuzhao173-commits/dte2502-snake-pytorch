"""PyTorch Deep Q-Learning agent for the DTE-2502 graded assignment.

This module is a PyTorch reimplementation of the DeepQLearningAgent from the
DragonWarrior15/snake-rl reference project. The environment and replay-buffer
interfaces are intentionally kept compatible with the original project.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import RMSprop

from replay_buffer import ReplayBufferNumpy


class Agent:
    """Common environment/buffer functionality used by the DQN agent."""

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True, version=""):
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (board_size, board_size, frames)
        self._version = version
        self.reset_buffer()
        self._board_grid = np.arange(0, board_size ** 2).reshape(board_size, -1)

    def get_gamma(self):
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        if buffer_size is not None:
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(
            self._buffer_size, self._board_size, self._n_frames, self._n_actions
        )

    def get_buffer_size(self):
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        self._buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

    def save_buffer(self, file_path="", iteration=None):
        iteration = 0 if iteration is None else int(iteration)
        Path(file_path).mkdir(parents=True, exist_ok=True)
        with open(Path(file_path) / f"buffer_{iteration:04d}", "wb") as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path="", iteration=None):
        iteration = 0 if iteration is None else int(iteration)
        with open(Path(file_path) / f"buffer_{iteration:04d}", "rb") as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        return point // self._board_size, point % self._board_size

    def _row_col_to_point(self, row, col):
        return row * self._board_size + col


class DQN(nn.Module):
    """CNN architecture matching model_config/v17.1.json.

    Keras used channels-last tensors. PyTorch uses channels-first tensors, so
    the public agent converts (N,H,W,C) -> (N,C,H,W) before calling this model.
    """

    def __init__(self, frames=2, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(frames, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.action_values = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # Keras Flatten on NHWC orders values as H, W, C. Convert the
        # PyTorch NCHW tensor back to NHWC before flattening so the imported
        # TensorFlow dense-layer weights preserve their original semantics.
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.action_values(x)


class DeepQLearningAgent(Agent):
    """Deep Q-Learning with replay memory, CNN function approximation and a target net."""

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True, version="",
                 device=None):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions,
                         use_target_net, version)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.reset_models()

    def reset_models(self):
        self._model = self._agent_model().to(self.device)
        if self._use_target_net:
            self._target_net = self._agent_model().to(self.device)
            self.update_target_net()
        self._optimizer = RMSprop(
            self._model.parameters(), lr=5e-4, alpha=0.9, eps=1e-7
        )

    def _agent_model(self):
        # v17.1 is the required model version. The architecture is explicitly
        # defined here so the final submission has no TensorFlow dependency.
        config_path = Path(__file__).resolve().parent / "model_config" / f"{self._version}.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg["board_size"] != self._board_size or cfg["frames"] != self._n_frames:
            raise ValueError("Model configuration does not match agent dimensions")
        if cfg["n_actions"] != self._n_actions:
            raise ValueError("Model configuration does not match action count")
        return DQN(frames=self._n_frames, n_actions=self._n_actions)

    def _normalize_board(self, board):
        return board.astype(np.float32) / 4.0

    def _prepare_input(self, board):
        board = np.asarray(board)
        if board.ndim == 3:
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        # Environment/reference project: N,H,W,C. PyTorch: N,C,H,W.
        return np.transpose(board, (0, 3, 1, 2)).copy()

    @torch.no_grad()
    def _get_model_outputs(self, board, model=None):
        model = self._model if model is None else model
        x = torch.from_numpy(self._prepare_input(board)).to(self.device)
        was_training = model.training
        model.eval()
        outputs = model(x).detach().cpu().numpy()
        if was_training:
            model.train()
        return outputs

    def move(self, board, legal_moves, value=None):
        q_values = self._get_model_outputs(board)
        masked = np.where(legal_moves == 1, q_values, -np.inf)
        return np.argmax(masked, axis=1)

    def get_action_proba(self, board, values=None):
        q_values = np.clip(self._get_model_outputs(board), -10, 10)
        q_values -= q_values.max(axis=1, keepdims=True)
        probs = np.exp(q_values)
        return probs / probs.sum(axis=1, keepdims=True)

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Perform one DQN replay update and return the mean Huber loss."""
        if self.get_buffer_size() == 0:
            return 0.0
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:
            r = np.sign(r)

        next_model = self._target_net if self._use_target_net else self._model
        with torch.no_grad():
            next_q = self._get_model_outputs(next_s, next_model)
            masked_next_q = np.where(legal_moves == 1, next_q, -np.inf)
            max_next_q = np.max(masked_next_q, axis=1, keepdims=True)
            discounted_reward = r.astype(np.float32) + self._gamma * max_next_q * (1 - done)

        x = torch.from_numpy(self._prepare_input(s)).to(self.device)
        target = torch.from_numpy(self._get_model_outputs(s)).to(self.device)
        actions = torch.from_numpy(np.argmax(a, axis=1).astype(np.int64)).to(self.device)
        rewards = torch.from_numpy(discounted_reward.astype(np.float32)).to(self.device).squeeze(1)

        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        q_values = self._model(x)
        selected_q = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
        # The reference implementation keeps all non-selected Q-values equal
        # to the current network output and replaces only the selected action.
        # That produces the same objective as the target tensor construction.
        target_selected = rewards
        target_full = target.clone()
        target_full.scatter_(1, actions.view(-1, 1), target_selected.view(-1, 1))
        loss = torch.nn.functional.smooth_l1_loss(q_values, target_full, beta=1.0, reduction="mean")
        loss.backward()
        self._optimizer.step()
        return float(loss.detach().cpu().item())

    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        if not self._use_target_net:
            print("Target network disabled")
            return
        for (name, a), (_, b) in zip(self._model.state_dict().items(), self._target_net.state_dict().items()):
            print(f"{name}: {bool(torch.equal(a, b))}")

    def print_models(self):
        print("Training model:\n", self._model)
        if self._use_target_net:
            print("Target network:\n", self._target_net)

    def save_model(self, file_path="", iteration=None):
        iteration = 0 if iteration is None else int(iteration)
        path = Path(file_path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self._model.state_dict(),
            "target_state_dict": self._target_net.state_dict() if self._use_target_net else None,
            "optimizer_state_dict": self._optimizer.state_dict(),
            "board_size": self._board_size,
            "frames": self._n_frames,
            "n_actions": self._n_actions,
            "gamma": self._gamma,
            "version": self._version,
        }
        torch.save(payload, path / f"model_{iteration:04d}.pth")

    def load_model(self, file_path="", iteration=None, map_location=None):
        iteration = 0 if iteration is None else int(iteration)
        checkpoint_path = Path(file_path) / f"model_{iteration:04d}.pth"
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=map_location or self.device,
                weights_only=False,
            )
        except TypeError:  # compatibility with older PyTorch releases
            payload = torch.load(checkpoint_path, map_location=map_location or self.device)
        self._model.load_state_dict(payload["model_state_dict"])
        if self._use_target_net and payload.get("target_state_dict") is not None:
            self._target_net.load_state_dict(payload["target_state_dict"])
        if payload.get("optimizer_state_dict") is not None:
            self._optimizer.load_state_dict(payload["optimizer_state_dict"])


def mean_huber_loss(y_true, y_pred, delta=1.0):
    """Compatibility helper matching the original custom Huber loss."""
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = 0.5 * error.pow(2)
    linear = delta * (abs_error - 0.5 * delta)
    return torch.where(abs_error < delta, quadratic, linear).mean()
