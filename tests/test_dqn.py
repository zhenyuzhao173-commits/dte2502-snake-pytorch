"""Small, dependency-light smoke tests for the PyTorch DQN port."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import DQN, DeepQLearningAgent


class DQNPortTests(unittest.TestCase):
    def test_cnn_output_shape(self):
        model = DQN(frames=2, n_actions=4)
        output = model(torch.zeros(3, 2, 10, 10))
        self.assertEqual(tuple(output.shape), (3, 4))

    def test_environment_input_layout(self):
        agent = DeepQLearningAgent(
            board_size=10, frames=2, buffer_size=8, gamma=0.99,
            n_actions=4, use_target_net=True, version="v17.1", device="cpu",
        )
        prepared = agent._prepare_input(np.zeros((5, 10, 10, 2), dtype=np.uint8))
        self.assertEqual(tuple(prepared.shape), (5, 2, 10, 10))

    def test_final_checkpoint_loads_and_predicts(self):
        agent = DeepQLearningAgent(
            board_size=10, frames=2, buffer_size=8, gamma=0.99,
            n_actions=4, use_target_net=True, version="v17.1", device="cpu",
        )
        agent.load_model(ROOT / "models" / "v17.1", 400)
        q_values = agent._get_model_outputs(np.zeros((1, 10, 10, 2), dtype=np.uint8))
        self.assertEqual(tuple(q_values.shape), (1, 4))
        self.assertTrue(np.isfinite(q_values).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
