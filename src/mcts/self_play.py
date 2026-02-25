"""
Self-Play Training Pipeline
============================
Owner: Iris

Drives the AlphaZero training cycle:
    1. Self-play → generate (state, mcts_policy, outcome) tuples
    2. Train network on collected data
    3. Evaluate new model vs previous best
    4. Repeat

Implementation lives in feature/mcts-engine branch.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque


@dataclass
class TrainingSample:
    """Single training example for the neural network."""
    state: np.ndarray           # observation tensor (e.g., 5x5x10)
    policy_target: np.ndarray   # MCTS visit count distribution
    value_target: float         # +1 or -1 from this player's perspective


class ReplayBuffer:
    """
    Fixed-size FIFO buffer of training samples.
    For 5x5 POC, 10k-50k samples is sufficient.
    """

    def __init__(self, max_size: int = 50_000):
        self.buffer: deque = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, samples: List[TrainingSample]):
        # TODO: Implement in feature/mcts-engine
        raise NotImplementedError(
            "ReplayBuffer logic lives in feature/mcts-engine")

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: Implement in feature/mcts-engine
        raise NotImplementedError(
            "ReplayBuffer logic lives in feature/mcts-engine")


# TODO: Implement play_one_game() and training_loop() in feature/mcts-engine
