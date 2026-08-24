"""
Environment Interface Contract
===============================
Defines the EXACT interface that MCTS expects from the game environment.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np


class QuoridorEnvInterface(ABC):
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Total number of possible actions (pawn moves + wall placements)."""
        ...

    @abstractmethod
    def get_valid_actions(self, state) -> np.ndarray:
        """Return array of valid action indices for the current player."""
        ...

    @abstractmethod
    def step(self, state, action: int) -> Tuple[Any, float, bool, dict]:
        """
        Apply action to state.

        Returns:
            next_state: updated game state
            reward: +1 win, -1 loss, 0 ongoing
            done: True if game is over
            info: dict with at minimum {"winner": int or None}
        """
        ...

    @abstractmethod
    def clone_state(self, state) -> Any:
        """Deep copy of the game state. MCTS needs this - don't skip it."""
        ...

    @abstractmethod
    def get_current_player(self, state) -> int:
        """Return 0 or 1 indicating whose turn it is."""
        ...

    @abstractmethod
    def reset(self) -> Any:
        """Return fresh initial game state."""
        ...

    @abstractmethod
    def state_to_tensor(self, state) -> np.ndarray:
        """
        Convert game state to the observation tensor for the neural network.

        Shape: (board_size, board_size, OBS_CHANNELS) - HWC format.
            5×5 board: (5, 5, 11)
            9×9 board: (9, 9, 11)

        All values normalized to [0, 1].
        See tensor_spec.py for channel breakdown and reference implementation.
        """
        ...
