"""
Environment Interface Contract
===============================
Defines the EXACT interface that MCTS expects from the game environment.
Reef: implement QuoridorEnvInterface in quoridor_env.py.
Iris: code against this interface.

Also includes MinimalQuoridorStub for testing MCTS in isolation
before the real engine is ready.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Any
import numpy as np
import copy


class QuoridorEnvInterface(ABC):
    """
    Contract between Game Engine (Reef) and MCTS/Training (Iris).

    Reef: implement every method below in quoridor_env.py.
    Iris: import this interface, code against it.
    Integration = Reef's class passes the test suite.
    """

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
        """Deep copy of the game state. MCTS needs this — don't skip it."""
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
        For 5x5: returns shape (5, 5, 10)
        For 9x9: returns shape (9, 9, 10)
        All values normalized to [0, 1].
        """
        ...


# =========================================================================
#  STUB ENVIRONMENT — for testing MCTS before real engine exists
#  Delete once Reef's QuoridorEnv is ready.
# =========================================================================

@dataclass
class StubState:
    """Minimal state: two pawns racing on a 5x5 grid, no walls."""
    board_size: int = 5
    positions: list = None       # [player0_row, player1_row]
    current_player: int = 0
    done: bool = False
    winner: Optional[int] = None

    def __post_init__(self):
        if self.positions is None:
            self.positions = [self.board_size - 1, 0]


class MinimalQuoridorStub(QuoridorEnvInterface):
    """
    Tiny pawn-race on 5x5. No walls.
    Actions: 0=up, 1=down
    Player 0 wins by reaching row 0. Player 1 wins by reaching row 4.

    NOT the real game. Exists solely to validate MCTS search logic.
    """

    ACTIONS = {0: -1, 1: 1}
    BOARD_SIZE = 5

    @property
    def action_space_size(self) -> int:
        return 2

    def reset(self) -> StubState:
        return StubState(board_size=self.BOARD_SIZE)

    def clone_state(self, state: StubState) -> StubState:
        return copy.deepcopy(state)

    def get_current_player(self, state: StubState) -> int:
        return state.current_player

    def get_valid_actions(self, state: StubState) -> np.ndarray:
        if state.done:
            return np.array([], dtype=np.int64)

        valid = []
        player = state.current_player
        row = state.positions[player]

        for action, dr in self.ACTIONS.items():
            nr = row + dr
            if 0 <= nr < self.BOARD_SIZE:
                valid.append(action)

        return np.array(valid, dtype=np.int64)

    def step(
        self, state: StubState, action: int
    ) -> Tuple[StubState, float, bool, dict]:
        state = copy.deepcopy(state)  # never mutate the caller's state
        player = state.current_player
        dr = self.ACTIONS[action]
        state.positions[player] += dr

        goal_row = 0 if player == 0 else self.BOARD_SIZE - 1
        if state.positions[player] == goal_row:
            state.done = True
            state.winner = player
            return state, 1.0, True, {"winner": player}

        state.current_player = 1 - player
        return state, 0.0, False, {"winner": None}

    def state_to_tensor(self, state: StubState) -> np.ndarray:
        """Dummy tensor for stub — not used in Phase 1."""
        tensor = np.zeros(
            (self.BOARD_SIZE, self.BOARD_SIZE, 10), dtype=np.float32)
        for i, pos in enumerate(state.positions):
            tensor[pos, self.BOARD_SIZE // 2, i] = 1.0
        return tensor
