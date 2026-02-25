"""
Monte Carlo Tree Search (AlphaZero-style)
==========================================
Owner: Iris

Phase 1: Random rollout evaluation (no neural network).
Phase 2: Swap in NN via the `evaluate_fn` parameter.

Implementation lives in feature/mcts-engine branch.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Tuple


@dataclass
class MCTSConfig:
    """All MCTS hyperparameters in one place."""
    num_simulations: int = 400
    c_puct: float = 1.41
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_rollout_depth: int = 100


# Type alias: state -> (policy array, value float)
EvaluateFn = Callable[[object], Tuple[np.ndarray, float]]


class MCTS:
    """
    AlphaZero-style MCTS.

    `env` must implement QuoridorEnvInterface (see src/env/env_interface.py):
        - env.get_valid_actions(state) -> np.ndarray
        - env.step(state, action) -> (next_state, reward, done, info)
        - env.clone_state(state) -> deep copy
        - env.action_space_size -> int
        - env.get_current_player(state) -> int (0 or 1)

    TODO: Implement in feature/mcts-engine
    """

    def __init__(
        self,
        config: MCTSConfig = None,
        evaluate_fn: Optional[EvaluateFn] = None,
    ):
        self.config = config or MCTSConfig()
        self.evaluate_fn = evaluate_fn

    def search(self, env, state) -> np.ndarray:
        """
        Run MCTS from the given state.

        Returns:
            np.ndarray of shape [action_space_size] — normalized visit count
            distribution. Use as training target for the policy head.
        """
        # TODO: Implement in feature/mcts-engine
        raise NotImplementedError("MCTS logic lives in feature/mcts-engine")
