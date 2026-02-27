"""
Monte Carlo Tree Search (AlphaZero-style)
==========================================
Owner: Iris

Phase 1: Random rollout evaluation (no neural network).
Phase 2: Swap in NN via the `evaluate_fn` parameter.

Usage:
    from src.mcts.mcts import MCTS, MCTSConfig

    # Phase 1 — random rollouts
    mcts = MCTS(config=MCTSConfig(num_simulations=400))
    action_probs = mcts.search(env, state)

    # Phase 2 — with neural network
    def nn_evaluate(state) -> Tuple[np.ndarray, float]:
        tensor = env.state_to_tensor(state)
        policy, value = model(tensor)
        return policy.numpy(), value.item()

    mcts = MCTS(config=MCTSConfig(num_simulations=800), evaluate_fn=nn_evaluate)
    action_probs = mcts.search(env, state)
"""

import math
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


class Node:
    """A single node in the MCTS tree."""

    __slots__ = [
        "state", "parent", "action", "prior",
        "visit_count", "value_sum", "children", "is_expanded",
        "valid_actions",
    ]

    def __init__(
        self,
        state,
        parent: Optional["Node"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, "Node"] = {}
        self.is_expanded: bool = False
        self.valid_actions: Optional[np.ndarray] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """
        PUCT score from the PARENT's perspective.

        Child stores Q from its own current player's view, so we negate it
        because the parent selects the action that is worst for the opponent
        (i.e., best for itself).
        """
        exploitation = -self.q_value
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return exploitation + exploration


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
    """

    def __init__(
        self,
        config: MCTSConfig = MCTSConfig(),
        evaluate_fn: Optional[EvaluateFn] = None,
    ):
        self.config = config
        self.evaluate_fn = evaluate_fn  # None => random rollout (Phase 1)

    def search(self, env, state, temperature: Optional[float] = None) -> np.ndarray:
        """
        Run MCTS from the given state.

        Args:
            temperature: Override config temperature for this search.
                         Pass explicitly instead of mutating config.

        Returns:
            np.ndarray of shape [action_space_size] — normalized visit count
            distribution. Use as training target for the policy head.
        """
        root = Node(state=env.clone_state(state))
        self._expand(root, env)
        self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            scratch_state = env.clone_state(state)
            done = False

            # === 1. SELECT ===
            while node.is_expanded and node.children:
                node = self._select_child(node)
                scratch_state, reward, done, info = env.step(
                    scratch_state, node.action
                )
                if done:
                    break

            # === 2. EXPAND & EVALUATE ===
            if done:
                # Terminal node reached after env.step().
                # Convention: env.step() returns reward=+1 for the player
                # who just MOVED (the parent's action led here). The
                # current node's perspective is that of the NEXT player
                # to move — i.e., the opponent of whoever just won.
                # Therefore: winner exists → current node lost → -1.
                #            Draw / no winner → 0.
                winner = info.get("winner", None)
                value = -1.0 if winner is not None else 0.0
            else:
                value = self._expand(node, env, scratch_state)

            # === 3. BACKPROPAGATE ===
            self._backpropagate(node, value)

        temp = temperature if temperature is not None else self.config.temperature
        return self._action_probabilities(root, env.action_space_size, temp)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _select_child(self, node: Node) -> Node:
        return max(
            node.children.values(),
            key=lambda c: c.ucb_score(self.config.c_puct),
        )

    def _expand(self, node: Node, env, state=None) -> float:
        """Expand leaf. Returns value estimate for current player."""
        if state is None:
            state = node.state

        valid_actions = env.get_valid_actions(state)
        node.valid_actions = valid_actions

        if len(valid_actions) == 0:
            node.is_expanded = True
            return 0.0

        if self.evaluate_fn is not None:
            # --- Phase 2: Neural Network ---
            policy, value = self.evaluate_fn(state)

            mask = np.zeros(env.action_space_size, dtype=np.float32)
            mask[valid_actions] = 1.0
            policy = policy * mask
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy = mask / mask.sum()

            for action in valid_actions:
                node.children[action] = Node(
                    state=None,  # reconstructed via scratch_state in search()
                    parent=node,
                    action=action,
                    prior=policy[action],
                )
        else:
            # --- Phase 1: Uniform prior + random rollout ---
            uniform_prior = 1.0 / len(valid_actions)
            for action in valid_actions:
                node.children[action] = Node(
                    state=None,  # reconstructed via scratch_state in search()
                    parent=node,
                    action=action,
                    prior=uniform_prior,
                )
            value = self._random_rollout(env, state)

        node.is_expanded = True
        return value

    def _random_rollout(self, env, state) -> float:
        """Phase 1: random playout until terminal or depth limit."""
        rollout_state = env.clone_state(state)
        current_player = env.get_current_player(rollout_state)

        for _ in range(self.config.max_rollout_depth):
            valid = env.get_valid_actions(rollout_state)
            if len(valid) == 0:
                return 0.0

            action = np.random.choice(valid)
            rollout_state, reward, done, info = env.step(rollout_state, action)

            if done:
                winner = info.get("winner", None)
                if winner == current_player:
                    return 1.0
                elif winner is not None:
                    return -1.0
                return 0.0

        return 0.0  # depth limit — inconclusive

    def _backpropagate(self, node: Node, value: float):
        """
        Propagate value up the tree, flipping sign at each level.

        Value is from the perspective of the current player at `node`.
        Parent's current player is the opponent => flip.
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def _add_dirichlet_noise(self, root: Node):
        """Dirichlet noise at root for exploration."""
        if not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(actions)
        )
        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            root.children[action].prior = (
                (1 - eps) * root.children[action].prior + eps * noise[i]
            )

    def _action_probabilities(
        self, root: Node, action_space_size: int, temperature: float
    ) -> np.ndarray:
        """Convert root visit counts to probability distribution."""
        counts = np.zeros(action_space_size, dtype=np.float64)
        for action, child in root.children.items():
            counts[action] = child.visit_count

        if temperature < 1e-8:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
            return probs

        counts = counts ** (1.0 / temperature)
        total = counts.sum()
        if total > 0:
            return counts / total
        return np.ones(action_space_size) / action_space_size
