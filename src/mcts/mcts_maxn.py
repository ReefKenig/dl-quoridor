"""
max^n MCTS (correct path for N players).

Differences vs the 2-player negamax MCTS:
  - Node.value_sum is a length-N vector, not a scalar.
  - Each node stores its mover (current_player). Selection maximises the
    child's Q component for the mover AT THE PARENT: Q_child[parent_mover].
    No negation. (The 2p code negates because it stores opponent-perspective
    scalars; here we keep the full vector and index it.)
  - Backprop ADDS the same leaf vector to every ancestor. No sign flip.
  - Terminal value is a vector: +1 winner, -1 others, 0 draw/timeout.

Zero-sum reduction check (N=2): winner=+1/loser=-1 => vector sums to 0, and
maximising Q[parent_mover] == minimising the opponent component == negamax.
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Tuple


@dataclass
class MCTSConfig:
    num_simulations: int = 400
    c_puct: float = 1.41
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_rollout_depth: int = 100


class Node:
    __slots__ = ["parent", "action", "prior", "current_player",
                 "visit_count", "value_sum", "children", "is_expanded", "valid_actions"]

    def __init__(self, parent=None, action=None, prior=0.0, num_players=2):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.current_player: Optional[int] = None
        self.visit_count = 0
        self.value_sum = np.zeros(num_players, dtype=np.float64)
        self.children = {}
        self.is_expanded = False
        self.valid_actions = None

    def q_vector(self):
        if self.visit_count == 0:
            return np.zeros_like(self.value_sum)
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct, parent_mover):
        # exploitation = child's expected outcome FOR THE PLAYER MOVING AT THE PARENT
        exploitation = self.q_vector()[parent_mover]
        exploration = (c_puct * self.prior *
                       math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        return exploitation + exploration


EvaluateFn = Callable[[object], Tuple[np.ndarray, np.ndarray]]


class MCTSMaxN:
    def __init__(self, config=MCTSConfig(), evaluate_fn=None, num_players=2):
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.num_players = num_players

    def search(self, env, state, temperature=None):
        N = self.num_players
        root = Node(num_players=N)
        root_state = env.clone_state(state)
        self._expand(root, env, root_state)
        self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            scratch_state = env.clone_state(state)
            done = False
            info = {}
            while node.is_expanded and node.children:
                node = self._select_child(node)
                scratch_state, _, done, info = env.step(
                    scratch_state, node.action)
                if done:
                    break
            if done:
                winner = info.get("winner", None)
                value = np.full(N, -1.0)
                if winner is None:
                    value[:] = 0.0
                else:
                    value[winner] = 1.0
            else:
                value = self._expand(node, env, scratch_state)
            self._backpropagate(node, value)

        temp = temperature if temperature is not None else self.config.temperature
        return self._action_probabilities(root, env.action_space_size, temp)

    def _select_child(self, node):
        pm = node.current_player
        return max(node.children.values(),
                   key=lambda c: c.ucb_score(self.config.c_puct, pm))

    def _expand(self, node, env, state):
        node.current_player = env.get_current_player(state)
        valid_actions = env.get_valid_actions(state)
        node.valid_actions = valid_actions
        if len(valid_actions) == 0:
            node.is_expanded = True
            return np.zeros(self.num_players)
        if self.evaluate_fn is not None:
            policy, value = self.evaluate_fn(state)   # value: (N,)
            mask = np.zeros(env.action_space_size, dtype=np.float32)
            mask[valid_actions] = 1.0
            policy = policy * mask
            s = policy.sum()
            policy = policy / s if s > 0 else mask / mask.sum()
            for a in valid_actions:
                node.children[a] = Node(parent=node, action=a,
                                        prior=policy[a], num_players=self.num_players)
            value = np.asarray(value, dtype=np.float64).reshape(
                self.num_players)
        else:
            up = 1.0 / len(valid_actions)
            for a in valid_actions:
                node.children[a] = Node(parent=node, action=a,
                                        prior=up, num_players=self.num_players)
            value = self._random_rollout(env, state)
        node.is_expanded = True
        return value

    def _random_rollout(self, env, state):
        N = self.num_players
        rs = env.clone_state(state)
        for _ in range(self.config.max_rollout_depth):
            valid = env.get_valid_actions(rs)
            if len(valid) == 0:
                return np.zeros(N)
            a = int(np.random.choice(valid))
            rs, _, done, info = env.step(rs, a)
            if done:
                winner = info.get("winner", None)
                v = np.full(N, -1.0)
                if winner is None:
                    v[:] = 0.0
                else:
                    v[winner] = 1.0
                return v
        return np.zeros(N)

    def _backpropagate(self, node, value):
        # add the SAME vector to every ancestor; no flip
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _add_dirichlet_noise(self, root):
        # epsilon <= 0 disables exploration noise entirely (used during eval so
        # strength is measured deterministically at true best-play). Short-circuit
        # so we don't even draw from the RNG — the mix below would be a no-op anyway.
        if self.config.dirichlet_epsilon <= 0 or not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon
        for i, a in enumerate(actions):
            root.children[a].prior = (
                1 - eps) * root.children[a].prior + eps * noise[i]

    def _action_probabilities(self, root, action_space_size, temperature):
        counts = np.zeros(action_space_size, dtype=np.float64)
        for a, c in root.children.items():
            counts[a] = c.visit_count
        if temperature < 1e-8:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
            return probs
        counts = counts ** (1.0 / temperature)
        total = counts.sum()
        if total > 0:
            return counts / total
        probs = np.zeros(action_space_size, dtype=np.float64)
        for a in root.children:
            probs[a] = 1.0
        t = probs.sum()
        return probs / t if t > 0 else probs
