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
    # Leaf-parallel MCTS: collect up to leaf_batch leaves per GPU forward, using a
    # virtual loss to diversify the concurrent tree walks. leaf_batch=1 disables
    # batching entirely (the sequential path, byte-identical to the original).
    leaf_batch: int = 1
    virtual_loss: float = 1.0


class Node:
    __slots__ = ["parent", "action", "prior", "current_player",
                 "visit_count", "value_sum", "children", "is_expanded",
                 "valid_actions", "n_vloss"]

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
        # Count of virtual-loss "visits" temporarily applied during a leaf-parallel
        # wave; always nets back to 0 once the wave's real backprops land.
        self.n_vloss = 0

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
        root = Node(num_players=self.num_players)
        root_state = env.clone_state(state)
        leaf_batch = max(1, int(self.config.leaf_batch))
        # Root expansion differs by mode: the sequential path's evaluate_fn takes a
        # single state; the leaf-parallel path's takes a list. Kept separate so
        # leaf_batch=1 stays byte-identical to the original search.
        if leaf_batch <= 1:
            self._expand(root, env, root_state)
        else:
            self._expand_root_batched(root, env, root_state)
        self._add_dirichlet_noise(root)

        if leaf_batch <= 1:
            self._run_sequential(root, env, state)
        else:
            self._run_leaf_parallel(root, env, state, leaf_batch)

        temp = temperature if temperature is not None else self.config.temperature
        return self._action_probabilities(root, env.action_space_size, temp)

    def _run_sequential(self, root, env, state):
        N = self.num_players
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

    def _select_child(self, node):
        pm = node.current_player
        return max(node.children.values(),
                   key=lambda c: c.ucb_score(self.config.c_puct, pm))

    def _expand(self, node, env, state):
        valid_actions = self._expand_valids(node, env, state)
        if len(valid_actions) == 0:
            node.is_expanded = True
            return np.zeros(self.num_players)
        if self.evaluate_fn is not None:
            policy, value = self.evaluate_fn(state)   # value: (N,)
            value = self._set_children_from_policy(
                node, env, valid_actions, policy, value)
        else:
            up = 1.0 / len(valid_actions)
            for a in valid_actions:
                node.children[a] = Node(parent=node, action=a,
                                        prior=up, num_players=self.num_players)
            value = self._random_rollout(env, state)
        node.is_expanded = True
        return value

    def _expand_valids(self, node, env, state):
        """Set node.current_player + valid_actions; return the valid action list."""
        node.current_player = env.get_current_player(state)
        valid_actions = env.get_valid_actions(state)
        node.valid_actions = valid_actions
        return valid_actions

    def _set_children_from_policy(self, node, env, valid_actions, policy, value):
        """Create masked-prior children and return the leaf value vector.

        Identical maths to the original _expand evaluate branch, factored out so
        the leaf-parallel path can reuse it after a batched forward pass.
        """
        mask = np.zeros(env.action_space_size, dtype=np.float32)
        mask[valid_actions] = 1.0
        policy = policy * mask
        s = policy.sum()
        policy = policy / s if s > 0 else mask / mask.sum()
        for a in valid_actions:
            node.children[a] = Node(parent=node, action=a,
                                    prior=policy[a], num_players=self.num_players)
        return np.asarray(value, dtype=np.float64).reshape(self.num_players)

    # ---- leaf-parallel (virtual-loss) search ------------------------------
    def _terminal_value(self, info):
        N = self.num_players
        value = np.full(N, -1.0)
        winner = info.get("winner", None)
        if winner is None:
            value[:] = 0.0
        else:
            value[winner] = 1.0
        return value

    def _expand_root_batched(self, root, env, state):
        """Expand the root via a 1-element batched eval (parallel-path callers hand
        in a list-accepting evaluate_fn)."""
        valids = self._expand_valids(root, env, state)
        if len(valids) == 0:
            root.is_expanded = True
            return
        policy, value = self.evaluate_fn([state])[0]
        self._set_children_from_policy(root, env, valids, policy, value)
        root.is_expanded = True

    def _descend(self, root, env, state):
        """Walk root->leaf, applying a virtual loss to each traversed edge so the
        next walk in the wave diverges. Returns (leaf, leaf_state, path, done, info).
        """
        vl = self.config.virtual_loss
        apply_vl = vl > 0
        node = root
        scratch = env.clone_state(state)
        path = [root]
        done = False
        info = {}
        while node.is_expanded and node.children:
            node = self._select_child(node)
            if apply_vl:
                # Pessimize this child for the player choosing at its parent, so a
                # concurrent walk is steered elsewhere. Undone in _remove_vloss.
                node.visit_count += 1
                node.value_sum[node.parent.current_player] -= vl
                node.n_vloss += 1
            scratch, _, done, info = env.step(scratch, node.action)
            path.append(node)
            if done:
                break
        return node, scratch, path, done, info

    def _remove_vloss(self, path):
        vl = self.config.virtual_loss
        if vl <= 0:
            return
        for nd in path[1:]:   # skip root (never carries virtual loss)
            nd.visit_count -= 1
            nd.value_sum[nd.parent.current_player] += vl
            nd.n_vloss -= 1

    def _run_leaf_parallel(self, root, env, state, leaf_batch):
        """Collect waves of up to leaf_batch leaves and evaluate each wave in one
        batched forward pass, then remove virtual loss and backprop the real values.
        """
        N = self.num_players
        remaining = self.config.num_simulations
        while remaining > 0:
            wave = min(leaf_batch, remaining)
            # (leaf_node, leaf_state, path) awaiting eval
            pending = []
            pending_nodes = set()
            resolved = []           # (path, value) terminals — no eval needed
            attempts = 0
            max_attempts = wave * 4  # bounded so heavy collisions cannot hang
            while len(pending) + len(resolved) < wave and attempts < max_attempts:
                attempts += 1
                node, scratch, path, done, info = self._descend(
                    root, env, state)
                if done:
                    resolved.append((path, self._terminal_value(info)))
                elif node in pending_nodes:
                    self._remove_vloss(path)   # collision: drop this walk
                else:
                    pending_nodes.add(node)
                    pending.append((node, scratch, path))
            if pending:
                results = self.evaluate_fn([p[1] for p in pending])
                for (node, scratch, path), (policy, value) in zip(pending, results):
                    valids = self._expand_valids(node, env, scratch)
                    if len(valids) == 0:
                        val = np.zeros(N)
                    else:
                        val = self._set_children_from_policy(
                            node, env, valids, policy, value)
                    node.is_expanded = True
                    self._remove_vloss(path)
                    self._backpropagate(node, val)
                    remaining -= 1
            for path, val in resolved:
                self._remove_vloss(path)
                self._backpropagate(path[-1], val)
                remaining -= 1
            if not pending and not resolved:
                break   # safety: no progress possible this wave

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
