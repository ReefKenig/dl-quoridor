"""Leaf-parallel (virtual-loss) MCTS + batched-inference wire.

Guards for the leaf_batch>1 path added to MCTSMaxN and the multi-leaf request
protocol in batched_inference_mp:

  - leaf_batch=1 stays the sequential path (validity + determinism).
  - leaf_batch>1 produces a valid visit distribution, exact simulation accounting
    (root.visit_count == num_simulations), and fully removes virtual loss.
  - the parallel distribution stays CLOSE to the sequential one.
  - the GPU batcher expands a stacked (b,C,H,W) request into b ordered replies and
    returns a single-tuple reply for a plain (C,H,W) request.
"""
import queue as queuemod
import threading

import numpy as np
import torch

from src.env.quoridor_env_mp import QuoridorEnvMP
from src.mcts.mcts_maxn import MCTSConfig, MCTSMaxN, Node
from src.mcts.batched_inference_mp import _inference_worker

N_PLAYERS = 2
SIMS = 120


def _env():
    return QuoridorEnvMP(board_size=5, num_players=N_PLAYERS,
                         max_turns=300, max_walls_per_player=3)


def _evals(action_space_size):
    """Deterministic, state-independent eval (uniform policy, zero value) so the
    search is fully deterministic once Dirichlet noise is disabled. Returns both a
    single-state fn (sequential) and a list fn (leaf-parallel)."""
    def ev_single(state):
        pol = np.full(action_space_size, 1.0 /
                      action_space_size, dtype=np.float32)
        return pol, np.zeros(N_PLAYERS, dtype=np.float32)

    def ev_many(states):
        return [ev_single(s) for s in states]

    return ev_single, ev_many


def _tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def test_sequential_valid_and_deterministic():
    env = _env()
    state = env.reset()
    ev_single, _ = _evals(env.action_space_size)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=SIMS,
                          dirichlet_epsilon=0.0, leaf_batch=1),
        evaluate_fn=ev_single, num_players=N_PLAYERS)
    p1 = mcts.search(env, state, temperature=1.0)
    p2 = mcts.search(env, state, temperature=1.0)
    assert abs(p1.sum() - 1.0) < 1e-9
    assert np.allclose(p1, p2)                       # deterministic (no noise)
    valid = set(env.get_valid_actions(state))
    assert all(p1[a] == 0.0 for a in range(len(p1)) if a not in valid)


def test_leaf_parallel_valid_distribution():
    env = _env()
    state = env.reset()
    _, ev_many = _evals(env.action_space_size)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=SIMS, dirichlet_epsilon=0.0,
                          leaf_batch=8, virtual_loss=1.0),
        evaluate_fn=ev_many, num_players=N_PLAYERS)
    p = mcts.search(env, state, temperature=1.0)
    assert abs(p.sum() - 1.0) < 1e-9
    valid = set(env.get_valid_actions(state))
    assert all(p[a] == 0.0 for a in range(len(p)) if a not in valid)


def test_leaf_parallel_accounting_and_vloss_cleanup():
    """White-box: exactly num_simulations backprops reach the root, root children
    visits sum to num_simulations, and every node's virtual loss is fully removed."""
    env = _env()
    state = env.reset()
    _, ev_many = _evals(env.action_space_size)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=SIMS, dirichlet_epsilon=0.0,
                          leaf_batch=8, virtual_loss=1.0),
        evaluate_fn=ev_many, num_players=N_PLAYERS)

    root = Node(num_players=N_PLAYERS)
    mcts._expand_root_batched(root, env, env.clone_state(state))
    mcts._run_leaf_parallel(root, env, state, leaf_batch=8)

    assert root.visit_count == SIMS
    assert sum(c.visit_count for c in root.children.values()) == SIMS

    stack, total_vloss = [root], 0
    while stack:
        nd = stack.pop()
        total_vloss += nd.n_vloss
        assert nd.visit_count >= 0
        stack.extend(nd.children.values())
    assert total_vloss == 0                          # virtual loss fully unwound


def test_leaf_parallel_close_to_sequential():
    env = _env()
    state = env.reset()
    ev_single, ev_many = _evals(env.action_space_size)
    seq = MCTSMaxN(
        config=MCTSConfig(num_simulations=SIMS,
                          dirichlet_epsilon=0.0, leaf_batch=1),
        evaluate_fn=ev_single, num_players=N_PLAYERS)
    par = MCTSMaxN(
        config=MCTSConfig(num_simulations=SIMS, dirichlet_epsilon=0.0,
                          leaf_batch=8, virtual_loss=1.0),
        evaluate_fn=ev_many, num_players=N_PLAYERS)
    p_seq = seq.search(env, state, temperature=1.0)
    p_par = par.search(env, state, temperature=1.0)
    # Virtual loss perturbs individual counts but the distribution stays close.
    assert _tv(p_seq, p_par) < 0.20


class _FakeNet:
    def eval(self):
        pass


class _FakeModel:
    """Minimal predict_batch stub: encodes each row's sum into policy[:, 0] so we
    can assert reply ordering matches request-row ordering."""

    def __init__(self, action_space_size, num_players):
        self.network = _FakeNet()
        self.device = torch.device("cpu")
        self.A = action_space_size
        self.N = num_players

    def predict_batch(self, stacked):
        b = stacked.shape[0]
        pol = torch.zeros(b, self.A)
        val = torch.zeros(b, self.N)
        for i in range(b):
            pol[i, 0] = float(stacked[i].sum())
        return pol, val


def test_batcher_expands_stacked_request_and_preserves_order():
    A, N, C, H, W = 5, 2, 3, 5, 5
    model = _FakeModel(A, N)
    request_q = queuemod.Queue()
    response_qs = {0: queuemod.Queue(), 1: queuemod.Queue()}
    stop = threading.Event()

    stacked = np.arange(
        3 * C * H * W, dtype=np.float32).reshape(3, C, H, W)
    single = np.ones((C, H, W), dtype=np.float32)
    request_q.put((0, 0, stacked))   # worker 0: 3 leaves in one message
    request_q.put((1, 0, single))    # worker 1: one leaf

    th = threading.Thread(
        target=_inference_worker,
        args=({0: model}, request_q, response_qs,
              8, stop, None, lambda *a: None),
        daemon=True)
    th.start()
    try:
        r0 = response_qs[0].get(timeout=5)   # LIST of 3 (policy, value)
        r1 = response_qs[1].get(timeout=5)   # single (policy, value)
    finally:
        stop.set()
        request_q.put("STOP")
        th.join(timeout=5)

    assert isinstance(r0, list) and len(r0) == 3
    assert isinstance(r1, tuple) and len(r1) == 2
    for i in range(3):
        assert abs(float(r0[i][0][0]) - float(stacked[i].sum())) < 1e-3
    assert abs(float(r1[0][0]) - float(single.sum())) < 1e-3


# --------------------------------------------------------------------------
# VectorizedSearch parity: the step-by-step coordinator (Option B) must reproduce
# the sequential search bit-for-bit — it is the correctness anchor for building
# vectorized self-play on top of it.
# --------------------------------------------------------------------------

def _state_dep_eval(env, num_players=N_PLAYERS):
    """Deterministic but state-DEPENDENT eval: distinct priors/values per leaf so a
    mis-wired coordinator (wrong node, policy, or backprop) breaks parity."""
    A = env.action_space_size

    def ev(state):
        t = env.state_to_tensor(state).ravel().astype(np.float64)
        s = float(t.sum())
        idx = np.arange(1, A + 1, dtype=np.float64)
        pol = np.abs(np.sin(s * 0.123 + idx)) + 1e-3
        pol = (pol / pol.sum()).astype(np.float32)
        val = np.tanh(np.array([s * (k + 1) * 0.01
                                for k in range(num_players)])).astype(np.float32)
        return pol, val
    return ev


def _drive(vs, ev):
    while not vs.done():
        leaf = vs.collect()
        if leaf is not None:
            pol, val = ev(leaf)
            vs.apply(pol, val)
    return vs


def test_vectorized_parity_matches_sequential():
    """Coordinator visit distribution is BIT-IDENTICAL to MCTSMaxN.search (eps=0)."""
    from src.mcts.mcts_maxn import VectorizedSearch

    env = _env()
    state = env.reset()
    ev = _state_dep_eval(env)
    cfg = MCTSConfig(num_simulations=SIMS, dirichlet_epsilon=0.0, leaf_batch=1)

    seq = MCTSMaxN(config=cfg, evaluate_fn=ev, num_players=N_PLAYERS)
    p_seq = seq.search(env, state, temperature=1.0)

    coord = MCTSMaxN(config=cfg, evaluate_fn=None, num_players=N_PLAYERS)
    vs = _drive(VectorizedSearch(coord, env, state), ev)
    p_vec = vs.action_probs(1.0)

    assert np.array_equal(p_seq, p_vec)
    assert vs.root.visit_count == SIMS
    assert sum(c.visit_count for c in vs.root.children.values()) == SIMS
    valid = set(env.get_valid_actions(state))
    assert all(p_vec[a] == 0.0 for a in range(len(p_vec)) if a not in valid)


def test_vectorized_parity_n4():
    """Parity also holds at N=4 (vector value head, max^n backprop)."""
    from src.mcts.mcts_maxn import VectorizedSearch

    env = QuoridorEnvMP(board_size=5, num_players=4,
                        max_turns=300, max_walls_per_player=3)
    state = env.reset()
    ev = _state_dep_eval(env, num_players=4)
    cfg = MCTSConfig(num_simulations=SIMS, dirichlet_epsilon=0.0, leaf_batch=1)

    seq = MCTSMaxN(config=cfg, evaluate_fn=ev, num_players=4)
    p_seq = seq.search(env, state, temperature=1.0)

    coord = MCTSMaxN(config=cfg, evaluate_fn=None, num_players=4)
    vs = _drive(VectorizedSearch(coord, env, state), ev)
    assert np.array_equal(p_seq, vs.action_probs(1.0))


def test_vectorized_parity_with_dirichlet_seeded():
    """With exploration noise on, parity holds when the RNG is seeded identically
    (Dirichlet is the only RNG draw; both search and coordinator draw it once)."""
    from src.mcts.mcts_maxn import VectorizedSearch

    env = _env()
    state = env.reset()
    ev = _state_dep_eval(env)
    cfg = MCTSConfig(num_simulations=SIMS, dirichlet_epsilon=0.25, leaf_batch=1)

    seq = MCTSMaxN(config=cfg, evaluate_fn=ev, num_players=N_PLAYERS)
    np.random.seed(1234)
    p_seq = seq.search(env, state, temperature=1.0)

    coord = MCTSMaxN(config=cfg, evaluate_fn=None, num_players=N_PLAYERS)
    np.random.seed(1234)
    vs = _drive(VectorizedSearch(coord, env, state), ev)
    assert np.array_equal(p_seq, vs.action_probs(1.0))
