"""
Parity + orchestration tests for vectorized_self_play_mp.py (Option B).

The correctness claim: the in-process vectorized driver produces the SAME training
samples as exact sequential self-play. With deterministic settings (dirichlet
eps=0, temperature=0 => one-hot action), we assert:

  - one game through the driver == an inline sequential reference, bit-for-bit
    (search visit dists, value targets, and mirror augmentation);
  - concurrency-invariance: G concurrent games == running them one at a time;
  - the return contract (exact game count, wins dict) matches the parallel path.

Sequential eval uses model.predict_batch on a 1-row stack so its numerics are
identical to the driver's batched forward.
"""
import numpy as np
import torch

from src.env.quoridor_env_mp import QuoridorEnvMP
from src.model.network_mp import QuoridorModelMP
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.mcts.self_play_mp import assign_vector_targets, augment_mp
from src.mcts.vectorized_self_play_mp import generate_vectorized_self_play_mp


class _Cfg:
    def __init__(self, num_players=2, eps=0.0):
        self.num_players = num_players
        self.board_size = 5
        self.max_walls_per_player = 3
        self.max_turns = 60
        self.mcts_simulations = 40
        self.discount = 0.99
        self.explore_moves = 6
        self.max_game_moves = 60
        self.mcts_dirichlet_epsilon = eps


def _model(num_players=2):
    m = QuoridorModelMP(board_size=5, action_space_size=44,
                        num_players=num_players, num_channels=16, num_res_blocks=2,
                        device="cpu")
    m.network.eval()
    return m


def _ev_single(model, env):
    """Single-state eval via predict_batch(1 row) — identical numerics to driver."""
    def ev(s):
        stacked = torch.from_numpy(np.ascontiguousarray(
            np.stack([env.state_to_tensor(s).transpose(2, 0, 1)]),
            dtype=np.float32)).to(model.device)
        pols, vals = model.predict_batch(stacked)
        return pols.cpu().numpy()[0], vals.cpu().numpy()[0]
    return ev


def _seq_selfplay(model, cfg):
    """Exact sequential self-play (temp=0 => argmax; eps=0). Reuses the same
    finalize + augmentation helpers as the driver."""
    N = cfg.num_players
    env = QuoridorEnvMP(board_size=cfg.board_size, num_players=N,
                        max_turns=cfg.max_turns,
                        max_walls_per_player=cfg.max_walls_per_player)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=cfg.mcts_simulations,
                          dirichlet_epsilon=cfg.mcts_dirichlet_epsilon,
                          max_rollout_depth=cfg.max_game_moves),
        evaluate_fn=_ev_single(model, env), num_players=N)
    state = env.reset()
    traj, mc, winner = [], 0, None
    while True:
        if mc >= cfg.max_game_moves:
            winner = None
            break
        probs = mcts.search(env, state, temperature=0.0)
        probs = probs / probs.sum()
        mover = env.get_current_player(state)
        traj.append((env.state_to_tensor(state), probs, mover))
        action = int(np.random.choice(len(probs), p=probs))
        state, _, done, info = env.step(state, action)
        mc += 1
        if done:
            winner = info.get("winner")
            break
    s = assign_vector_targets(traj, winner, N, cfg.discount)
    aug = [augment_mp(t, p, v, N, env.board_size) for (t, p, v) in s]
    return s + aug, winner


def _assert_samples_equal(a, b):
    assert len(a) == len(b), f"sample count {len(a)} != {len(b)}"
    for (sa, pa, va), (sb, pb, vb) in zip(a, b):
        assert np.array_equal(sa, sb)
        assert np.array_equal(pa, pb)
        assert np.array_equal(va, vb)


def test_vectorized_one_game_matches_sequential():
    """One driven game is bit-identical to the sequential reference (N=2)."""
    cfg = _Cfg(num_players=2, eps=0.0)
    model = _model(2)
    ref_samples, _ = _seq_selfplay(model, cfg)
    drv_samples, _ = generate_vectorized_self_play_mp(
        model, cfg, total_games=1, vec_games=1,
        explore_temp=0.0, final_temp=0.0)
    _assert_samples_equal(ref_samples, drv_samples)


def test_vectorized_one_game_matches_sequential_n4():
    """Parity for one game at N=4 (vector value head)."""
    cfg = _Cfg(num_players=4, eps=0.0)
    model = _model(4)
    ref_samples, _ = _seq_selfplay(model, cfg)
    drv_samples, _ = generate_vectorized_self_play_mp(
        model, cfg, total_games=1, vec_games=1,
        explore_temp=0.0, final_temp=0.0)
    _assert_samples_equal(ref_samples, drv_samples)


def test_vectorized_concurrency_invariant():
    """Deterministic games are identical whether run G-at-once or one at a time."""
    cfg = _Cfg(num_players=2, eps=0.0)
    model = _model(2)
    wide, wins_w = generate_vectorized_self_play_mp(
        model, cfg, total_games=3, vec_games=3,
        explore_temp=0.0, final_temp=0.0)
    seq, wins_s = generate_vectorized_self_play_mp(
        model, cfg, total_games=3, vec_games=1,
        explore_temp=0.0, final_temp=0.0)
    _assert_samples_equal(wide, seq)
    assert wins_w == wins_s
    assert sum(wins_w.values()) == 3


def test_vectorized_exact_game_count_uneven():
    """Return contract: plays exactly total_games even when it isn't a multiple of
    vec_games, and the progress callback fires total_games times."""
    cfg = _Cfg(num_players=2, eps=0.25)
    model = _model(2)
    seen = []
    samples, wins = generate_vectorized_self_play_mp(
        model, cfg, total_games=5, vec_games=3,
        on_games_complete=lambda d, t, w: seen.append(d))
    assert sum(wins.values()) == 5
    assert seen[-1] == 5
    assert len(samples) > 0
