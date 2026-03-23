"""
MCTS Validation Tests
======================
Owner: Iris

Run from project root:
    pytest tests/test_mcts.py -v
    pytest tests/test_mcts.py -v -k "not slow"   # skip long MCTS test
"""

import numpy as np
import pytest

from src.mcts.mcts import MCTS, MCTSConfig
from src.mcts.self_play import ReplayBuffer, TrainingSample
from src.env.quoridor_env import QuoridorEnv


def test_replay_buffer():
    """Verify replay buffer stores and samples correctly."""
    buf = ReplayBuffer(max_size=1000)
    samples = [
        TrainingSample(
            state=np.random.default_rng().random((5, 5, 10)).astype(np.float32),
            policy_target=np.random.default_rng().dirichlet(np.ones(44)),
            value_target=np.random.default_rng().choice([-1.0, 1.0]),
        )
        for _ in range(200)
    ]
    buf.add(samples)

    states, policies, values = buf.sample_batch(32)

    assert states.shape == (32, 5, 5, 10)
    assert policies.shape == (32, 44)
    assert values.shape == (32,)
    assert len(buf) == 200


def test_replay_buffer_overflow():
    """Verify FIFO eviction when buffer exceeds max_size."""
    buf = ReplayBuffer(max_size=50)
    samples = [
        TrainingSample(
            state=np.zeros((5, 5, 10), dtype=np.float32),
            policy_target=np.ones(44) / 44,
            value_target=1.0,
        )
        for _ in range(100)
    ]
    buf.add(samples)
    assert len(buf) == 50


@pytest.mark.slow
def test_mcts_vs_random_real_env(num_games: int = 20, num_sims: int = 200):
    """MCTS (random rollouts) vs random agent on real 5x5 QuoridorEnv."""
    env = QuoridorEnv(is_poc=True)
    mcts = MCTS(config=MCTSConfig(num_simulations=num_sims))

    wins = 0
    mcts_player = 0

    for _ in range(num_games):
        state = env.reset()
        move_count = 0

        while not state.game_over and move_count < 200:
            if state.current_player == mcts_player:
                probs = mcts.search(env, state, temperature=0.1)
                action = np.argmax(probs)
            else:
                valid = env.get_valid_actions(state)
                action = np.random.default_rng().choice(valid)

            state, _, _, _ = env.step(state, action)
            move_count += 1

        if state.winner == mcts_player:
            wins += 1

    win_rate = wins / num_games
    assert win_rate > 0.50, f"MCTS win rate on real env {win_rate:.1%} < 50%"
