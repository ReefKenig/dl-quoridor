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
from src.env.env_interface import MinimalQuoridorStub


def test_mcts_vs_random_stub(num_games: int = 100, num_sims: int = 400):
    """MCTS (random rollouts) vs random agent on stub. Expect >80% win rate."""
    env = MinimalQuoridorStub()
    mcts = MCTS(config=MCTSConfig(num_simulations=num_sims))

    wins = {0: 0, 1: 0, "draw": 0}
    mcts_player = 0

    for game_idx in range(num_games):
        state = env.reset()
        move_count = 0

        while not state.done and move_count < 200:
            if state.current_player == mcts_player:
                probs = mcts.search(env, state, temperature=0.1)
                action = np.argmax(probs)
            else:
                valid = env.get_valid_actions(state)
                action = np.random.choice(valid)

            state, _, done, info = env.step(state, action)
            move_count += 1

        if state.winner == mcts_player:
            wins[0] += 1
        elif state.winner is not None:
            wins[1] += 1
        else:
            wins["draw"] += 1

    win_rate = wins[0] / num_games
    assert win_rate > 0.80, f"MCTS win rate {win_rate:.1%} < 80%"


def test_replay_buffer():
    """Verify replay buffer stores and samples correctly."""
    buf = ReplayBuffer(max_size=1000)
    samples = [
        TrainingSample(
            state=np.random.randn(5, 5, 10).astype(np.float32),
            policy_target=np.random.dirichlet(np.ones(4)),
            value_target=np.random.choice([-1.0, 1.0]),
        )
        for _ in range(200)
    ]
    buf.add(samples)

    states, policies, values = buf.sample_batch(32)

    assert states.shape == (32, 5, 5, 10)
    assert policies.shape == (32, 4)
    assert values.shape == (32,)
    assert len(buf) == 200


def test_replay_buffer_overflow():
    """Verify FIFO eviction when buffer exceeds max_size."""
    buf = ReplayBuffer(max_size=50)
    samples = [
        TrainingSample(
            state=np.zeros((5, 5, 10), dtype=np.float32),
            policy_target=np.ones(4) / 4,
            value_target=1.0,
        )
        for _ in range(100)
    ]
    buf.add(samples)
    assert len(buf) == 50


@pytest.mark.slow
def test_mcts_vs_random_real():
    """MCTS vs Random on real QuoridorEnv (5×5 POC)."""
    from src.env.quoridor_env import QuoridorEnv

    env = QuoridorEnv(is_poc=True)
    mcts = MCTS(config=MCTSConfig(num_simulations=400))

    wins = 0
    num_games = 50
    for _ in range(num_games):
        state = env.reset()
        while True:
            if env.get_current_player(state) == 0:
                probs = mcts.search(env, state, temperature=0.1)
                action = np.argmax(probs)
            else:
                valid = env.get_valid_actions(state)
                action = np.random.choice(valid)
            state, _, done, info = env.step(state, action)
            if done:
                if info["winner"] == 0:
                    wins += 1
                break

    win_rate = wins / num_games
    assert win_rate > 0.80, f"MCTS win rate {win_rate:.1%} < 80%"
