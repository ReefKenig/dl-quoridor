"""
Smoke Test
===========
Run: pytest tests/smoke_test.py -v

End-to-end integration: env + model + MCTS + self-play.
"""

import pytest

from src.env.quoridor_env import QuoridorEnv
from src.model.network import QuoridorModel
from src.mcts.mcts import MCTS, MCTSConfig
from src.mcts.self_play import play_one_game


@pytest.fixture
def env():
    return QuoridorEnv(board_size=5)


@pytest.fixture
def model():
    return QuoridorModel(
        board_size=5, action_space_size=44, num_channels=16, num_res_blocks=2
    )


def test_full_game(env, model):
    """A full self-play game produces valid training samples."""

    def nn_evaluate(state):
        tensor = env.state_to_tensor(state)
        return model.predict(tensor)

    mcts = MCTS(config=MCTSConfig(num_simulations=50), evaluate_fn=nn_evaluate)
    samples, winner = play_one_game(env, mcts, max_moves=100)

    assert len(samples) > 0
    assert samples[0].state.shape[0] == 5
    assert samples[0].policy_target.shape == (44,)
    assert winner in (0, 1, None)
