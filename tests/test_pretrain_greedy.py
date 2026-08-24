"""Greedy-imitation warm start (scripts/pretrain_greedy.py).

The generator must produce the same sample format self-play does - the whole
point is that RL continues from these weights on the same tensors and targets.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import NUM_MOVE_ACTIONS, QuoridorEnvMP
from src.env.tensor_spec_mp import CURRENT_SPEC

from src.mcts.pretrain_data import generate_games, to_arrays


@pytest.fixture(scope="module")
def env():
    return QuoridorEnvMP(board_size=5, num_players=2, max_turns=60,
                         max_walls_per_player=3, spec_version=CURRENT_SPEC)


@pytest.fixture(scope="module")
def games(env):
    return generate_games(env, num_games=12, opening_max=4, max_moves=60,
                          discount=0.99, discount_unit="round", base_seed=0,
                          log=lambda *a: None)


def test_targets_are_one_hot_pawn_moves(games):
    """Greedy never places a wall, so every policy target is a pawn move."""
    S, P, V = to_arrays(games)
    assert np.allclose(P.sum(axis=1), 1.0)
    hot = P.argmax(axis=1)
    # Mirror augmentation maps moves to moves, so this holds for both halves.
    assert (hot < NUM_MOVE_ACTIONS).all()


def test_value_targets_are_signed_vectors(games):
    _, _, V = to_arrays(games)
    assert V.shape[1] == 2
    # One winner per sample: exactly one positive component.
    assert ((V > 0).sum(axis=1) == 1).all()
    assert (np.abs(V) <= 1.0).all()


def test_augmentation_doubles_each_game(games):
    for game in games:
        assert len(game) % 2 == 0


def test_generation_is_deterministic_in_the_seed(env):
    a = generate_games(env, 3, 4, 60, 0.99, "round", base_seed=7,
                       log=lambda *a: None)
    b = generate_games(env, 3, 4, 60, 0.99, "round", base_seed=7,
                       log=lambda *a: None)
    assert len(a) == len(b)
    for ga, gb in zip(a, b):
        for (sa, pa, va), (sb, pb, vb) in zip(ga, gb):
            assert np.array_equal(sa, sb) and np.array_equal(pa, pb)


def test_opening_states_can_contain_walls(env):
    """The random opening is the state diversity: 98% of legal opening actions
    are walls, so greedy must race through walled positions too."""
    games = generate_games(env, 12, opening_max=4, max_moves=60,
                           discount=0.99, discount_unit="round", base_seed=0,
                           log=lambda *a: None)
    S, _, _ = to_arrays(games)
    walls = S[:, :, :, 2:4].sum(axis=(1, 2, 3))   # h+v wall planes at N=2
    assert (walls > 0).any()
