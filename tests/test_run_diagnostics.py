"""Per-iteration data diagnostics.

These exist because this project has twice been unable to attribute a change to a
cause. They measure the wall-curriculum mechanism directly rather than by
inference: how much of the POLICY TARGET is wall actions, how many trained-on
positions contain a wall at all, and whether the value head is worse on walled
states than wall-free ones -- which is the coverage gap the masked phase creates.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import (NUM_MOVE_ACTIONS, QuoridorEnvMP,
                                     compute_action_space_size)
from src.mcts.training_mp import sample_diagnostics


def _sample(board_size=9, num_players=2, walls=(), policy_wall_mass=0.0):
    """One (tensor, policy, value) triple with a chosen wall content."""
    env = QuoridorEnvMP(board_size=board_size, num_players=num_players,
                        max_walls_per_player=10)
    state = env.reset()
    for is_h, r, c in walls:
        (state.h_walls if is_h else state.v_walls).add((r, c))
    tensor = env.state_to_tensor(state)

    action_space = compute_action_space_size(board_size)
    policy = np.zeros(action_space, np.float32)
    policy[0] = 1.0 - policy_wall_mass
    if policy_wall_mass:
        policy[NUM_MOVE_ACTIONS:] = policy_wall_mass / (action_space - NUM_MOVE_ACTIONS)
    value = np.zeros(num_players, np.float32)
    return tensor, policy, value


def test_no_samples_reports_nothing():
    assert sample_diagnostics([], num_players=2) == {}


def test_wall_free_samples_read_as_wall_free():
    d = sample_diagnostics([_sample() for _ in range(4)], num_players=2)
    assert d["walled_state_share"] == 0.0
    assert d["walls_on_board_mean"] == 0.0


def test_a_wall_on_the_board_is_detected():
    d = sample_diagnostics([_sample(walls=[(True, 2, 3)])], num_players=2)
    assert d["walled_state_share"] == 1.0
    assert d["walls_on_board_mean"] > 0


def test_the_walled_share_is_a_fraction_not_a_flag():
    samples = [_sample(), _sample(walls=[(True, 1, 1)]),
               _sample(), _sample(walls=[(False, 2, 2)])]
    assert sample_diagnostics(samples, num_players=2)["walled_state_share"] == 0.5


def test_policy_wall_mass_tracks_the_target_not_the_board():
    """The quantity the curriculum analysis measured, on the policy TARGET."""
    free = sample_diagnostics([_sample(policy_wall_mass=0.0)], num_players=2)
    heavy = sample_diagnostics([_sample(policy_wall_mass=0.25)], num_players=2)
    assert free["policy_wall_mass"] == pytest.approx(0.0, abs=1e-6)
    assert heavy["policy_wall_mass"] == pytest.approx(0.25, abs=1e-5)


def test_wall_planes_are_read_at_the_right_channels_for_four_players():
    # The wall planes sit at channels N and N+1, so a hardcoded 2 would read
    # pawn planes at N=4 and report walls that are not there.
    clean = sample_diagnostics(
        [_sample(num_players=4) for _ in range(3)], num_players=4)
    assert clean["walled_state_share"] == 0.0
    walled = sample_diagnostics(
        [_sample(num_players=4, walls=[(True, 3, 3)])], num_players=4)
    assert walled["walled_state_share"] == 1.0


def test_value_error_is_split_by_state_type_when_a_model_is_given():
    class _Stub:
        def predict(self, tensor):
            # Wrong by 1.0 on walled states, exact on wall-free ones.
            walled = np.count_nonzero(tensor[:, :, 2:4]) > 0
            return None, np.array([1.0, 1.0]) if walled else np.array([0.0, 0.0])

    samples = [_sample(), _sample(walls=[(True, 1, 1)]), _sample()]
    d = sample_diagnostics(samples, num_players=2, model=_Stub())
    assert d["value_mae_wallfree"] == pytest.approx(0.0)
    assert d["value_mae_walled"] == pytest.approx(1.0)


def test_no_model_means_no_value_error_keys():
    d = sample_diagnostics([_sample()], num_players=2)
    assert "value_mae_walled" not in d and "value_mae_wallfree" not in d
