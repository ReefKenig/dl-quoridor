"""Distance planes are versioned, because the divisor changed under old models.

Spec v1 divides BFS distance by board_size**2, v2 by 2*board_size. Feeding a v1
checkpoint v2 planes rescales channels 8/9 by board_size/2 with no error raised,
so the only defence is that every loader passes the spec its model trained under.
"""
import json
import os

import pytest

from src.env.pathing import (CURRENT_SPEC, SPEC_V1_DIST_SQ, SPEC_V2_DIST_2BS,
                             dist_norm, distance_map)
from src.env.quoridor_env_mp import QuoridorEnvMP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_divisors_match_their_spec():
    assert dist_norm(9, SPEC_V1_DIST_SQ) == 81
    assert dist_norm(9, SPEC_V2_DIST_2BS) == 18
    assert dist_norm(5, SPEC_V1_DIST_SQ) == 25
    assert dist_norm(5, SPEC_V2_DIST_2BS) == 10
    assert CURRENT_SPEC == SPEC_V2_DIST_2BS


def test_unknown_spec_is_rejected_not_defaulted():
    with pytest.raises(ValueError):
        dist_norm(9, 99)


@pytest.mark.parametrize("board_size", [5, 9])
def test_v1_and_v2_differ_by_the_expected_scale(board_size):
    """The whole reason the spec is versioned: same board, different numbers."""
    v1 = distance_map(board_size, ("row", 0), set(), set(),
                      spec_version=SPEC_V1_DIST_SQ)
    v2 = distance_map(board_size, ("row", 0), set(), set(),
                      spec_version=SPEC_V2_DIST_2BS)

    # One step from the goal, before either divisor clips.
    assert v1[1, 0] == pytest.approx(1.0 / (board_size * board_size))
    assert v2[1, 0] == pytest.approx(1.0 / (2 * board_size))
    assert v2[1, 0] == pytest.approx(v1[1, 0] * board_size / 2)


def test_both_specs_stay_in_range_and_agree_on_the_goal_edge():
    for spec in (SPEC_V1_DIST_SQ, SPEC_V2_DIST_2BS):
        d = distance_map(9, ("row", 8), set(), set(), spec_version=spec)
        assert d.min() >= 0.0 and d.max() <= 1.0
        assert (d[8, :] == 0.0).all()


def test_env_threads_its_spec_into_the_tensor():
    state_v1 = QuoridorEnvMP(board_size=5, num_players=2,
                             spec_version=SPEC_V1_DIST_SQ)
    state_v2 = QuoridorEnvMP(board_size=5, num_players=2,
                             spec_version=SPEC_V2_DIST_2BS)

    t1 = state_v1.state_to_tensor(state_v1.reset())
    t2 = state_v2.state_to_tensor(state_v2.reset())

    # Distance planes for N=2 are channels 6 and 7 (3N+3 = 9 planes).
    assert not (t1[:, :, 6] == t2[:, :, 6]).all()
    # Everything else is spec-independent.
    assert (t1[:, :, :6] == t2[:, :, :6]).all()


def test_env_defaults_to_the_current_spec():
    env = QuoridorEnvMP(board_size=5, num_players=2)
    assert env.spec_version == CURRENT_SPEC


def test_every_registered_model_declares_its_spec():
    """A checkpoint without a declared spec would silently take the current one."""
    with open(os.path.join(REPO, "runs", "MODELS.json")) as f:
        registry = json.load(f)["models"]

    for name, spec in registry.items():
        assert "tensor_spec" in spec, f"{name} does not declare tensor_spec"
        assert spec["tensor_spec"] in (SPEC_V1_DIST_SQ, SPEC_V2_DIST_2BS)
