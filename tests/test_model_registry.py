"""
Model Registry Tests
====================
Run: pytest tests/test_model_registry.py -v

The registry's job is to keep a checkpoint together with the architecture,
tensor spec and wall count it was trained under. A mismatch in architecture
raises; a mismatch in spec or walls does not, which is why they are asserted
here rather than left to a shape check at load time.
"""

import json

import pytest

from src.utils.model_registry import (
    available_variants,
    build_env,
    build_model,
    load_registry,
    variant_key,
    variant_spec,
)


def _write_registry(tmp_path, models, variants):
    root = tmp_path
    (root / "runs").mkdir()
    path = root / "runs" / "MODELS.json"
    with open(path, "w") as f:
        json.dump({"models": models, "variants": variants}, f)
    return path, root


def test_variant_key():
    assert variant_key(5, 2) == "5x5_2p"
    assert variant_key(9, 4) == "9x9_4p"


def test_prose_keys_are_not_variants(tmp_path):
    path, root = _write_registry(
        tmp_path,
        {"m": {"path": "x.pt", "tensor_spec": 1, "in_channels": 9}},
        {"_note": "prose", "5x5_2p": {"model": "m", "max_walls": 3}},
    )
    registry = load_registry(path)
    assert set(available_variants(registry)) == {"5x5_2p"}


def test_unknown_variant_lists_what_exists(tmp_path):
    path, root = _write_registry(
        tmp_path,
        {"m": {"path": "x.pt", "tensor_spec": 1, "in_channels": 9}},
        {"5x5_2p": {"model": "m", "max_walls": 3}},
    )
    with pytest.raises(KeyError, match="5x5_2p"):
        variant_spec(9, 4, registry=load_registry(path), root=root)


def test_run_dir_entry_goes_through_the_ship_resolver(tmp_path):
    """A run_dir entry follows resolve_ship_checkpoint, not a pinned file."""
    path, root = _write_registry(
        tmp_path,
        {"m": {"run_dir": "runs/a_run", "tensor_spec": 2, "in_channels": 9,
               "num_channels": 128, "num_res_blocks": 8}},
        {"9x9_2p": {"model": "m", "max_walls": 10, "max_turns": 200}},
    )
    run = root / "runs" / "a_run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"untrained")
    (run / "greedy_peak.pt").write_bytes(b"the peak")

    spec = variant_spec(9, 2, registry=load_registry(path), root=root)
    assert spec.checkpoint.endswith("greedy_peak.pt")
    assert "greedy_peak.pt" in spec.label
    assert spec.max_walls == 10 and spec.max_turns == 200
    assert spec.num_channels == 128 and spec.num_res_blocks == 8
    assert spec.tensor_spec == 2


def test_missing_path_entry_is_reported_not_raised(tmp_path):
    path, root = _write_registry(
        tmp_path,
        {"m": {"path": "runs/gone.pt", "tensor_spec": 1, "in_channels": 9}},
        {"5x5_2p": {"model": "m", "max_walls": 3}},
    )
    spec = variant_spec(5, 2, registry=load_registry(path), root=root)
    assert spec.checkpoint is None
    assert not spec.is_loadable
    assert "missing" in spec.label


def test_arch_defaults_apply_to_legacy_entries(tmp_path):
    """The 5x5 POC entries predate num_channels/num_res_blocks being recorded."""
    path, root = _write_registry(
        tmp_path,
        {"m": {"path": "x.pt", "tensor_spec": 1, "in_channels": 9}},
        {"5x5_2p": {"model": "m", "max_walls": 3}},
    )
    spec = variant_spec(5, 2, registry=load_registry(path), root=root)
    assert (spec.num_channels, spec.num_res_blocks, spec.max_turns) == (64, 4, 300)


@pytest.mark.parametrize("board_size,num_players", [(5, 2), (5, 4), (9, 2), (9, 4)])
def test_shipped_registry_builds_a_loadable_pair(board_size, num_players):
    """Every variant in runs/MODELS.json builds an env and a model that agree.

    Does not load weights — that needs the checkpoints on disk, which a clone
    without the run dirs will not have.
    """
    spec = variant_spec(board_size, num_players)
    env = build_env(spec)
    model = build_model(spec, env.action_space_size)

    assert env.board_size == board_size
    assert env.num_players == num_players
    assert env.max_walls_per_player == spec.max_walls
    assert env.spec_version == spec.tensor_spec
    # The tensor the env produces must match the planes the net expects.
    assert env.state_to_tensor(env.reset()).shape[-1] == spec.in_channels
    assert model.network.policy_fc.out_features == env.action_space_size


def test_wall_candidates_defaults_to_unrestricted(tmp_path):
    """Entries that predate the restriction must not silently acquire one."""
    path, root = _write_registry(
        tmp_path,
        {"m": {"path": "x.pt", "tensor_spec": 1, "in_channels": 9}},
        {"5x5_2p": {"model": "m", "max_walls": 3}},
    )
    spec = variant_spec(5, 2, registry=load_registry(path), root=root)
    assert spec.wall_candidates == 0


def test_wall_candidates_comes_from_the_variant(tmp_path):
    path, root = _write_registry(
        tmp_path,
        {"m": {"path": "x.pt", "tensor_spec": 2, "in_channels": 9}},
        {"9x9_2p": {"model": "m", "max_walls": 10, "wall_candidates": 16}},
    )
    spec = variant_spec(9, 2, registry=load_registry(path), root=root)
    assert spec.wall_candidates == 16


@pytest.mark.parametrize("board_size,num_players,expected", [
    (5, 2, 0), (5, 4, 0), (9, 2, 16), (9, 4, 16),
])
def test_shipped_registry_serves_each_model_at_its_measured_best_K(
        board_size, num_players, expected):
    """K is the budget each checkpoint scores highest under, not the one it
    trained under — those coincide for v9/v10 but not for the served 9x9 N=2
    model, n2_9x9_v4, which trained at K=0 and scores 70.0% there against
    97.5% at K=16. Serving the wrong K fails silently: the demo plays badly.
    5x5 scores best unrestricted.
    """
    assert variant_spec(board_size, num_players).wall_candidates == expected


@pytest.mark.parametrize("board_size,num_players,expected", [
    (5, 2, 150), (5, 4, 150), (9, 2, 160), (9, 4, 320),
])
def test_shipped_registry_serves_each_model_at_its_training_horizon(
        board_size, num_players, expected):
    """max_turns counts PLIES, the same unit as training's max_game_moves.

    Values are each run's own horizon (runs/<run>/config.json, untracked so
    they are literals here). It is set per variant to give every player the
    same ~80-ply budget, which is why N=4 is double N=2 rather than equal: a
    demo served below its run's horizon calls a timeout on games the measured
    configuration had room to convert.
    """
    assert variant_spec(board_size, num_players).max_turns == expected
