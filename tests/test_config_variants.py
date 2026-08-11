"""Per-variant settings, and the ply-budget invariant that must hold between them.

`max_game_moves` counts individual plies, not rounds. A single shared value
therefore gives each N=2 player twice the budget of each N=4 player on the same
board: 160 plies is 80 turns each at N=2 but only 40 each at N=4. At 9x9, where
the shortest path is 8 steps and 20 walls are in play, that produced an 84-90%
timeout rate, all-zero value targets, and a collapsed value head.
"""
import json
from pathlib import Path

import pytest

from src.utils.config import (ply_budget_per_player, resolve_run_config,
                              resolve_variant)

CONFIG_9X9 = Path("configs/config_9x9.json")


@pytest.fixture(scope="module")
def cfg9():
    return json.loads(CONFIG_9X9.read_text())


def test_max_game_moves_resolves_per_variant(cfg9):
    assert resolve_variant(cfg9, "n2")["max_game_moves"] == 160
    assert resolve_variant(cfg9, "n4")["max_game_moves"] == 320


def test_ply_budget_per_player_is_equal_across_variants(cfg9):
    """The invariant. If this fails, one player count is being starved.

    Encoded as an executable rule precisely because the original bug was
    invisible: a single shared 160 looks symmetric until you notice the unit is
    plies and N differs.
    """
    n2 = ply_budget_per_player(cfg9, "n2")
    n4 = ply_budget_per_player(cfg9, "n4")

    assert n2 == n4 == 80.0, (
        f"per-player ply budget differs: n2={n2}, n4={n4}. max_game_moves must "
        f"scale with num_players.")


def test_explore_moves_per_player_is_equal_across_variants(cfg9):
    """explore_moves counts plies too, so sharing it starves N=4 the same way.

    Temperature sampling runs for the first `explore_moves` plies. A shared 20
    is ~10 exploratory turns each at N=2 but only 5 at N=4 — a quarter of the
    self-play diversity, on the harder problem.
    """
    n2 = resolve_run_config(cfg9, "n2")
    n4 = resolve_run_config(cfg9, "n4")
    per_player = (n2["explore_moves"] / n2["num_players"],
                  n4["explore_moves"] / n4["num_players"])

    assert per_player[0] == per_player[1], (
        f"per-player exploration differs: n2={per_player[0]}, n4={per_player[1]}. "
        f"explore_moves must scale with num_players.")


def test_n4_cap_is_not_the_old_shared_value(cfg9):
    """Pins the regression: 160 at N=4 is the setting that caused the collapse."""
    assert resolve_variant(cfg9, "n4")["max_game_moves"] != 160


def test_variant_overrides_shadow_the_shared_training_block():
    raw = {"training": {"max_game_moves": 160, "batch_size": 128},
           "variants": {"n4": {"num_players": 4, "max_game_moves": 320}}}

    merged = resolve_variant(raw, "n4")

    assert merged["max_game_moves"] == 320     # overridden
    assert merged["batch_size"] == 128         # inherited


def test_shared_value_is_used_when_a_variant_does_not_override():
    raw = {"training": {"max_game_moves": 160},
           "variants": {"n2": {"num_players": 2}}}

    assert resolve_variant(raw, "n2")["max_game_moves"] == 160


def test_unknown_variant_falls_back_to_the_shared_block():
    raw = {"training": {"max_game_moves": 200}, "variants": {}}

    assert resolve_variant(raw, "n8")["max_game_moves"] == 200


def test_ply_budget_requires_num_players():
    raw = {"training": {"max_game_moves": 160}, "variants": {"n2": {}}}

    with pytest.raises(KeyError):
        ply_budget_per_player(raw, "n2")


def test_other_variant_settings_still_resolve(cfg9):
    """Guards against the merge dropping the fields that already lived there."""
    # 4-player Quoridor is 5 walls each; keep the game faithful.
    assert resolve_variant(cfg9, "n2")["max_walls_per_player"] == 10
    assert resolve_variant(cfg9, "n4")["max_walls_per_player"] == 5
    assert resolve_variant(cfg9, "n4")["num_players"] == 4
    for v in ("n2", "n4"):
        assert resolve_variant(cfg9, v)["eval_games"] > 0


def test_run_length_resolves_per_variant(cfg9):
    """num_iterations/games_per_iteration live only in the variant blocks.

    The 9x9 notebooks read them through resolve_run_config; if they were moved back
    to the shared `training` block both variants would silently train for the
    same number of games despite N=4 games costing ~3x as much.
    """
    for key in ("num_iterations", "games_per_iteration"):
        assert key not in cfg9["training"], f"{key} must stay per-variant"
        for v in ("n2", "n4"):
            assert resolve_variant(cfg9, v)[key] > 0


def test_run_config_flattens_every_section(cfg9):
    """One dict, so callers never choose which section to read from.

    The 9x9 notebooks previously read some settings per-section and others
    through the variant merge, which is how they came to ignore num_workers and
    inference_batch_size entirely while appearing to be configured.
    """
    rc = resolve_run_config(cfg9, "n4")

    assert rc["board_size"] == 9                  # top level
    assert rc["num_simulations"] == cfg9["mcts"]["num_simulations"]
    assert rc["num_channels"] == cfg9["network"]["num_channels"]
    # inference_batch_size, not num_workers: both variants override the worker
    # count so they can run concurrently inside one cpu quota.
    assert rc["inference_batch_size"] == cfg9["parallel"]["inference_batch_size"]
    assert rc["batch_size"] == cfg9["training"]["batch_size"]
    assert rc["num_players"] == 4                 # variant


def test_run_config_variant_overrides_win(cfg9):
    n2, n4 = resolve_run_config(cfg9, "n2"), resolve_run_config(cfg9, "n4")

    assert (n2["max_game_moves"], n4["max_game_moves"]) == (160, 320)
    assert (n2["num_iterations"], n4["num_iterations"]) == (60, 40)
    # A variant key beats the section it also lives in — the two runs share a
    # 16-cpu quota, so neither may take the parallel section's worker count.
    assert n2["num_workers"] == n4["num_workers"] == 8
    assert n2["num_workers"] != cfg9["parallel"]["num_workers"]
    assert n2["num_simulations"] == n4["num_simulations"]   # shared


def test_run_config_rejects_a_key_defined_twice():
    """Ambiguity must fail loudly rather than resolve by section order."""
    raw = {"mcts": {"batch_size": 16}, "training": {"batch_size": 128},
           "variants": {"n2": {}}}

    with pytest.raises(ValueError, match="batch_size"):
        resolve_run_config(raw, "n2")


def test_run_config_drops_note_keys(cfg9):
    """`_`-prefixed entries document the JSON; they are not settings."""
    rc = resolve_run_config(cfg9, "n2")

    assert not [k for k in rc if k.startswith("_")]
    assert any(k.startswith("_") for k in cfg9["training"]), "fixture lost its notes"


def test_no_root_noise_at_any_difficulty():
    """Dirichlet noise is a self-play exploration device, not a play-time one.

    At 9x9 eps=0.25 lifts opening wall prior mass from 0.00024 to 0.245, so a
    quarter of the demo's moves became near-random walls that search could not
    reject. Difficulty is simulations and temperature.
    """
    from src.utils.config import DIFFICULTY_SETTINGS

    noisy = {name: s["dirichlet_epsilon"]
             for name, s in DIFFICULTY_SETTINGS.items()
             if s["dirichlet_epsilon"]}
    assert not noisy, f"root noise at play time: {noisy}"


def test_hard_reproduces_the_measured_configuration():
    """Every reported number was scored at eps=0 and temperature ~0, so at
    least one difficulty must be able to reproduce it."""
    from src.utils.config import DIFFICULTY_SETTINGS

    hard = DIFFICULTY_SETTINGS["hard"]
    assert hard["dirichlet_epsilon"] == 0.0
    assert hard["temperature"] == 0.0
