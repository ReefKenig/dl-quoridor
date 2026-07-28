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

from src.utils.config import (ply_budget_per_player, resolve_variant,
                              variant_setting)

CONFIG_9X9 = Path("configs/config_9x9.json")


@pytest.fixture(scope="module")
def cfg9():
    return json.loads(CONFIG_9X9.read_text())


def test_max_game_moves_resolves_per_variant(cfg9):
    assert variant_setting(cfg9, "n2", "max_game_moves") == 160
    assert variant_setting(cfg9, "n4", "max_game_moves") == 320


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


def test_n4_cap_is_not_the_old_shared_value(cfg9):
    """Pins the regression: 160 at N=4 is the setting that caused the collapse."""
    assert variant_setting(cfg9, "n4", "max_game_moves") != 160


def test_variant_overrides_shadow_the_shared_training_block():
    raw = {"training": {"max_game_moves": 160, "batch_size": 128},
           "variants": {"n4": {"num_players": 4, "max_game_moves": 320}}}

    merged = resolve_variant(raw, "n4")

    assert merged["max_game_moves"] == 320     # overridden
    assert merged["batch_size"] == 128         # inherited


def test_shared_value_is_used_when_a_variant_does_not_override():
    raw = {"training": {"max_game_moves": 160},
           "variants": {"n2": {"num_players": 2}}}

    assert variant_setting(raw, "n2", "max_game_moves") == 160


def test_unknown_variant_falls_back_to_the_shared_block():
    raw = {"training": {"max_game_moves": 200}, "variants": {}}

    assert variant_setting(raw, "n8", "max_game_moves") == 200


def test_ply_budget_requires_num_players():
    raw = {"training": {"max_game_moves": 160}, "variants": {"n2": {}}}

    with pytest.raises(KeyError):
        ply_budget_per_player(raw, "n2")


def test_other_variant_settings_still_resolve(cfg9):
    """Guards against the merge dropping the fields that already lived there."""
    assert variant_setting(cfg9, "n2", "max_walls_per_player") == 10
    assert variant_setting(cfg9, "n4", "max_walls_per_player") == 5
    assert variant_setting(cfg9, "n4", "num_players") == 4
    # 4-player Quoridor is 5 walls each; keep the game faithful.
    assert variant_setting(cfg9, "n4", "eval_games") == 40
