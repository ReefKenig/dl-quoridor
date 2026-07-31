"""Wall curriculum: masking walls out of self-play, and restoring them for eval.

The mask removes 128 of 140 actions at 9x9, so a leak in either direction is
expensive and silent — a masked eval scores against the wrong game, and a
self-play iteration that failed to mask trains on the prior it was meant to fix.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.mcts.training_mp import TrainingConfigMP, _wall_mask


def _num_wall_actions(board_size):
    return 2 * (board_size - 1) ** 2


@pytest.mark.parametrize("board_size,num_players", [(5, 2), (9, 2), (5, 4)])
def test_masked_env_offers_pawn_moves_only(board_size, num_players):
    masked = QuoridorEnvMP(board_size=board_size, num_players=num_players,
                           walls_enabled=False)
    full = QuoridorEnvMP(board_size=board_size, num_players=num_players)

    masked_actions = masked.get_valid_actions(masked.reset())
    full_actions = full.get_valid_actions(full.reset())

    assert len(masked_actions) > 0
    # 12 is the pawn-move block; everything above it is a wall placement.
    assert masked_actions.max() < 12
    assert set(masked_actions) == set(a for a in full_actions if a < 12)
    assert len(full_actions) - len(masked_actions) == _num_wall_actions(board_size)


def test_masked_game_plays_to_a_winner_without_walls():
    env = QuoridorEnvMP(board_size=9, num_players=2, max_turns=200,
                        max_walls_per_player=10, walls_enabled=False)
    state = env.reset()
    rng = np.random.default_rng(0)
    while not state.game_over:
        actions = env.get_valid_actions(state)
        assert len(actions) > 0, "masked env left a player with no legal action"
        state, _, _, _ = env.step(state, int(rng.choice(actions)))

    assert state.winner is not None, "a pure race should not hit the turn limit"
    assert not state.h_walls and not state.v_walls


def test_wall_mask_restores_on_normal_exit():
    env = QuoridorEnvMP(board_size=5, num_players=2)
    cfg = TrainingConfigMP()

    with _wall_mask(cfg, env, masked=True):
        assert env.walls_enabled is False
        assert cfg.walls_enabled is False

    assert env.walls_enabled is True
    assert cfg.walls_enabled is True


def test_wall_mask_restores_when_self_play_raises():
    """Self-play aborts loudly on zero samples; the env must not stay masked,
    because the notebook hands the same env to the eval cells afterwards."""
    env = QuoridorEnvMP(board_size=5, num_players=2)
    cfg = TrainingConfigMP()

    with pytest.raises(RuntimeError):
        with _wall_mask(cfg, env, masked=True):
            raise RuntimeError("self-play produced no samples")

    assert env.walls_enabled is True
    assert cfg.walls_enabled is True


def test_unmasked_iteration_leaves_walls_enabled():
    env = QuoridorEnvMP(board_size=5, num_players=2)
    cfg = TrainingConfigMP()

    with _wall_mask(cfg, env, masked=False):
        assert env.walls_enabled is True

    assert env.walls_enabled is True


def test_worker_config_carries_the_mask():
    """The mask reaches spawned self-play workers through the config dict, not
    through the parent's env object."""
    from src.mcts.parallel_self_play_mp import generate_parallel_self_play_mp
    import inspect

    src = inspect.getsource(generate_parallel_self_play_mp)
    assert '"walls_enabled": getattr(cfg, "walls_enabled", True)' in src

    cfg = TrainingConfigMP(board_size=9, num_players=2)
    cfg.walls_enabled = False
    env = QuoridorEnvMP(board_size=9, num_players=2,
                        walls_enabled=getattr(cfg, "walls_enabled", True))
    assert env.get_valid_actions(env.reset()).max() < 12
    assert compute_action_space_size(9) == 140
