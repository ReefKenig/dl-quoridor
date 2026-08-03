"""Wall curriculum: the self-play wall budget, and restoring it for eval.

Budget 0 removes 128 of 140 actions at 9x9, so a leak in either direction is
expensive and silent — a masked eval scores against the wrong game, and a
self-play iteration that failed to mask trains on the prior it was meant to fix.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.mcts.training_mp import TrainingConfigMP, _wall_budget, wall_budget_at
from src.utils.schedule import game_is_masked


def _num_wall_actions(board_size):
    return 2 * (board_size - 1) ** 2


# --- the budget as the env sees it -------------------------------------------

@pytest.mark.parametrize("board_size,num_players", [(5, 2), (9, 2), (5, 4)])
def test_a_zero_budget_offers_pawn_moves_only(board_size, num_players):
    masked = QuoridorEnvMP(board_size=board_size, num_players=num_players,
                           wall_budget=0)
    full = QuoridorEnvMP(board_size=board_size, num_players=num_players)

    masked_actions = masked.get_valid_actions(masked.reset())
    full_actions = full.get_valid_actions(full.reset())

    assert len(masked_actions) > 0
    # 12 is the pawn-move block; everything above it is a wall placement.
    assert masked_actions.max() < 12
    assert set(masked_actions) == set(a for a in full_actions if a < 12)
    assert len(full_actions) - len(masked_actions) == _num_wall_actions(board_size)


def test_a_partial_budget_still_offers_every_wall_placement():
    # The ramp limits how MANY walls a player may spend, not which squares —
    # otherwise the policy would be learning a different action space each time.
    env = QuoridorEnvMP(board_size=9, num_players=2, max_walls_per_player=10,
                        wall_budget=2)
    full = QuoridorEnvMP(board_size=9, num_players=2, max_walls_per_player=10)
    assert set(env.get_valid_actions(env.reset())) == \
        set(full.get_valid_actions(full.reset()))


def test_the_budget_is_what_a_player_actually_gets():
    env = QuoridorEnvMP(board_size=9, num_players=2, max_walls_per_player=10,
                        wall_budget=3)
    assert env.reset().walls_remaining == [3, 3]


def test_max_walls_stays_the_plane_scale_while_the_budget_moves():
    # walls-remaining is encoded as remaining/max_walls; if the ramp moved
    # max_walls the network would see a different scale every iteration.
    for budget in (0, 1, 5, 10):
        state = QuoridorEnvMP(board_size=9, num_players=2,
                              max_walls_per_player=10,
                              wall_budget=budget).reset()
        assert state.max_walls == 10


def test_a_budget_over_the_maximum_is_clamped():
    env = QuoridorEnvMP(board_size=9, num_players=2, max_walls_per_player=5,
                        wall_budget=99)
    assert env.reset().walls_remaining == [5, 5]


def test_no_budget_means_the_full_allowance():
    env = QuoridorEnvMP(board_size=9, num_players=2, max_walls_per_player=10)
    assert env.reset().walls_remaining == [10, 10]


def test_a_zero_budget_game_plays_to_a_winner_without_walls():
    env = QuoridorEnvMP(board_size=9, num_players=2, max_turns=200,
                        max_walls_per_player=10, wall_budget=0)
    state = env.reset()
    rng = np.random.default_rng(0)
    while not state.game_over:
        actions = env.get_valid_actions(state)
        assert len(actions) > 0, "a zero budget left a player with no legal action"
        state, _, _, _ = env.step(state, int(rng.choice(actions)))

    assert state.winner is not None, "a pure race should not hit the turn limit"
    assert not state.h_walls and not state.v_walls


# --- the schedule -------------------------------------------------------------

def test_the_mask_comes_first_then_one_wall_at_a_time():
    assert [wall_budget_at(i, 3, 1, 5) for i in range(11)] == \
        [0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5]


def test_each_step_is_held_for_ramp_hold_iterations():
    # hold=3: three iterations at each wall count before the next.
    assert [wall_budget_at(i, 0, 3, 3) for i in range(10)] == \
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]


def test_a_long_hold_may_never_reach_the_full_allowance():
    # A 150-iteration run holding 20 per step tops out at 7 of 10. Eval always
    # plays the full game regardless.
    assert wall_budget_at(149, 10, 20, 10) == 7


def test_the_ramp_never_skips_straight_back_to_full():
    # The relapse this exists to prevent: 0 walls one iteration, all of them
    # the next.
    sched = [wall_budget_at(i, 10, 20, 10) for i in range(150)]
    assert sched[9] == 0 and sched[10] == 1
    assert max(b - a for a, b in zip(sched, sched[1:])) == 1


def test_no_hold_reproduces_the_old_hard_switch():
    assert [wall_budget_at(i, 3, 0, 5) for i in range(5)] == [0, 0, 0, 5, 5]


def test_no_curriculum_at_all_is_always_full():
    assert [wall_budget_at(i, 0, 0, 5) for i in range(3)] == [5, 5, 5]


# --- the mixed curriculum -----------------------------------------------------
# Ramping the budget cannot reintroduce walls gradually (see
# test_a_partial_budget_still_offers_every_wall_placement: budget 1 and budget 10
# expose the same 131 actions). Mixing race-only games into every iteration is
# what keeps racing AND walled states in the buffer for the whole run.

def test_no_fraction_means_no_masked_games():
    assert [game_is_masked(g, 0.0) for g in range(5)] == [False] * 5


def test_a_full_fraction_masks_every_game():
    assert [game_is_masked(g, 1.0) for g in range(5)] == [True] * 5


@pytest.mark.parametrize("fraction,total", [(0.5, 40), (0.25, 40), (0.75, 40),
                                            (0.3, 160), (0.5, 7)])
def test_the_mix_hits_the_requested_share(fraction, total):
    # floor, so a half-game remainder is spent on walls rather than racing.
    n = sum(game_is_masked(g, fraction) for g in range(total))
    assert n == int(total * fraction + 1e-9)


def test_the_mix_is_spread_not_front_loaded():
    # Workers claim games off a shared counter, so an iteration that ends early
    # (or a worker that dies) must still have seen both kinds of game.
    fraction, total = 0.5, 40
    for prefix in (4, 10, 20):
        n = sum(game_is_masked(g, fraction) for g in range(prefix))
        assert abs(n - prefix * fraction) <= 1


def test_masked_and_walled_games_differ_in_the_action_space():
    # The point of the mix: both distributions reach the buffer every iteration.
    masked = QuoridorEnvMP(board_size=9, num_players=2, max_walls_per_player=10,
                           wall_budget=0)
    walled = QuoridorEnvMP(board_size=9, num_players=2, max_walls_per_player=10,
                           wall_budget=10)
    assert len(masked.get_valid_actions(masked.reset())) == 3
    assert len(walled.get_valid_actions(walled.reset())) == 131


def test_worker_config_carries_the_mix_fraction():
    from src.mcts.parallel_self_play_mp import generate_parallel_self_play_mp
    import inspect

    src = inspect.getsource(generate_parallel_self_play_mp)
    assert '"wall_mask_fraction"' in src
    assert TrainingConfigMP().wall_mask_fraction == 0.0


def test_the_worker_sets_the_budget_per_game():
    """The mix is per game, so the worker must re-set the budget inside its game
    loop rather than relying on the env it built once at startup."""
    from src.mcts import parallel_self_play_mp
    import inspect

    src = inspect.getsource(parallel_self_play_mp)
    assert "game_is_masked(game_index, mask_fraction)" in src
    assert "env.wall_budget = 0" in src


# --- self-play only -----------------------------------------------------------

def test_budget_restores_on_normal_exit():
    env = QuoridorEnvMP(board_size=5, num_players=2)
    cfg = TrainingConfigMP()

    with _wall_budget(cfg, env, 0):
        assert env.wall_budget == 0 and cfg.wall_budget == 0

    assert env.wall_budget is None and cfg.wall_budget is None


def test_budget_restores_when_self_play_raises():
    """Self-play aborts loudly on zero samples; the env must not stay masked,
    because the notebook hands the same env to the eval cells afterwards."""
    env = QuoridorEnvMP(board_size=5, num_players=2)
    cfg = TrainingConfigMP()

    with pytest.raises(RuntimeError):
        with _wall_budget(cfg, env, 0):
            raise RuntimeError("self-play produced no samples")

    assert env.wall_budget is None and cfg.wall_budget is None


def test_eval_always_gets_the_full_allowance():
    # A partial budget during self-play must not leak into the gate or the
    # greedy baseline, or the metrics stop being comparable across the run.
    env = QuoridorEnvMP(board_size=9, num_players=2, max_walls_per_player=10)
    cfg = TrainingConfigMP()
    with _wall_budget(cfg, env, 2):
        pass
    assert env.reset().walls_remaining == [10, 10]


def test_worker_config_carries_the_budget():
    """The budget reaches spawned self-play workers through the config dict,
    not through the parent's env object."""
    from src.mcts.parallel_self_play_mp import generate_parallel_self_play_mp
    import inspect

    src = inspect.getsource(generate_parallel_self_play_mp)
    assert '"wall_budget": getattr(cfg, "wall_budget", None)' in src

    cfg = TrainingConfigMP(board_size=9, num_players=2)
    cfg.wall_budget = 0
    env = QuoridorEnvMP(board_size=9, num_players=2,
                        wall_budget=getattr(cfg, "wall_budget", None))
    assert env.get_valid_actions(env.reset()).max() < 12
    assert compute_action_space_size(9) == 140
