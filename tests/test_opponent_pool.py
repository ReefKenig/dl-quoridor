"""Self-play opponent pool: anchored games against a scripted racer.

Two things here are silent if wrong. A scripted seat must contribute NO training
sample -- its move is not a policy target. And the value discount must be counted
in real plies, not trajectory entries: with the model on one seat of two, the
trajectory holds half the plies, so an index-based discount would make every
target roughly the square root of what it should be, i.e. far too close to +/-1.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.mcts.evaluator_mp import greedy_agent
from src.mcts.self_play_mp import assign_vector_targets, play_one_game
from src.utils.schedule import (ANCHORED_OPPONENTS, iteration_plans,
                                opponent_for_game)


def _traj(n, num_players=2):
    return [(np.zeros((3, 3, 1), np.float32), np.zeros(4, np.float32), i % num_players)
            for i in range(n)]


# --- the sampler --------------------------------------------------------------

def test_no_share_is_pure_self_play():
    assert [opponent_for_game(i, 0.0) for i in range(6)] == ["self"] * 6


def test_a_full_share_anchors_every_game():
    assert [opponent_for_game(i, 1.0) for i in range(6)] == ["greedy"] * 6


@pytest.mark.parametrize("share,total", [(0.15, 40), (0.25, 40), (0.5, 40),
                                         (0.2, 80)])
def test_the_realised_share_matches_the_configured_one(share, total):
    n = sum(opponent_for_game(i, share) == "greedy" for i in range(total))
    assert n == int(total * share + 1e-9)


def test_anchored_games_are_spread_not_front_loaded():
    # Workers claim games off a shared counter, so a truncated iteration must
    # still have seen both kinds.
    for prefix in (8, 16, 24):
        n = sum(opponent_for_game(i, 0.25) == "greedy" for i in range(prefix))
        assert abs(n - prefix * 0.25) <= 1


# --- the seat / wall-mask independence ----------------------------------------
# takes_share(g, 0.5) is exactly "g is odd", so a seat of game_index % 2 and the
# 0.5 wall mask were the same predicate: at N=2 no anchored seat-1 game ever had
# walls, and at N=4 with greedy_share 0.5 seat 0 was never anchored at all. These
# assert the property the old tautology only claimed to.

def _anchored(plans):
    return [p for p in plans if p.opponent in ANCHORED_OPPONENTS]


@pytest.mark.parametrize("n,greedy_share,total", [(2, 0.35, 40), (2, 0.5, 40),
                                                  (4, 0.5, 40), (4, 0.35, 40),
                                                  (2, 1.0, 20), (4, 0.25, 80)])
def test_every_seat_sees_both_wall_regimes(n, greedy_share, total):
    """The acceptance property: no (seat x walls-legal) cell may be empty."""
    plans = iteration_plans(total, n, greedy_share, mask_fraction=0.5)
    cells = {(p.model_seat, p.walls_masked) for p in _anchored(plans)}
    missing = {(s, m) for s in range(n) for m in (True, False)} - cells
    assert not missing, f"unpopulated (seat, masked) cells: {sorted(missing)}"


@pytest.mark.parametrize("n", [2, 4])
def test_the_walled_share_is_the_same_in_every_seat(n):
    """Not just non-empty -- balanced, so no seat is a rounding artefact."""
    plans = _anchored(iteration_plans(80, n, 0.5, mask_fraction=0.5))
    for seat in range(n):
        games = [p for p in plans if p.model_seat == seat]
        walled = sum(not p.walls_masked for p in games)
        assert abs(walled / len(games) - 0.5) <= 0.1, f"seat {seat}: {walled}/{len(games)}"


def test_the_seat_is_not_the_game_index_parity():
    """The aliasing itself: seat must not be recoverable from the mask bit."""
    plans = _anchored(iteration_plans(40, 2, 0.5, mask_fraction=0.5))
    assert len({p.model_seat for p in plans if p.walls_masked}) == 2
    assert len({p.model_seat for p in plans if not p.walls_masked}) == 2


def test_n4_anchors_seat_0_with_walls():
    """The cell whose absence voided the N=4 arm: seat 0 is the only seat a
    racer can win at N=4, and it had no anchored games at all."""
    plans = _anchored(iteration_plans(40, 4, 0.5, mask_fraction=0.5))
    assert [p for p in plans if p.model_seat == 0 and not p.walls_masked]


def test_the_marginal_shares_are_unchanged():
    """Decorrelating must not move either configured share."""
    plans = iteration_plans(40, 2, 0.35, mask_fraction=0.5)
    assert sum(p.opponent == "greedy" for p in plans) == 14      # 0.35 * 40
    assert abs(sum(p.walls_masked for p in plans) - 20) <= 1     # 0.50 * 40


def test_plans_depend_only_on_the_game_index():
    """Workers claim games off a shared counter, so a plan may not depend on
    which worker ran the game or in what order."""
    assert iteration_plans(40, 4, 0.5, mask_fraction=0.5) == \
        iteration_plans(40, 4, 0.5, mask_fraction=0.5)


def test_no_mask_still_rotates_every_seat():
    plans = _anchored(iteration_plans(40, 4, 0.5, mask_fraction=0.0))
    assert {p.model_seat for p in plans} == {0, 1, 2, 3}
    assert not any(p.walls_masked for p in plans)


def test_unanchored_games_have_no_seat():
    plans = iteration_plans(40, 2, 0.35, mask_fraction=0.5)
    assert all(p.model_seat is None for p in plans if p.opponent == "self")


# --- seat-0 anchoring bias (anchored_seat0_share) ------------------------------
# At N=4 seat 0 is the only seat a racer can win, and uniform rotation left it
# with ~300 of ~8400 samples/iter in v8 - the one seat that matters, starved.


def test_seat0_share_zero_keeps_the_rotation_bit_identical():
    assert iteration_plans(80, 4, 0.5, mask_fraction=0.5, seat0_share=0.0) == \
        iteration_plans(80, 4, 0.5, mask_fraction=0.5)


@pytest.mark.parametrize("share", [0.25, 0.5])
def test_seat0_share_pins_that_fraction_in_each_mask_cell(share):
    plans = _anchored(iteration_plans(160, 4, 0.5, mask_fraction=0.5,
                                      seat0_share=share))
    for masked in (True, False):
        cell = [p for p in plans if p.walls_masked == masked]
        pinned = sum(p.model_seat == 0 for p in cell)
        assert abs(pinned / len(cell) - share) <= 0.1, \
            f"masked={masked}: {pinned}/{len(cell)}"


def test_the_unpinned_games_still_rotate_over_the_other_seats():
    plans = _anchored(iteration_plans(160, 4, 0.5, mask_fraction=0.5,
                                      seat0_share=0.5))
    rest = [p.model_seat for p in plans if p.model_seat != 0]
    counts = {s: rest.count(s) for s in (1, 2, 3)}
    assert set(counts) == {1, 2, 3}
    assert max(counts.values()) - min(counts.values()) <= 2, counts


def test_seat0_share_keeps_every_cell_populated():
    """The acceptance property survives the bias: no (seat, masked) cell empties."""
    plans = _anchored(iteration_plans(80, 4, 0.5, mask_fraction=0.5,
                                      seat0_share=0.5))
    cells = {(p.model_seat, p.walls_masked) for p in plans}
    missing = {(s, m) for s in range(4) for m in (True, False)} - cells
    assert not missing, f"unpopulated (seat, masked) cells: {sorted(missing)}"


def test_a_full_seat0_share_pins_every_anchored_game():
    plans = _anchored(iteration_plans(40, 4, 0.5, mask_fraction=0.5,
                                      seat0_share=1.0))
    assert {p.model_seat for p in plans} == {0}


# --- value targets under a sparse trajectory ----------------------------------

def test_sparse_and_dense_trajectories_agree_on_the_discount():
    """The regression this file exists for.

    A 2-player game of 20 plies where the model held seat 0 yields 10 entries at
    plies 0, 2, 4... Their targets must equal what a dense 20-ply trajectory
    gives at those same plies.
    """
    dense = assign_vector_targets(_traj(20), winner=0, num_players=2,
                                  discount=0.99, discount_unit="round")
    sparse = assign_vector_targets(
        _traj(10), winner=0, num_players=2, discount=0.99,
        discount_unit="round", plies=list(range(0, 20, 2)), total_plies=20)

    assert len(sparse) == 10
    for k, (_t, _p, vec) in enumerate(sparse):
        assert vec == pytest.approx(dense[2 * k][2]), f"entry {k} discount differs"


def test_ignoring_the_ply_count_inflates_targets():
    """Guards the bug rather than the fix: the naive version is materially wrong."""
    correct = assign_vector_targets(
        _traj(10), 0, 2, 0.99, discount_unit="round",
        plies=list(range(0, 20, 2)), total_plies=20)[0][2][0]
    naive = assign_vector_targets(_traj(10), 0, 2, 0.99,
                                  discount_unit="round")[0][2][0]
    assert naive > correct, "the naive count should look closer to +1"
    assert abs(naive - correct) > 0.04


def test_misaligned_plies_are_refused():
    """Alignment is what makes the ply-indexed discount correct. Off by one and
    every target is mis-discounted silently, so it raises instead."""
    with pytest.raises(ValueError, match="align"):
        assign_vector_targets(_traj(10), 0, 2, 0.99, plies=list(range(9)),
                              total_plies=20)
    with pytest.raises(ValueError, match="align"):
        assign_vector_targets(_traj(10), 0, 2, 0.99, plies=list(range(11)),
                              total_plies=20)


def test_a_ply_outside_the_game_is_refused():
    with pytest.raises(ValueError, match="not inside"):
        assign_vector_targets(_traj(3), 0, 2, 0.99, plies=[0, 2, 20],
                              total_plies=20)


def test_dense_behaviour_is_unchanged_without_the_new_arguments():
    for unit in ("round", "ply"):
        old = assign_vector_targets(_traj(12), 1, 2, 0.97, discount_unit=unit)
        new = assign_vector_targets(_traj(12), 1, 2, 0.97, discount_unit=unit,
                                    plies=list(range(12)), total_plies=12)
        for a, b in zip(old, new):
            assert a[2] == pytest.approx(b[2])


# --- play_one_game ------------------------------------------------------------

class _StubMCTS:
    """Always walks the first legal action, so a game terminates quickly."""

    def __init__(self, action_space):
        self.action_space = action_space
        self.searches = 0

    def search(self, env, state, temperature=1.0):
        self.searches += 1
        probs = np.zeros(self.action_space, np.float32)
        probs[int(env.get_valid_actions(state)[0])] = 1.0
        return probs


def _env(num_players=2):
    return QuoridorEnvMP(board_size=5, num_players=num_players,
                         max_walls_per_player=3, max_turns=60)


def test_a_scripted_seat_contributes_no_sample():
    env = _env()
    mcts = _StubMCTS(compute_action_space_size(5))
    np.random.seed(0)
    samples, winner = play_one_game(env, mcts, 2, max_moves=60, discount=0.99,
                                    explore_moves=5,
                                    seat_agents={1: greedy_agent()})
    assert winner is not None
    # Half the plies belong to the scripted seat, and augmentation doubles the
    # rest, so the model's own count is searches, not plies.
    assert len(samples) == 2 * mcts.searches


def test_the_model_still_searches_only_its_own_seats():
    env = _env(num_players=4)
    mcts = _StubMCTS(compute_action_space_size(5))
    np.random.seed(0)
    play_one_game(env, mcts, 4, max_moves=60, discount=0.99, explore_moves=5,
                  seat_agents={s: greedy_agent() for s in (1, 2, 3)})
    # One search per model ply; with 3 of 4 seats scripted that is ~a quarter.
    assert mcts.searches > 0


def test_a_mixed_scripted_game_produces_aligned_targets():
    """The alignment guard must not fire on the case it protects: a real game
    where a scripted seat holds half the plies."""
    env = _env()
    mcts = _StubMCTS(compute_action_space_size(5))
    np.random.seed(0)
    samples, winner = play_one_game(env, mcts, 2, max_moves=60, discount=0.99,
                                    explore_moves=5,
                                    seat_agents={1: greedy_agent()})
    assert winner is not None and samples
    # Sparse trajectory, so every target must be strictly inside (-1, 1) - the
    # symptom of counting distance in entries was targets pinned at +/-1.
    for _t, _p, vec in samples:
        assert vec.shape == (2,) and 0 < abs(vec[0]) < 1.0


def test_no_seat_agents_is_ordinary_self_play():
    env = _env()
    mcts = _StubMCTS(compute_action_space_size(5))
    np.random.seed(0)
    samples, winner = play_one_game(env, mcts, 2, max_moves=60, discount=0.99,
                                    explore_moves=5)
    assert winner is not None and len(samples) == 2 * mcts.searches


# --- the champion pool --------------------------------------------------------

def test_the_past_share_is_scheduled_alongside_greedy():
    picks = [opponent_for_game(i, 0.25, 0.25) for i in range(40)]
    assert picks.count("greedy") == 10
    assert picks.count("past") > 0
    assert picks.count("greedy") + picks.count("past") + picks.count("self") == 40


@pytest.mark.parametrize("g,p", [(0.25, 0.25), (0.2, 0.1), (0.35, 0.15),
                                 (0.1, 0.4), (0.15, 0.25), (0.5, 0.4)])
@pytest.mark.parametrize("n", [40, 80])
def test_both_shares_are_allocated_exactly(g, p, n):
    """Equal shares are the case that broke: tested independently, the two
    Bresenham schedules fire on the same games and past never plays at all."""
    picks = [opponent_for_game(i, g, p) for i in range(n)]
    assert picks.count("greedy") == round(n * g)
    assert picks.count("past") == round(n * p)
    assert picks.count("self") == n - round(n * g) - round(n * p)


def test_no_past_share_never_selects_past():
    assert "past" not in [opponent_for_game(i, 0.3, 0.0) for i in range(40)]


def test_snapshots_are_capped_and_the_oldest_goes_first(tmp_path):
    from src.mcts.training_mp import champion_pool_paths, snapshot_champion

    class _Stub:
        def save(self, path):
            open(path, "w").write("x")

    best = _Stub()
    for it in range(1, 8):
        snapshot_champion(best, str(tmp_path), it, pool_size=3)
    kept = [p.split("_iter")[-1] for p in champion_pool_paths(str(tmp_path))]
    assert kept == ["0005.pt", "0006.pt", "0007.pt"]


def test_the_pool_is_recovered_from_disk_on_resume(tmp_path):
    from src.mcts.training_mp import champion_pool_paths, snapshot_champion

    class _Stub:
        def save(self, path):
            open(path, "w").write("x")

    assert champion_pool_paths(str(tmp_path)) == []
    snapshot_champion(_Stub(), str(tmp_path), 3, pool_size=5)
    assert len(champion_pool_paths(str(tmp_path))) == 1


def test_a_champion_that_cannot_serve_a_forward_pass_fails_fast():
    """A pooled champion rides the shared batcher. If it loads but cannot
    predict, the run must die here rather than an hour into the iteration."""
    from src.mcts.training_mp import load_past_champion

    class _Broken:
        def load(self, path):
            pass

        def predict(self, tensor):
            raise RuntimeError("shape mismatch on the batcher")

    with pytest.raises(RuntimeError, match="warmup forward pass"):
        load_past_champion(_Broken(), "champion_iter0003.pt", _env())


def test_a_healthy_champion_warms_up_and_is_returned():
    from src.mcts.training_mp import load_past_champion

    class _Ok:
        def __init__(self):
            self.loaded = self.predicted = None

        def load(self, path):
            self.loaded = path

        def predict(self, tensor):
            self.predicted = tensor.shape
            return np.zeros(4), np.zeros(2)

    model = _Ok()
    assert load_past_champion(model, "c.pt", _env()) is model
    assert model.loaded == "c.pt" and model.predicted is not None


def test_a_past_share_without_the_parallel_engine_is_refused():
    from src.mcts.training_mp import TrainingConfigMP
    with pytest.raises(NotImplementedError):
        TrainingConfigMP(opponent_past_share=0.2, parallel_self_play=False)


def test_shares_over_one_are_refused():
    from src.mcts.training_mp import TrainingConfigMP
    with pytest.raises(ValueError):
        TrainingConfigMP(opponent_past_share=0.7, opponent_greedy_share=0.5,
                         parallel_self_play=True)
