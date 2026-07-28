"""EvalResultMP win-rate accounting under draws.

Quoridor has no true draws — a draw is a timeout at max_game_moves, i.e. a game
that measured nothing. Counting those in the denominator made the N=4 gate
unpassable: at the observed 84-90% timeout rate the candidate's ceiling was ~10%
against a fair+margin bar of 28%.
"""
from src.mcts.evaluator_mp import EvalResultMP


def _result(num_players, games, wins, draws):
    return EvalResultMP(
        num_players=num_players,
        num_games=games,
        candidate_wins=wins,
        opponent_wins=games - wins - draws,
        draws=draws,
    )


def test_win_rate_is_over_decided_games():
    """40 games, 4 wins, 30 timeouts -> 4/10 decided, not 4/40."""
    r = _result(num_players=4, games=40, wins=4, draws=30)

    assert r.decided_games == 10
    assert r.candidate_win_rate == 0.4
    assert r.draw_rate == 0.75


def test_no_draws_behaves_as_before():
    """With zero draws the denominator is unchanged — no silent regression."""
    r = _result(num_players=2, games=40, wins=30, draws=0)

    assert r.decided_games == 40
    assert r.candidate_win_rate == 0.75


def test_all_draws_does_not_accept_and_does_not_divide_by_zero():
    r = _result(num_players=4, games=40, wins=0, draws=40)

    assert r.decided_games == 0
    assert r.candidate_win_rate == 0.0          # not ZeroDivisionError
    assert not r.should_accept(threshold=0.28)


def test_n4_high_draw_rate_can_now_accept():
    """The case that was structurally impossible before.

    16 decided games (40% of 40, above the min-decided floor), candidate takes 8
    of them = 50%, clearing the 0.28 bar. Under the old all-games denominator
    this scored 8/40 = 20% and was rejected regardless of strength.
    """
    r = _result(num_players=4, games=40, wins=8, draws=24)

    assert r.candidate_win_rate == 0.5
    assert r.should_accept(threshold=0.28)
    assert 8 / r.num_games < 0.28               # old rule would have rejected


def test_too_few_decided_games_cannot_promote():
    """A lucky single decided game reads as 100% — must not clear the gate."""
    r = _result(num_players=4, games=40, wins=1, draws=39)

    assert r.candidate_win_rate == 1.0
    assert not r.should_accept(threshold=0.28)


def test_min_decided_floor_is_the_boundary():
    """Exactly at the floor (25% of games decided) is allowed through."""
    at_floor = _result(num_players=4, games=40, wins=6, draws=30)
    below = _result(num_players=4, games=40, wins=6, draws=31)

    assert at_floor.decided_games == 10          # == 0.25 * 40
    assert at_floor.should_accept(threshold=0.28)
    assert not below.should_accept(threshold=0.28)


def test_summary_reports_decided_and_draw_rate():
    r = _result(num_players=4, games=40, wins=4, draws=30)

    s = r.summary()
    assert "decided 10" in s
    assert "75%" in s
