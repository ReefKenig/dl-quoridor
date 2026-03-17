"""
Evaluator Tests
================
Run: pytest tests/test_evaluator.py -v
"""

import logging
import pytest

from src.mcts.mcts import MCTS, MCTSConfig
from src.mcts.evaluator import (
    evaluate, evaluate_against_random,
    mcts_agent, random_agent, EvalResult, GameRecord,
)
from src.env.env_interface import MinimalQuoridorStub
from src.env.quoridor_env import QuoridorEnv

logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_mcts_vs_random():
    """MCTS agent should beat random agent with >80% win rate."""
    env = MinimalQuoridorStub()
    mcts = MCTS(config=MCTSConfig(num_simulations=400))
    agent_a = mcts_agent(mcts, temperature=0.1)

    result = evaluate_against_random(env, agent_a, num_games=50)
    assert result.agent_a_win_rate > 0.80, f"Win rate {result.agent_a_win_rate:.1%}"


def test_strong_vs_weak_mcts():
    """More simulations should beat fewer simulations (or tie on trivial envs)."""
    env = MinimalQuoridorStub()
    strong = mcts_agent(
        MCTS(config=MCTSConfig(num_simulations=400)), temperature=0.1
    )
    weak = mcts_agent(
        MCTS(config=MCTSConfig(num_simulations=50)), temperature=0.1
    )

    result = evaluate(env, agent_a=strong, agent_b=weak, num_games=40)
    # On the trivial stub env, both may solve it perfectly.
    # Strong should at least not lose more than weak.
    assert result.agent_a_win_rate >= result.agent_b_win_rate


def test_side_alternation():
    """Agent A should play as both player 0 and player 1."""
    env = MinimalQuoridorStub()
    agent = mcts_agent(
        MCTS(config=MCTSConfig(num_simulations=100)), temperature=0.1
    )

    result = evaluate(env, agent_a=agent, agent_b=random_agent(), num_games=20)
    sides = {g.agent_a_player for g in result.games}
    assert 0 in sides and 1 in sides


def test_should_accept():
    """EvalResult.should_accept threshold logic."""
    dummy = GameRecord(winner=0, num_moves=10,
                       duration_s=0.1, agent_a_player=0)
    r = EvalResult(agent_a_wins=60, agent_b_wins=40, draws=0)
    r.games = [dummy] * 100

    assert r.should_accept(threshold=0.55)
    assert not r.should_accept(threshold=0.65)


@pytest.mark.slow
def test_mcts_vs_random_real_env():
    """MCTS agent should beat random on real 5x5 QuoridorEnv."""
    env = QuoridorEnv(is_poc=True)
    mcts = MCTS(config=MCTSConfig(num_simulations=200))
    agent_a = mcts_agent(mcts, temperature=0.1)

    result = evaluate_against_random(env, agent_a, num_games=20)
    assert result.agent_a_win_rate > 0.50, (
        f"Win rate on real env {result.agent_a_win_rate:.1%} < 50%"
    )
