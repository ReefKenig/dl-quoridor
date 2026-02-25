"""
Model Evaluator
================
Owner: Iris

Pits two agents against each other over N games to determine
if a new model should replace the current best.

Used in the training loop after each iteration to gate model updates.
Also used standalone to benchmark against baseline agents (random, minimax).

Usage:
    from src.mcts.evaluator import evaluate, EvalResult

    # Compare two MCTS agents (different models or different sim counts)
    result = evaluate(
        env=env,
        agent_a=mcts_new,
        agent_b=mcts_old,
        num_games=40,
    )
    if result.should_accept(threshold=0.55):
        # Accept new model
        ...
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, List

from src.mcts.mcts import MCTS

logger = logging.getLogger(__name__)


# An agent is anything that takes (env, state) and returns an action.
# This lets us evaluate MCTS vs random, MCTS vs MCTS, MCTS vs minimax, etc.
AgentFn = Callable  # (env, state) -> int


def mcts_agent(mcts: MCTS, temperature: float = 0.1) -> AgentFn:
    """Wrap an MCTS instance as an AgentFn."""
    def agent(env, state) -> int:
        probs = mcts.search(env, state, temperature=temperature)
        return int(np.argmax(probs))
    return agent


def random_agent() -> AgentFn:
    """Baseline: uniform random over legal actions."""
    def agent(env, state) -> int:
        valid = env.get_valid_actions(state)
        return int(np.random.choice(valid))
    return agent


@dataclass
class GameRecord:
    """Record of a single evaluation game."""
    winner: Optional[int]       # 0, 1, or None (draw)
    num_moves: int
    duration_s: float
    agent_a_player: int         # which side agent_a played (0 or 1)


@dataclass
class EvalResult:
    """Aggregated results from an evaluation run."""
    games: List[GameRecord] = field(default_factory=list)
    agent_a_wins: int = 0
    agent_b_wins: int = 0
    draws: int = 0

    @property
    def num_games(self) -> int:
        return len(self.games)

    @property
    def agent_a_win_rate(self) -> float:
        if self.num_games == 0:
            return 0.0
        return self.agent_a_wins / self.num_games

    @property
    def agent_b_win_rate(self) -> float:
        if self.num_games == 0:
            return 0.0
        return self.agent_b_wins / self.num_games

    @property
    def avg_game_length(self) -> float:
        if self.num_games == 0:
            return 0.0
        return sum(g.num_moves for g in self.games) / self.num_games

    @property
    def avg_duration_s(self) -> float:
        if self.num_games == 0:
            return 0.0
        return sum(g.duration_s for g in self.games) / self.num_games

    def should_accept(self, threshold: float = 0.55) -> bool:
        """Should agent_a replace agent_b?"""
        return self.agent_a_win_rate > threshold

    def summary(self) -> str:
        return (
            f"Games: {self.num_games} | "
            f"A wins: {self.agent_a_wins} ({self.agent_a_win_rate:.1%}) | "
            f"B wins: {self.agent_b_wins} ({self.agent_b_win_rate:.1%}) | "
            f"Draws: {self.draws} | "
            f"Avg moves: {self.avg_game_length:.1f} | "
            f"Avg time: {self.avg_duration_s:.2f}s"
        )


def evaluate(
    env,
    agent_a: AgentFn,
    agent_b: AgentFn,
    num_games: int = 40,
    max_moves: int = 500,
    verbose: bool = True,
) -> EvalResult:
    """
    Play num_games between agent_a and agent_b.

    Sides alternate: agent_a plays as player 0 for half the games,
    player 1 for the other half. This eliminates first-move advantage bias.

    Args:
        env: QuoridorEnvInterface
        agent_a: the candidate (new model)
        agent_b: the incumbent (current best)
        num_games: total games to play (should be even)
        max_moves: safety cap per game
        verbose: print progress

    Returns:
        EvalResult with per-game records and aggregate stats
    """
    result = EvalResult()

    for game_idx in range(num_games):
        # Alternate sides: agent_a is player 0 for first half, player 1 for second
        agent_a_player = 0 if game_idx < num_games // 2 else 1
        agents = {agent_a_player: agent_a, 1 - agent_a_player: agent_b}

        state = env.reset()
        start_time = time.perf_counter()
        num_moves = 0
        done = False
        info: dict = {}

        while num_moves < max_moves:
            current_player = env.get_current_player(state)
            agent = agents[current_player]
            action = agent(env, state)
            state, reward, done, info = env.step(state, action)
            num_moves += 1

            if done:
                break

        duration = time.perf_counter() - start_time
        winner = info.get("winner") if done else None

        record = GameRecord(
            winner=winner,
            num_moves=num_moves,
            duration_s=duration,
            agent_a_player=agent_a_player,
        )
        result.games.append(record)

        # Attribute win
        if winner is None:
            result.draws += 1
        elif winner == agent_a_player:
            result.agent_a_wins += 1
        else:
            result.agent_b_wins += 1

        if verbose and (game_idx + 1) % 10 == 0:
            logger.info(
                "  [%d/%d] A: %d | B: %d | Draw: %d",
                game_idx + 1, num_games,
                result.agent_a_wins, result.agent_b_wins, result.draws,
            )

    if verbose:
        logger.info("Eval complete: %s", result.summary())

    return result


def evaluate_against_random(
    env,
    agent: AgentFn,
    num_games: int = 100,
    verbose: bool = True,
) -> EvalResult:
    """Convenience: evaluate an agent against the random baseline."""
    return evaluate(
        env=env,
        agent_a=agent,
        agent_b=random_agent(),
        num_games=num_games,
        verbose=verbose,
    )
