"""
N-player evaluation harness.

A candidate agent is rotated through all N seats; the remaining seats are filled
by an `opponent` AgentFn (reused — agents are stateless per search). Fair share
for the candidate is 1/N, so `should_accept` compares the candidate win-rate to a
threshold that should sit above 1/N (the training loop sets it per N).

At N=2 this reduces to the usual candidate-vs-opponent duel (fair share 0.5).
"""
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)
AgentFn = Callable[[Any, Any], int]   # (env, state) -> action


def mcts_agent_mp(mcts, temperature: float = 0.1) -> AgentFn:
    def agent(env, state) -> int:
        probs = mcts.search(env, state, temperature=temperature)
        return int(np.argmax(probs))
    return agent


def random_agent() -> AgentFn:
    def agent(env, state) -> int:
        return int(np.random.choice(env.get_valid_actions(state)))
    return agent


@dataclass
class EvalResultMP:
    num_players: int
    num_games: int = 0
    candidate_wins: int = 0
    opponent_wins: int = 0
    draws: int = 0
    # candidate wins per seat it sat in
    seat_wins: dict = field(default_factory=dict)
    games_per_seat: dict = field(default_factory=dict)

    # An eval whose decided games fall below this fraction of games played is too
    # thin to gate on: at a 90% timeout rate a single lucky decided game reads as
    # a 100% win rate. Reject rather than promote on that evidence.
    MIN_DECIDED_FRACTION = 0.25

    @property
    def decided_games(self) -> int:
        """Games that produced a winner. Quoridor has no true draws — a draw here
        is a timeout at max_game_moves, i.e. a game that measured nothing."""
        return self.num_games - self.draws

    @property
    def draw_rate(self) -> float:
        return self.draws / self.num_games if self.num_games else 0.0

    @property
    def candidate_win_rate(self) -> float:
        """Win rate over *decided* games.

        Dividing by num_games counted every timeout as a candidate loss, which
        capped the achievable rate at (1 - draw_rate). At N=4's observed 84-90%
        timeout rate that ceiling was ~10% against a `fair + margin` bar of 28%,
        so the gate could not be cleared at any strength.
        """
        return (self.candidate_wins / self.decided_games
                if self.decided_games else 0.0)

    @property
    def fair_share(self) -> float:
        return 1.0 / self.num_players

    def should_accept(self, threshold: float) -> bool:
        min_decided = self.num_games * self.MIN_DECIDED_FRACTION
        if self.decided_games < min_decided or self.decided_games == 0:
            return False
        return self.candidate_win_rate > threshold

    def summary(self) -> str:
        per_seat = " ".join(
            f"s{seat}:{self.seat_wins.get(seat, 0)}/{self.games_per_seat.get(seat, 0)}"
            for seat in range(self.num_players)
        )
        return (f"games={self.num_games} (decided {self.decided_games}) | "
                f"candidate {self.candidate_wins} "
                f"({self.candidate_win_rate:.1%} of decided, "
                f"fair={self.fair_share:.1%}) | "
                f"opp {self.opponent_wins} | draws {self.draws} "
                f"({self.draw_rate:.0%}) | by-seat[{per_seat}]")


def evaluate_mp(env, candidate: AgentFn, opponent: AgentFn,
                num_games: int = 24, max_moves: int = 400,
                verbose: bool = False, on_progress=None) -> EvalResultMP:
    """Candidate rotates through seats 0..N-1; opponent fills the rest.

    on_progress: optional callback(games_done, total, result) invoked every few
    games so long eval phases don't look stuck (e.g. write to games.log).
    """
    N = env.num_players
    res = EvalResultMP(num_players=N)
    for g in range(num_games):
        cand_seat = g % N
        agents = {s: (candidate if s == cand_seat else opponent)
                  for s in range(N)}
        state = env.reset()
        winner = None
        for _ in range(max_moves):
            cp = env.get_current_player(state)
            action = agents[cp](env, state)
            state, _, done, info = env.step(state, action)
            if done:
                winner = info.get("winner")
                break
        res.num_games += 1
        res.games_per_seat[cand_seat] = res.games_per_seat.get(
            cand_seat, 0) + 1
        if winner is None:
            res.draws += 1
        elif winner == cand_seat:
            res.candidate_wins += 1
            res.seat_wins[cand_seat] = res.seat_wins.get(cand_seat, 0) + 1
        else:
            res.opponent_wins += 1
        if verbose and (g + 1) % max(1, num_games // 4) == 0:
            logger.info("  [%d/%d] %s", g + 1, num_games, res.summary())
        # Progress heartbeat every 5 games (and on the last one).
        if on_progress is not None and ((g + 1) % 5 == 0 or (g + 1) == num_games):
            on_progress(g + 1, num_games, res)
    return res


def evaluate_against_random_mp(env, candidate: AgentFn, num_games: int = 24,
                               max_moves: int = 400, on_progress=None) -> EvalResultMP:
    return evaluate_mp(env, candidate, random_agent(),
                       num_games=num_games, max_moves=max_moves,
                       on_progress=on_progress)
