"""
N-player evaluation harness.

A candidate agent is rotated through all N seats; the remaining seats are filled
by an `opponent` AgentFn (reused — agents are stateless per search). Fair share
for the candidate is 1/N, so `should_accept` compares the candidate win-rate to a
threshold that should sit above 1/N (the training loop sets it per N).

At N=2 this reduces to the usual candidate-vs-opponent duel (fair share 0.5).
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)
AgentFn = Callable[[Any, Any], int]   # (env, state) -> action


# Sampling the opening from the search's own visit distribution, rather than
# playing argmax from move one. See mcts_agent_mp for why.
DEFAULT_EVAL_OPENING_PLIES = 4
OPENING_TEMPERATURE = 1.0


def mcts_agent_mp(mcts, temperature: float = 0.1,
                  opening_plies: int = 0,
                  opening_temperature: float = OPENING_TEMPERATURE) -> AgentFn:
    """Argmax over visit counts, except for the first `opening_plies` moves.

    With `dirichlet_epsilon=0` and argmax selection, MCTS is a deterministic
    function of the position, so every eval game sharing a seat assignment
    replays move-for-move. A 40-game gating eval was therefore measuring N
    distinct games — 2 at N=2, 4 at N=4 — while reporting n=40.

    The opening plies are sampled from the visit distribution at a higher
    temperature (proportional to visits at 1.0), so games diverge immediately
    while every move still comes from the model's own search. Note that the
    default eval temperature of 0.1 raises counts to the 10th power, which is
    effectively one-hot — sampling *that* would not produce diversity.
    """
    def agent(env, state, ply: int = 0, rng=None) -> int:
        if rng is not None and ply < opening_plies:
            probs = mcts.search(env, state, temperature=opening_temperature)
            total = probs.sum()
            if total > 0:
                return int(rng.choice(len(probs), p=probs / total))
        probs = mcts.search(env, state, temperature=temperature)
        return int(np.argmax(probs))
    return agent


def random_agent() -> AgentFn:
    def agent(env, state, ply: int = 0, rng=None) -> int:
        r = np.random if rng is None else rng
        return int(r.choice(env.get_valid_actions(state)))
    return agent


def eval_rng(base_seed: int, game_index: int):
    """Per-game RNG, shared by the sequential and parallel eval paths.

    Keyed on the game index alone so an outcome never depends on which worker
    happened to run it, and an explicit RandomState rather than the global
    np.random so eval cannot be perturbed by unrelated seeding elsewhere.
    """
    return np.random.RandomState(base_seed + game_index)


def play_eval_game(env, agents, max_moves: int, rng=None) -> Optional[int]:
    """Play one evaluation game. Returns the winning seat, or None on timeout."""
    state = env.reset()
    for ply in range(max_moves):
        cp = env.get_current_player(state)
        action = agents[cp](env, state, ply, rng)
        state, _, done, info = env.step(state, action)
        if done:
            return info.get("winner")
    return None


def tally_game(res: "EvalResultMP", cand_seat: int, winner: Optional[int]):
    """Fold one game's outcome into a result. Shared so the sequential path, the
    parallel driver and the test reference cannot drift apart in their bookkeeping."""
    res.num_games += 1
    res.games_per_seat[cand_seat] = res.games_per_seat.get(cand_seat, 0) + 1
    if winner is None:
        res.draws += 1
    elif winner == cand_seat:
        res.candidate_wins += 1
        res.seat_wins[cand_seat] = res.seat_wins.get(cand_seat, 0) + 1
    else:
        res.opponent_wins += 1
    return res


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
                verbose: bool = False, on_progress=None,
                base_seed: int = 0) -> EvalResultMP:
    """Candidate rotates through seats 0..N-1; opponent fills the rest.

    on_progress: optional callback(games_done, total, result) invoked every few
    games so long eval phases don't look stuck (e.g. write to games.log).
    base_seed: per-game RNG seed base — drives both the random opponent and the
    sampled opening, so games are distinct but reproducible.
    """
    N = env.num_players
    res = EvalResultMP(num_players=N)
    for g in range(num_games):
        cand_seat = g % N
        agents = {s: (candidate if s == cand_seat else opponent)
                  for s in range(N)}
        winner = play_eval_game(env, agents, max_moves,
                                rng=eval_rng(base_seed, g))
        tally_game(res, cand_seat, winner)
        if verbose and (g + 1) % max(1, num_games // 4) == 0:
            logger.info("  [%d/%d] %s", g + 1, num_games, res.summary())
        # Progress heartbeat every 5 games (and on the last one).
        if on_progress is not None and ((g + 1) % 5 == 0 or (g + 1) == num_games):
            on_progress(g + 1, num_games, res)
    return res


def evaluate_against_random_mp(env, candidate: AgentFn, num_games: int = 24,
                               max_moves: int = 400, on_progress=None,
                               base_seed: int = 0) -> EvalResultMP:
    return evaluate_mp(env, candidate, random_agent(),
                       num_games=num_games, max_moves=max_moves,
                       on_progress=on_progress, base_seed=base_seed)
