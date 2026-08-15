"""
N-player evaluation harness.

A candidate agent is rotated through all N seats; the remaining seats are filled
by an `opponent` AgentFn (reused — agents are stateless per search). Fair share
for the candidate is 1/N, so `should_accept` compares the candidate win-rate to a
threshold that should sit above 1/N (the training loop sets it per N).

At N=2 this reduces to the usual candidate-vs-opponent duel (fair share 0.5).
"""
import logging
import math

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.env.quoridor_env_mp import NUM_MOVE_ACTIONS, decode_wall_action

logger = logging.getLogger(__name__)
AgentFn = Callable[[Any, Any], int]   # (env, state) -> action


# Sampling the opening from the search's own visit distribution, rather than
# playing argmax from move one. See mcts_agent_mp for why.
DEFAULT_EVAL_OPENING_PLIES = 4
OPENING_TEMPERATURE = 1.0


def mcts_agent_mp(mcts, temperature: float = 0.1,
                  opening_plies: int = 0,
                  opening_temperature: float = OPENING_TEMPERATURE) -> AgentFn:
    """Argmax over visit counts, except the first `opening_plies` moves.

    Without a sampled opening, eps=0 + argmax makes every game with the same
    seat assignment identical. Sampling needs its own temperature: the eval
    default of 0.1 raises counts to the 10th power and is effectively one-hot.
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


def raw_policy_agent(model) -> AgentFn:
    """Argmax of the raw policy over valid actions — no search. The strength
    floor: any MCTS on top only adds to it."""
    def agent(env, state, ply: int = 0, rng=None) -> int:
        policy, _ = model.predict(env.state_to_tensor(state))
        valid = env.get_valid_actions(state)
        return int(max(valid, key=lambda a: policy[a]))
    return agent


# Actions 0..11 are pawn moves; everything above is a wall placement. Re-exported
# from the env, which owns the action layout.


def greedy_agent() -> AgentFn:
    """Pawn-rush: the legal move that most shortens your own path, never a wall.
    A fixed yardstick, unlike vs-random (which saturates) or the moving gate."""
    def agent(env, state, ply: int = 0, rng=None) -> int:
        r = np.random if rng is None else rng
        valid = env.get_valid_actions(state)
        moves = [int(a) for a in valid if a < NUM_MOVE_ACTIONS]
        if not moves:
            return int(r.choice(valid))

        cp = state.current_player
        best, ties = None, []
        for action in moves:
            nxt, *_ = env.step(state, action)
            dist = env.distance_to_goal(nxt, cp)
            if dist is None:            # walled in by this move — never choose it
                continue
            if best is None or dist < best:
                best, ties = dist, [action]
            elif dist == best:
                ties.append(action)
        if not ties:
            return int(r.choice(moves))
        # Break ties randomly: a deterministic argmin would make every game with
        # the same seat assignment replay identically.
        return int(r.choice(ties))
    return agent


WIN_SCORE = 1e6


def _path_difference(env, state):
    """Per-player score: how far ahead of the nearest rival each player is.

    Zero-sum at N=2, so the maxn backup below reduces to minimax there — the
    same reduction `run_reduction.py` proves for the search itself.
    """
    n = state.num_players
    if state.game_over and state.winner is not None:
        return [WIN_SCORE if i == state.winner else -WIN_SCORE for i in range(n)]
    # Walled in reads as maximally behind rather than crashing the backup.
    dist = [env.distance_to_goal(state, i) for i in range(n)]
    dist = [WIN_SCORE if d is None else d for d in dist]
    return [min(dist[:i] + dist[i + 1:]) - dist[i] for i in range(n)]


def _wall_candidates(env, state, valid, max_candidates):
    """Wall actions touching some player's shortest path, nearest-first.

    128 wall slots at 9x9 make full-width search hopeless, and almost all of
    them are nowhere near anybody's route. Restricting to the ones that can
    actually change a path length is what makes depth 2 affordable.
    """
    walls = [a for a in valid if a >= NUM_MOVE_ACTIONS]
    if not walls or len(walls) <= max_candidates:
        return walls
    on_path = set()
    for i in range(state.num_players):
        path = env._shortest_path(state.positions[i], state.goals[i],
                                  state.h_walls, state.v_walls)
        for cell in (path or []):
            on_path.add(cell)
    def rank(action):
        _is_h, r, c = decode_wall_action(action, env.board_size)
        # A wall at (r, c) sits between the four cells touching that corner.
        touching = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
        return -len(touching & on_path)
    walls.sort(key=rank)
    return walls[:max_candidates]


MAX_MINIMAX_DEPTH = 3


def minimax_agent(depth: int = 2, max_wall_candidates: int = 16) -> AgentFn:
    """Depth-limited maxn on the path-difference heuristic.

    The conventional Quoridor baseline, and unlike `greedy_agent` it PLACES
    WALLS — so it probes the skill the 9x9 models lack and does not saturate
    the way vs-random does. Held out from training on purpose: see
    docs/opponent_pool_and_evaluation.md.

    Cost: the branching factor is <=12 pawn moves + max_wall_candidates walls,
    and every leaf runs a BFS per player. At the defaults (28 wide) depth 2 is
    ~800 leaves per move; depth 4 would be ~600k and stall an eval, so depth is
    capped at MAX_MINIMAX_DEPTH rather than left to fail as a timeout.
    """
    if depth < 1:
        raise ValueError(f"minimax depth must be at least 1, got {depth}")
    if depth > MAX_MINIMAX_DEPTH:
        raise ValueError(
            f"minimax depth {depth} exceeds MAX_MINIMAX_DEPTH="
            f"{MAX_MINIMAX_DEPTH}: at a branching factor of "
            f"{NUM_MOVE_ACTIONS + max_wall_candidates} this is "
            f"~{(NUM_MOVE_ACTIONS + max_wall_candidates) ** depth:,} leaf "
            f"evaluations per move, each running a BFS per player.")
    def search(env, state, d):
        if state.game_over or d == 0:
            return _path_difference(env, state)
        valid = env.get_valid_actions(state)
        if len(valid) == 0:
            return _path_difference(env, state)
        cp = state.current_player
        actions = [int(a) for a in valid if a < NUM_MOVE_ACTIONS]
        actions += _wall_candidates(env, state, [int(a) for a in valid],
                                    max_wall_candidates)
        best = None
        for action in actions:
            nxt, *_ = env.step(state, action)
            score = search(env, nxt, d - 1)
            if best is None or score[cp] > best[0][cp]:
                best = (score, [action])
            elif score[cp] == best[0][cp]:
                best[1].append(action)
        return best[0] if best else _path_difference(env, state)

    def agent(env, state, ply: int = 0, rng=None) -> int:
        r = np.random if rng is None else rng
        valid = [int(a) for a in env.get_valid_actions(state)]
        if not valid:
            raise ValueError("minimax_agent called on a state with no legal action")
        cp = state.current_player
        actions = [a for a in valid if a < NUM_MOVE_ACTIONS]
        actions += _wall_candidates(env, state, valid, max_wall_candidates)
        best, ties = None, []
        for action in actions:
            nxt, *_ = env.step(state, action)
            score = search(env, nxt, depth - 1)[cp]
            if best is None or score > best:
                best, ties = score, [action]
            elif score == best:
                ties.append(action)
        # Random tie-break, as in greedy_agent: a deterministic argmax would
        # make every game with the same seat assignment replay identically.
        return int(r.choice(ties))
    return agent


def eval_rng(base_seed: int, game_index: int):
    """Per-game RNG, keyed on the game index so the outcome never depends on
    which worker ran it. Explicit RandomState, not global np.random."""
    return np.random.RandomState(base_seed + game_index)


def adjudicate_timeout(env, state) -> Optional[int]:
    """Winner of a stalled game by shortest path to goal; a tie stays a draw.

    The standard Quoridor adjudication, used by the GATE only: two trained
    wall-capable models stall each other past any move cap (v9: 3-6 of 40
    gate games decided at 320 plies while the candidate won 80-100% of the
    ones that finished), so 'decided' starves and the gate can never fire.
    Racing opponents finish their games on their own and stay unadjudicated.
    """
    dists = [env.distance_to_goal(state, p) for p in range(env.num_players)]
    dists = [math.inf if d is None else d for d in dists]
    best = min(dists)
    leaders = [p for p, d in enumerate(dists) if d == best]
    return leaders[0] if len(leaders) == 1 and best < math.inf else None


def play_eval_game(env, agents, max_moves: int, rng=None,
                   adjudicate: bool = False):
    """Play one evaluation game.

    Returns (winner, adjudicated): winner None is a timeout draw; adjudicated
    is True when the winner came from distance adjudication, never from play.
    """
    state = env.reset()
    for ply in range(max_moves):
        cp = env.get_current_player(state)
        action = agents[cp](env, state, ply, rng)
        state, _, done, info = env.step(state, action)
        if done:
            return info.get("winner"), False
    if adjudicate:
        winner = adjudicate_timeout(env, state)
        if winner is not None:
            return winner, True
    return None, False


def tally_game(res: "EvalResultMP", cand_seat: int, winner: Optional[int],
               adjudicated: bool = False):
    """Fold one game's outcome into a result. Shared by all three eval paths."""
    res.num_games += 1
    res.games_per_seat[cand_seat] = res.games_per_seat.get(cand_seat, 0) + 1
    if winner is None:
        res.draws += 1
    elif winner == cand_seat:
        res.candidate_wins += 1
        res.seat_wins[cand_seat] = res.seat_wins.get(cand_seat, 0) + 1
    else:
        res.opponent_wins += 1
    res.adjudicated += adjudicated
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
    # Wins awarded by distance adjudication rather than reaching a goal.
    adjudicated: int = 0

    # Below this fraction of decided games the eval is too thin to gate on.
    MIN_DECIDED_FRACTION = 0.25

    @property
    def decided_games(self) -> int:
        """Games with a winner. A draw here is a timeout, not a real draw."""
        return self.num_games - self.draws

    @property
    def draw_rate(self) -> float:
        return self.draws / self.num_games if self.num_games else 0.0

    @property
    def candidate_win_rate(self) -> float:
        """Win rate over *decided* games; timeouts are excluded, not counted
        as losses (which capped the achievable rate at 1 - draw_rate)."""
        return (self.candidate_wins / self.decided_games
                if self.decided_games else 0.0)

    @property
    def fair_share(self) -> float:
        return 1.0 / self.num_players

    @property
    def adj_note(self) -> str:
        """', N adjudicated' when any win came from adjudication, else ''."""
        return f", {self.adjudicated} adjudicated" if self.adjudicated else ""

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
        return (f"games={self.num_games} (decided {self.decided_games}"
                f"{self.adj_note}) | "
                f"candidate {self.candidate_wins} "
                f"({self.candidate_win_rate:.1%} of decided, "
                f"fair={self.fair_share:.1%}) | "
                f"opp {self.opponent_wins} | draws {self.draws} "
                f"({self.draw_rate:.0%}) | by-seat[{per_seat}]")


def evaluate_mp(env, candidate: AgentFn, opponent: AgentFn,
                num_games: int = 24, max_moves: int = 400,
                verbose: bool = False, on_progress=None,
                base_seed: int = 0, adjudicate: bool = False) -> EvalResultMP:
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
        winner, adjudicated = play_eval_game(env, agents, max_moves,
                                             rng=eval_rng(base_seed, g),
                                             adjudicate=adjudicate)
        tally_game(res, cand_seat, winner, adjudicated)
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
