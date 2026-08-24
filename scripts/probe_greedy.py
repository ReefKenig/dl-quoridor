"""Why does the model never beat the greedy racer?

Separates two explanations for a 0% greedy score: a policy that cannot win the
race, or the sampled eval opening (4 plies at temperature 1.0) throwing away the
tempo that decides a race against a deterministic rusher.

--trace additionally plays one game move by move and reports how the candidate
spent its turns, which turns "it loses" into a mechanism.

    PYTHONPATH=. python3 scripts/probe_greedy.py runs/n2_9x9_v4 --games 20
    PYTHONPATH=. python3 scripts/probe_greedy.py runs/n2_9x9_v4 --trace
"""
import argparse
import json
import os

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.mcts.evaluator_mp import (NUM_MOVE_ACTIONS, eval_rng, greedy_agent,
                                   mcts_agent_mp, play_eval_game)
from src.mcts.training_mp import TrainingConfigMP, _mcts
from src.model.network_mp import QuoridorModelMP
from src.utils.config import read_frozen_config
from src.utils.checkpoint import resolve_ship_checkpoint


def probe(env, agent_for, games, seats, base_seed):
    """Play `games` per seat; returns (wins, decided, per-seat wins)."""
    wins = decided = 0
    by_seat = {s: 0 for s in seats}
    for seat in seats:
        for g in range(games):
            cand = agent_for()
            agents = {s: (cand if s == seat else greedy_agent())
                      for s in range(env.num_players)}
            winner, _ = play_eval_game(env, agents, env.max_turns,
                                       rng=eval_rng(base_seed + 1000 * seat, g))
            if winner is not None:
                decided += 1
                if winner == seat:
                    wins += 1
                    by_seat[seat] += 1
    return wins, decided, by_seat


def trace(env, cand, seat, base_seed=0):
    """Play one game and report how the candidate spent its turns.

    A racer wins on tempo, so the diagnosis is what the candidate does with its
    turns: walls it cannot afford, moves that do not shorten its path, or a clean
    race it simply loses.
    """
    rng = eval_rng(base_seed, 0)
    agents = {s: (cand if s == seat else greedy_agent())
              for s in range(env.num_players)}
    state = env.reset()
    walls = closer = other = 0
    start = env.distance_to_goal(state, seat)
    ply = 0

    while not state.game_over and ply < env.max_turns:
        mover = state.current_player
        action = int(agents[mover](env, state, ply=ply, rng=rng))
        before = env.distance_to_goal(state, seat)
        state, _r, _done, _info = env.step(state, action)
        if mover == seat:
            after = env.distance_to_goal(state, seat)
            if action >= NUM_MOVE_ACTIONS:
                walls += 1
            elif after is not None and before is not None and after < before:
                closer += 1
            else:
                other += 1
        ply += 1

    turns = walls + closer + other
    dist = {s: env.distance_to_goal(state, s) for s in range(env.num_players)}
    winner = "draw/timeout" if state.winner is None else f"P{state.winner}"
    print(f"\n  TRACE (candidate in seat {seat}, {ply} plies, winner {winner})")
    print(f"    candidate turns: {turns}   walls {walls}   "
          f"moved closer {closer}   neither {other}")
    print(f"    its distance to goal: {start} -> {dist[seat]}")
    print(f"    all distances at the end: {dist}")
    if not turns:
        return
    print(f"    {100 * walls / turns:.0f}% of its turns were walls, "
          f"{100 * closer / turns:.0f}% advanced it")
    if state.winner == seat:
        print("    -> it won this game")
    elif walls > closer:
        print("    -> walls more than it advances: losing the race on tempo")
    elif other > closer:
        print("    -> most pawn moves do not shorten its path: it is not racing")
    else:
        print("    -> it races cleanly but is still outpaced")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--games", type=int, default=20, help="games per seat, per setting")
    ap.add_argument("--sims", type=int, default=0, help="0 = eval_simulations from config")
    ap.add_argument("--channels", type=int, default=0, help="0 = from configs/config_<b>x<b>.json")
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--wall-candidates", type=int, default=None,
                    dest="wall_candidates",
                    help="expanded walls per node; default = the run's own setting")
    ap.add_argument("--trace", action="store_true",
                    help="also play one game move by move and diagnose the loss")
    args = ap.parse_args()

    # Fills in spec_version for dirs frozen before it existed (all v1).
    rc = read_frozen_config(args.run_dir)
    if rc is None:
        ap.error(f"no readable config.json in {args.run_dir}")
    N, board = rc["num_players"], rc["board_size"]
    sims = args.sims or rc.get("eval_simulations") or rc["mcts_simulations"]

    # Run configs never recorded the network shape; it lives in the repo config.
    net = json.load(open(f"configs/config_{board}x{board}.json")).get("network", {})
    channels = args.channels or net.get("num_channels", 64)
    blocks = args.blocks or net.get("num_res_blocks", 4)

    path, label = resolve_ship_checkpoint(args.run_dir)
    print(f"{args.run_dir}: {label}\n  N={N} board={board} sims={sims} "
          f"net={channels}x{blocks} ({args.games} games per seat)\n")

    env = QuoridorEnvMP(board_size=board, num_players=N,
                        max_turns=rc["max_game_moves"],
                        max_walls_per_player=rc["max_walls_per_player"],
                        spec_version=rc["spec_version"])
    model = QuoridorModelMP(
        board_size=board, action_space_size=compute_action_space_size(board),
        in_channels=3 * N + 3, num_channels=channels,
        num_res_blocks=blocks, num_players=N, device="auto")
    model.load(path)

    # Without this the probe searches all 131 opening actions while the model
    # was trained under a restricted search - a different agent than the run's.
    wall_candidates = (args.wall_candidates if args.wall_candidates is not None
                       else rc.get("mcts_wall_candidates", 0) or 0)
    print(f"  mcts_wall_candidates={wall_candidates}"
          f"{' (from the run config)' if args.wall_candidates is None else ''}")
    mcfg = TrainingConfigMP(num_players=N, mcts_simulations=sims,
                            max_game_moves=rc["max_game_moves"],
                            mcts_wall_candidates=wall_candidates)
    for plies in (0, rc.get("eval_opening_plies", 4)):
        def agent_for(p=plies):
            return mcts_agent_mp(_mcts(model, env, mcfg, sims=sims,
                                       dirichlet_epsilon=0.0),
                                 temperature=0.1, opening_plies=p)
        # greedy is deterministic, so with no sampled opening every game in a
        # seat is the same replay: score one and say so, rather than reporting
        # `args.games` independent games that were all the same game.
        games = 1 if plies == 0 else args.games
        wins, decided, by_seat = probe(env, agent_for, games,
                                       list(range(N)), base_seed=7 + plies)
        rate = wins / decided if decided else 0.0
        seats = " ".join(f"seat{s}:{w}" for s, w in by_seat.items())
        note = " (1 game/seat - a 0-ply opening replays)" if plies == 0 else ""
        print(f"  opening_plies={plies}: {wins}/{decided} decided = {rate:.1%}"
              f"  [{seats}]{note}")

    if args.trace:
        cand = mcts_agent_mp(_mcts(model, env, mcfg, sims=sims, dirichlet_epsilon=0.0),
                             temperature=0.1, opening_plies=0)
        trace(env, cand, seat=0)

    print("\n  Both near 0% -> the policy cannot win the race.")
    print("  0 plies much higher -> the sampled opening is costing the tempo,")
    print("  and the training-time greedy number understates real strength.")


if __name__ == "__main__":
    main()
