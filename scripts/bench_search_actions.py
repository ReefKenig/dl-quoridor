"""Reproduce the search-resolution and cost numbers quoted for wall_candidates.

Prints the table in PR #43: how many actions search expands with and without the
restriction, the visits/action that buys at a given simulation budget, and what
the filter costs in wall time.

    PYTHONPATH=. python scripts/bench_search_actions.py --board 9 --sims 600

Everything here is a property of the env, so it needs no checkpoint and runs on
CPU in seconds. Strength numbers are a different measurement - see
scripts/eval_all_checkpoints.py.
"""
import argparse
import time

import numpy as np

from src.env.quoridor_env_mp import NUM_MOVE_ACTIONS, QuoridorEnvMP


def _env(board, players):
    walls = 3 if board == 5 else (10 if players == 2 else 5)
    return QuoridorEnvMP(board_size=board, num_players=players,
                         max_walls_per_player=walls,
                         max_turns=160 if players == 2 else 320)


def _random_plies(env, count, seed=0):
    """`count` states reached by uniform random legal play, restarting on game end."""
    rng = np.random.default_rng(seed)
    out, state = [], env.reset()
    while len(out) < count:
        actions = env.get_valid_actions(state)
        if len(actions) == 0 or state.game_over:
            state = env.reset()
            continue
        state = env.step(state, int(rng.choice(actions)))[0]
        out.append(state)
    return out


def _timed(fn, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=int, default=9)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--sims", type=int, default=600)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--plies", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    env = _env(args.board, args.players)
    k, sims = args.k, args.sims
    print(f"board {args.board}x{args.board}  N={args.players}  "
          f"sims={sims}  K={k}\n")

    opening = env.reset()
    full = len(env.get_valid_actions(opening))
    narrow = len(env.get_search_actions(opening, k))
    print(f"{'':<28}{'actions':>9}{'visits/action':>15}")
    print(f"{'opening, unrestricted':<28}{full:>9}{sims / full:>15.1f}")
    print(f"{'opening, K=' + str(k):<28}{narrow:>9}{sims / narrow:>15.1f}")

    states = _random_plies(env, args.plies)
    fulls = [len(env.get_valid_actions(s)) for s in states]
    narrows = [len(env.get_search_actions(s, k)) for s in states]
    mf, mn = float(np.mean(fulls)), float(np.mean(narrows))
    print(f"{'mean over ' + str(args.plies) + ' plies':<28}"
          f"{mf:>9.1f}{sims / mf:>15.1f}")
    print(f"{'  same plies, K=' + str(k):<28}"
          f"{mn:>9.1f}{sims / mn:>15.1f}")

    # Cost: the filter reuses the blockers already computed for wall legality,
    # so the overhead is the ranking, not another BFS sweep.
    t_full = _timed(lambda: [env.get_valid_actions(s) for s in states],
                    args.repeats)
    t_narrow = _timed(lambda: [env.get_search_actions(s, k) for s in states],
                      args.repeats)
    print(f"\nlisting {len(states)} states: unrestricted {t_full * 1e3:.1f} ms, "
          f"K={k} {t_narrow * 1e3:.1f} ms  "
          f"(+{100 * (t_narrow / t_full - 1):.0f}%)")

    walls_kept = sum(1 for a in env.get_search_actions(opening, k)
                     if a >= NUM_MOVE_ACTIONS)
    print(f"opening walls: {full - (narrow - walls_kept)} legal -> "
          f"{walls_kept} searched")


if __name__ == "__main__":
    main()
