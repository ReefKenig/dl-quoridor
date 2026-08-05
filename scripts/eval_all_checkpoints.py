"""Re-score every checkpoint under ONE protocol: greedy, held-out minimax, K=0 vs K=16.

Answers the question Branch 1 left open — is the wall-candidate restriction a
TRAINING result or an inference-time one? Scoring each checkpoint at both K under
an identical protocol is the only way to tell, so K is a column here, not a
setting.

    PYTHONPATH=. .venv/bin/python scripts/eval_all_checkpoints.py

Three things this protocol gets right, each of which has produced a wrong number
in this project before:

- Games go through `evaluate_mp`, the harness training itself uses. A second game
  loop here is how these numbers drifted from the ones training reports.
- `opening_plies` comes from the run's frozen config and may not be 0. Against
  deterministic opponents an unsampled opening replays one identical game per
  seat, so "10 games/seat" was ~1 game reported as 0/10 or 10/10.
- Each checkpoint plays on the tensor spec it TRAINED under. v1 planes normalise
  distance by board_size**2 and v2 by 2*board_size, so the wrong spec silently
  rescales two channels and changes wall behaviour.

Writes outputs/held_out_eval.json (every row) and outputs/v7_vs_v4.json (the K
ablation slice). Both files previously held broken-protocol numbers; regenerate
them here rather than citing what is on disk.
"""
import json
import os
import sys
import time

import torch

from src.env.quoridor_env_mp import (NUM_MOVE_ACTIONS, QuoridorEnvMP,
                                     compute_action_space_size)
from src.mcts.evaluator_mp import (DEFAULT_EVAL_OPENING_PLIES, evaluate_mp,
                                   greedy_agent, mcts_agent_mp, minimax_agent)
from src.mcts.mcts_maxn import MCTSConfig, MCTSMaxN
from src.model.network_mp import QuoridorModelMP
from src.utils.config import read_frozen_config

GAMES_PER_SEAT = int(os.environ.get("GAMES_PER_SEAT", 20))
SIMS = int(os.environ.get("SIMS", 200))
DEPTH = int(os.environ.get("MINIMAX_DEPTH", 2))
WALL_CANDIDATES = [int(k) for k in
                   os.environ.get("WALL_CANDIDATES", "0,16").split(",")]
# Restrict to matching checkpoints, e.g. ONLY=n2_9x9_v7,probe_n2_ramp
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]

# Newest first. spec and opening plies come from each run's frozen config.json
# where it has one; a run with no spec_version recorded predates v2, so it is v1.
CHECKPOINTS = [
    ("runs/n2_9x9_v7/latest.pt", 2, 9),
    ("runs/n4_9x9_v7/ship.pt", 4, 9),
    ("runs/probe_n2_ramp/best.pt", 2, 9),
    ("runs/probe_n4_ramp/best.pt", 4, 9),
    ("runs/n2_9x9_v6/ship.pt", 2, 9),
    ("runs/n4_9x9_v6/ship.pt", 4, 9),
    ("runs/n2_9x9_v4/ship.pt", 2, 9),
    ("runs/n4_9x9_v5/ship.pt", 4, 9),
    ("runs/n2_5x5_v1/ship.pt", 2, 5),
    ("runs/n4_5x5_v3/ship.pt", 4, 5),
]


def geometry(num_players, board):
    """Walls and ply cap from configs/config_9x9.json, so eval matches training."""
    walls = 3 if board == 5 else (10 if num_players == 2 else 5)
    max_moves = 60 if board == 5 else (160 if num_players == 2 else 320)
    return walls, max_moves


def run_settings(path):
    """(spec_version, opening_plies, trained_K) from the run's frozen config."""
    frozen = read_frozen_config(os.path.dirname(path)) or {}
    return (int(frozen.get("spec_version") or 1),
            frozen.get("eval_opening_plies"),
            int(frozen.get("mcts_wall_candidates") or 0))


def resolve_opening_plies(path, frozen_plies, games_per_seat):
    """Refuse a protocol that scores many games but plays one.

    The Issue 9 bug: greedy and minimax are deterministic, so with no sampled
    opening every game in a seat is the same replay and N games is really 1.
    """
    plies = DEFAULT_EVAL_OPENING_PLIES if frozen_plies is None else int(frozen_plies)
    if plies == 0 and games_per_seat > 1:
        raise SystemExit(
            f"{path} records eval_opening_plies=0. Against greedy and minimax, "
            f"which are deterministic, that replays ONE game per seat — "
            f"{games_per_seat} games/seat would be reported from 1 distinct "
            f"game. Set GAMES_PER_SEAT=1 to score a single replay deliberately, "
            f"or fix the run's config. See tests/test_eval_opening_diversity.py.")
    return plies


def infer_shape(state):
    """(channels, blocks) from the checkpoint itself, so no config is needed."""
    channels = state["conv_input.weight"].shape[0]
    blocks = len({k.split(".")[1] for k in state if k.startswith("res_blocks.")})
    return channels, blocks


def load(path, num_players, board):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["network_state"]
    channels, blocks = infer_shape(state)
    model = QuoridorModelMP(
        board_size=board, action_space_size=compute_action_space_size(board),
        in_channels=3 * num_players + 3, num_channels=channels,
        num_res_blocks=blocks, num_players=num_players, device="cpu")
    model.network.load_state_dict(state)
    return model, channels, blocks


def make_env(num_players, board, spec):
    walls, max_moves = geometry(num_players, board)
    return QuoridorEnvMP(board_size=board, num_players=num_players,
                         max_walls_per_player=walls, max_turns=max_moves,
                         spec_version=spec)


def candidate_agent(model, env, num_players, wall_candidates, opening_plies,
                    max_moves):
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=SIMS, dirichlet_epsilon=0.0,
                          max_rollout_depth=max_moves,
                          wall_candidates=wall_candidates),
        evaluate_fn=lambda s: model.predict(env.state_to_tensor(s)),
        num_players=num_players)
    return mcts_agent_mp(mcts, temperature=0.1, opening_plies=opening_plies)


def opening_wall_mass(model, env):
    """Policy mass on walls at the opening, before any search.

    The cheapest read on "does this checkpoint race or spam walls": an untrained
    9x9 net sits near 0.98, because 128 of 131 legal opening actions are walls.
    """
    state = env.reset()
    probs, _ = model.predict(env.state_to_tensor(state))
    valid = env.get_valid_actions(state)
    total = sum(float(probs[a]) for a in valid)
    walls = sum(float(probs[a]) for a in valid if a >= NUM_MOVE_ACTIONS)
    return round(walls / total, 4) if total > 0 else None


def as_row(result, num_players):
    return {
        "seats": {str(s): [result.seat_wins.get(s, 0), n]
                  for s, n in sorted(result.games_per_seat.items())},
        "wins": result.candidate_wins,
        "decided": result.decided_games,
        "games": result.num_games,
        "rate": round(result.candidate_win_rate, 4),
        "racer_ceiling": round(1.0 / num_players, 4),
    }


def score_greedy_vs_minimax(num_players, board):
    """Control: is the minimax bar reachable at all on this board?

    Nothing in the project beats minimax at 9x9, including 5x5 models that play
    their board optimally, so a flat zero may be a statement about the opponent
    rather than about the model. Greedy carries no network, so this is the same
    protocol with the model removed.
    """
    _walls, max_moves = geometry(num_players, board)
    env = make_env(num_players, board, spec=1)
    res = evaluate_mp(env, greedy_agent(), minimax_agent(depth=DEPTH),
                      num_games=GAMES_PER_SEAT * num_players,
                      max_moves=max_moves, base_seed=7)
    return as_row(res, num_players)


def selected():
    return [(p, n, b) for p, n, b in CHECKPOINTS
            if not ONLY or any(tag in p for tag in ONLY)]


def write_outputs(rows, controls):
    """Persist after every checkpoint. Minimax costs ~20 s/game at 9x9, so a
    full sweep runs for hours and must not lose everything to one interrupt."""
    os.makedirs("outputs", exist_ok=True)
    meta = {"games_per_seat": GAMES_PER_SEAT, "sims": SIMS,
            "minimax_depth": DEPTH, "wall_candidates": WALL_CANDIDATES,
            "complete": len(rows) == len(selected()) * len(WALL_CANDIDATES) * 2,
            "protocol": "evaluate_mp; opening_plies from each run's frozen "
                        "config; each checkpoint on its own tensor spec"}
    with open("outputs/held_out_eval.json", "w") as f:
        json.dump({**meta, "results": rows, "controls": controls}, f, indent=2)
    # The K-ablation slice: the same rows, restricted to the comparison that
    # decides whether Branch 1 is a training or an inference result.
    ablation = [r for r in rows if r["board"] == 9]
    with open("outputs/v7_vs_v4.json", "w") as f:
        json.dump({**meta, "results": ablation}, f, indent=2)
    return ablation


def main():
    rows = []
    for path, n, board in selected():
        if not os.path.exists(path):
            print(f"SKIP {path} (missing)")
            continue

        spec, frozen_plies, trained_k = run_settings(path)
        opening_plies = resolve_opening_plies(path, frozen_plies, GAMES_PER_SEAT)
        model, ch, bl = load(path, n, board)
        env = make_env(n, board, spec)
        _walls, max_moves = geometry(n, board)
        wall_mass = opening_wall_mass(model, env)

        for k in WALL_CANDIDATES:
            cand = candidate_agent(model, env, n, k, opening_plies, max_moves)
            for opp_name, opp in (("greedy", greedy_agent()),
                                  ("minimax", minimax_agent(depth=DEPTH))):
                t0 = time.time()
                res = evaluate_mp(env, cand, opp,
                                  num_games=GAMES_PER_SEAT * n,
                                  max_moves=max_moves, base_seed=13)
                row = {"ckpt": path, "num_players": n, "board": board,
                       "spec": spec, "trained_wall_candidates": trained_k,
                       "wall_candidates": k, "opponent": opp_name,
                       "opening_plies": opening_plies, "sims": SIMS,
                       "channels": ch, "blocks": bl,
                       "opening_wall_mass": wall_mass,
                       "secs": round(time.time() - t0, 1),
                       **as_row(res, n)}
                rows.append(row)
                seats = " ".join(f"s{s}:{w}/{g}"
                                 for s, (w, g) in row["seats"].items())
                print(f"{path:32s} K={k:<3d} vs {opp_name:8s} "
                      f"{100*row['rate']:5.1f}%  [{seats}]  "
                      f"({row['decided']}/{row['games']} decided, {row['secs']}s)")
        write_outputs(rows, [])

    print("\n--- control: greedy vs minimax, no network ---")
    controls = []
    for n, board in sorted({(n, b) for _p, n, b in selected()}):
        row = score_greedy_vs_minimax(n, board)
        row.update({"matchup": "greedy_vs_minimax", "num_players": n,
                    "board": board, "minimax_depth": DEPTH})
        controls.append(row)
        print(f"  {board}x{board} N={n}: greedy scores {100*row['rate']:5.1f}% "
              f"of {row['decided']} decided vs minimax depth {DEPTH}")

    ablation = write_outputs(rows, controls)
    print(f"\nwrote outputs/held_out_eval.json ({len(rows)} rows, "
          f"{len(controls)} controls) and outputs/v7_vs_v4.json "
          f"({len(ablation)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
