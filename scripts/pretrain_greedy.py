"""Supervised warm start: clone the greedy racer, then hand the net to RL.

Six 9x9 N=4 attempts (v5 through v8) failed the same way for the same
structural reason: 128 of 131 legal opening actions are walls, so an untrained
net starts as a wall-spammer, and the only seat a racer can win (seat 0)
contributes ~3.5% of the gradient. This script attacks the prior directly -
generate greedy-vs-greedy games (no network, no MCTS, cheap on CPU), train the
policy head to imitate greedy's moves and the value head on outcomes, and save
a checkpoint for `TrainingConfigMP.init_checkpoint`.

It is also the decisive capacity test: if the network cannot even IMITATE a
scripted racer (low held-out agreement), the network is the problem, not the
training signal - learned locally in minutes instead of after 6 pod-hours.

Usage (defaults match configs/config_9x9.json's network):
  PYTHONPATH=. python scripts/pretrain_greedy.py --players 4 --board 9 \
      --games 2000 --epochs 4 --out runs/pretrain_n4_9x9/pretrain.pt \
      --eval-games 40
"""
import argparse
import json
import os

import numpy as np
import torch

from src.env.quoridor_env_mp import (NUM_MOVE_ACTIONS, QuoridorEnvMP,
                                     compute_action_space_size)
from src.env.tensor_spec_mp import CURRENT_SPEC
from src.mcts.evaluator_mp import evaluate_mp, greedy_agent, raw_policy_agent
from src.mcts.pretrain_data import (generate_games, pretrain_report_path,
                                    to_arrays)
from src.model.network_mp import QuoridorModelMP


def agreement(model, S, target_hot, batch=512):
    """Held-out top-1 agreement with greedy's move choice."""
    hits = 0
    for i in range(0, len(S), batch):
        x = torch.from_numpy(S[i:i + batch]).float().to(
            model.device).permute(0, 3, 1, 2)
        pol, _ = model.predict_batch(x)
        hits += (pol.argmax(1).cpu().numpy() == target_hot[i:i + batch]).sum()
    return hits / len(S)


def opening_wall_mass(model, env):
    policy, _ = model.predict(env.state_to_tensor(env.reset()))
    return float(policy[NUM_MOVE_ACTIONS:].sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--board", type=int, default=9)
    ap.add_argument("--walls", type=int, default=None,
                    help="max walls/player; default: the env's official count")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--opening-max", type=int, default=6,
                    help="random plies before greedy takes over (sampled 0..max)")
    ap.add_argument("--max-moves", type=int, default=320)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--discount-unit", default="ply", choices=["ply", "round"])
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--holdout", type=float, default=0.05,
                    help="fraction of GAMES held out for the agreement metric")
    ap.add_argument("--eval-games", type=int, default=0,
                    help="raw-policy games per seat vs greedy after training")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = QuoridorEnvMP(board_size=args.board, num_players=args.players,
                        max_turns=args.max_moves,
                        max_walls_per_player=args.walls,
                        spec_version=CURRENT_SPEC)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Generating {args.games} greedy games (N={args.players}, "
          f"{args.board}x{args.board}, opening 0-{args.opening_max})...")
    games = generate_games(env, args.games, args.opening_max, args.max_moves,
                           args.discount, args.discount_unit, args.seed)
    n_hold = max(1, int(len(games) * args.holdout))
    S, P, V = to_arrays(games[n_hold:])
    Sh, Ph, _ = to_arrays(games[:n_hold])
    del games
    hot_h = Ph.argmax(1)
    print(f"{len(S)} train / {len(Sh)} held-out samples")

    model = QuoridorModelMP(
        board_size=args.board,
        action_space_size=compute_action_space_size(args.board),
        in_channels=3 * args.players + 3, num_channels=args.channels,
        num_res_blocks=args.blocks, num_players=args.players,
        lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    print(f"device={model.device} | opening wall mass before: "
          f"{opening_wall_mass(model, env):.4f}")

    idx = np.arange(len(S))
    agree = 0.0
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        lp = lv = steps = 0
        for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
            b = idx[i:i + args.batch_size]
            pl, vl = model.train_step(S[b], P[b], V[b])
            lp, lv, steps = lp + pl, lv + vl, steps + 1
        agree = agreement(model, Sh, hot_h)
        print(f"epoch {epoch + 1}/{args.epochs} | loss_p={lp / steps:.4f} "
              f"loss_v={lv / steps:.4f} | held-out agreement={agree:.1%}")

    wall_mass = opening_wall_mass(model, env)
    print(f"final: agreement={agree:.1%} | opening wall mass={wall_mass:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    report = {"agreement": round(float(agree), 4),
              "opening_wall_mass": round(wall_mass, 6),
              "games": args.games, "samples": int(len(S)),
              "epochs": args.epochs, "players": args.players,
              "board": args.board, "spec_version": CURRENT_SPEC,
              "channels": args.channels, "blocks": args.blocks,
              "discount": args.discount, "discount_unit": args.discount_unit,
              "seed": args.seed}
    if args.eval_games:
        print(f"Raw-policy eval vs greedy ({args.eval_games} games/seat)...")
        # The same harness and RNG scheme as every other eval in the repo, so
        # the report's numbers share the decided-games semantics of meta.json.
        ev = evaluate_mp(env, raw_policy_agent(model), greedy_agent(),
                         num_games=args.eval_games * env.num_players,
                         max_moves=args.max_moves, base_seed=args.seed + 1)
        print(f"  {ev.summary()}")
        report["raw_policy_vs_greedy"] = {
            str(s): [ev.seat_wins.get(s, 0), ev.games_per_seat.get(s, 0)]
            for s in range(env.num_players)}
        report["raw_policy_rate_decided"] = round(ev.candidate_win_rate, 4)
        report["raw_policy_decided"] = ev.decided_games
    with open(pretrain_report_path(args.out), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
