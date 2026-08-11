"""Supervised warm start: clone the greedy racer, then hand the net to RL.

Six 9x9 N=4 attempts (v5 through v8) failed the same way for the same
structural reason: 128 of 131 legal opening actions are walls, so an untrained
net starts as a wall-spammer, and the only seat a racer can win (seat 0)
contributes ~3.5% of the gradient. This script attacks the prior directly —
generate greedy-vs-greedy games (no network, no MCTS, cheap on CPU), train the
policy head to imitate greedy's moves and the value head on outcomes, and save
a checkpoint for `TrainingConfigMP.init_checkpoint`.

It is also the decisive capacity test: if the network cannot even IMITATE a
scripted racer (low held-out agreement), the network is the problem, not the
training signal — learned locally in minutes instead of after 6 pod-hours.

Usage (defaults match configs/config_9x9.json's network):
  PYTHONPATH=. python scripts/pretrain_greedy.py --players 4 --board 9 \
      --games 2000 --epochs 4 --out runs/pretrain_n4_9x9/pretrain.pt \
      --eval-games 40
"""
import argparse
import json
import os
import time

import numpy as np

from src.env.quoridor_env_mp import (NUM_MOVE_ACTIONS, QuoridorEnvMP,
                                     compute_action_space_size)
from src.env.tensor_spec_mp import CURRENT_SPEC
from src.mcts.evaluator_mp import greedy_agent
from src.mcts.self_play_mp import assign_vector_targets, augment_mp, game_seed
from src.model.network_mp import QuoridorModelMP


def generate_games(env, num_games, opening_max, max_moves, discount,
                   discount_unit, base_seed, log=print):
    """Per-game sample lists (kept separate so the holdout split is by game).

    A random opening (mostly walls — they are 98% of legal actions) diversifies
    the states greedy then races through; only greedy's own plies become policy
    targets. Timeouts are dropped by assign_vector_targets, same as self-play.
    """
    greedy = greedy_agent()
    games, t0 = [], time.time()
    for g in range(num_games):
        rng = np.random.RandomState(game_seed(base_seed, g))
        opening = rng.randint(0, opening_max + 1)
        state = env.reset()
        trajectory, plies = [], []
        move_count, winner = 0, None
        while move_count < max_moves:
            if move_count < opening:
                action = int(rng.choice(env.get_valid_actions(state)))
            else:
                action = int(greedy(env, state, move_count, rng))
                onehot = np.zeros(env.action_space_size, dtype=np.float32)
                onehot[action] = 1.0
                trajectory.append((env.state_to_tensor(state), onehot,
                                   env.get_current_player(state)))
                plies.append(move_count)
            state, _, done, info = env.step(state, action)
            move_count += 1
            if done:
                winner = info.get("winner")
                break
        samples = assign_vector_targets(
            trajectory, winner, env.num_players, discount,
            discount_unit=discount_unit, plies=plies, total_plies=move_count)
        aug = [augment_mp(t, p, v, env.num_players, env.board_size)
               for (t, p, v) in samples]
        if samples:
            games.append(samples + aug)
        if (g + 1) % 200 == 0:
            log(f"  {g + 1}/{num_games} games, "
                f"{sum(len(s) for s in games)} samples ({time.time() - t0:.0f}s)")
    return games


def to_arrays(games):
    flat = [s for game in games for s in game]
    S = np.stack([s[0] for s in flat]).astype(np.float32)
    P = np.stack([s[1] for s in flat]).astype(np.float32)
    V = np.stack([s[2] for s in flat]).astype(np.float32)
    return S, P, V


def agreement(model, S, P, batch=512):
    """Held-out top-1 agreement with greedy's move choice."""
    import torch
    hits = 0
    for i in range(0, len(S), batch):
        x = torch.from_numpy(S[i:i + batch]).float().permute(0, 3, 1, 2)
        pol, _ = model.predict_batch(x.to(model.device))
        hits += (pol.argmax(1).cpu().numpy()
                 == P[i:i + batch].argmax(1)).sum()
    return hits / len(S)


def opening_wall_mass(model, env):
    policy, _ = model.predict(env.state_to_tensor(env.reset()))
    return float(policy[NUM_MOVE_ACTIONS:].sum())


def raw_policy_agent(model):
    """Argmax of the raw policy over valid actions — no search. Strength floor:
    any MCTS on top only adds to it."""
    def agent(env, state, ply=0, rng=None):
        policy, _ = model.predict(env.state_to_tensor(state))
        valid = env.get_valid_actions(state)
        return int(max(valid, key=lambda a: policy[a]))
    return agent


def eval_vs_greedy(model, env, games_per_seat, max_moves, base_seed, log=print):
    """Model (raw policy) in one seat, greedy in the rest. Returns {seat: [w, n]}."""
    greedy = greedy_agent()
    me = raw_policy_agent(model)
    out = {}
    for seat in range(env.num_players):
        wins = 0
        for g in range(games_per_seat):
            rng = np.random.RandomState(game_seed(base_seed + 7919 * seat, g))
            state = env.reset()
            for ply in range(max_moves):
                mover = env.get_current_player(state)
                agent = me if mover == seat else greedy
                state, _, done, info = env.step(
                    state, int(agent(env, state, ply, rng)))
                if done:
                    wins += info.get("winner") == seat
                    break
        out[seat] = [wins, games_per_seat]
        log(f"  seat {seat}: {wins}/{games_per_seat}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--board", type=int, default=9)
    ap.add_argument("--walls", type=int, default=None,
                    help="max walls/player; default: official (10 at N=2, 5 at N=4)")
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

    walls = args.walls if args.walls is not None else (10 if args.players == 2 else 5)
    env = QuoridorEnvMP(board_size=args.board, num_players=args.players,
                        max_turns=args.max_moves, max_walls_per_player=walls,
                        spec_version=CURRENT_SPEC)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    print(f"Generating {args.games} greedy games (N={args.players}, "
          f"{args.board}x{args.board}, opening 0-{args.opening_max})...")
    games = generate_games(env, args.games, args.opening_max, args.max_moves,
                           args.discount, args.discount_unit, args.seed)
    n_hold = max(1, int(len(games) * args.holdout))
    S, P, V = to_arrays(games[n_hold:])
    Sh, Ph, _ = to_arrays(games[:n_hold])
    print(f"{len(S)} train / {len(Sh)} held-out samples from {len(games)} games")

    model = QuoridorModelMP(
        board_size=args.board,
        action_space_size=compute_action_space_size(args.board),
        in_channels=3 * args.players + 3, num_channels=args.channels,
        num_res_blocks=args.blocks, num_players=args.players,
        lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    print(f"device={model.device} | opening wall mass before: "
          f"{opening_wall_mass(model, env):.4f}")

    idx = np.arange(len(S))
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        lp = lv = steps = 0
        for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
            b = idx[i:i + args.batch_size]
            pl, vl = model.train_step(S[b], P[b], V[b])
            lp, lv, steps = lp + pl, lv + vl, steps + 1
        print(f"epoch {epoch + 1}/{args.epochs} | loss_p={lp / steps:.4f} "
              f"loss_v={lv / steps:.4f} | held-out agreement="
              f"{agreement(model, Sh, Ph):.1%}")

    wall_mass = opening_wall_mass(model, env)
    agree = agreement(model, Sh, Ph)
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
        report["raw_policy_vs_greedy"] = eval_vs_greedy(
            model, env, args.eval_games, args.max_moves, args.seed + 1)
    with open(os.path.splitext(args.out)[0] + "_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
