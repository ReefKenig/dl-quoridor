"""Why does the model never beat the greedy racer?

Separates two explanations for a 0% greedy score: a policy that cannot win the
race, or the sampled eval opening (4 plies at temperature 1.0) throwing away the
tempo that decides a race against a deterministic rusher.

    PYTHONPATH=. python3 scripts/probe_greedy.py runs/n2_9x9_v4 --games 20
"""
import argparse
import json
import os

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.mcts.evaluator_mp import (eval_rng, greedy_agent, mcts_agent_mp,
                                   play_eval_game)
from src.mcts.training_mp import TrainingConfigMP, _mcts
from src.model.network_mp import QuoridorModelMP
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
            winner = play_eval_game(env, agents, env.max_turns,
                                    rng=eval_rng(base_seed + 1000 * seat, g))
            if winner is not None:
                decided += 1
                if winner == seat:
                    wins += 1
                    by_seat[seat] += 1
    return wins, decided, by_seat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--games", type=int, default=20, help="games per seat, per setting")
    ap.add_argument("--sims", type=int, default=0, help="0 = eval_simulations from config")
    ap.add_argument("--channels", type=int, default=0, help="0 = from configs/config_<b>x<b>.json")
    ap.add_argument("--blocks", type=int, default=0)
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.json")
    rc = json.load(open(cfg_path))
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
                        max_walls_per_player=rc["max_walls_per_player"])
    model = QuoridorModelMP(
        board_size=board, action_space_size=compute_action_space_size(board),
        in_channels=3 * N + 3, num_channels=channels,
        num_res_blocks=blocks, num_players=N, device="auto")
    model.load(path)

    mcfg = TrainingConfigMP(num_players=N, mcts_simulations=sims,
                            max_game_moves=rc["max_game_moves"])
    for plies in (0, rc.get("eval_opening_plies", 4)):
        def agent_for(p=plies):
            return mcts_agent_mp(_mcts(model, env, mcfg, sims=sims,
                                       dirichlet_epsilon=0.0),
                                 temperature=0.1, opening_plies=p)
        wins, decided, by_seat = probe(env, agent_for, args.games,
                                       list(range(N)), base_seed=7 + plies)
        rate = wins / decided if decided else 0.0
        seats = " ".join(f"seat{s}:{w}" for s, w in by_seat.items())
        print(f"  opening_plies={plies}: {wins}/{decided} decided = {rate:.1%}  [{seats}]")

    print("\n  Both near 0% -> the policy cannot win the race.")
    print("  0 plies much higher -> the sampled opening is costing the tempo,")
    print("  and the training-time greedy number understates real strength.")


if __name__ == "__main__":
    main()
