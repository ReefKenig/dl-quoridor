"""
Large-sample re-evaluation of ship checkpoints.

Runs 240 games (candidate rotates through all seats) against random.
Produces a JSON report with win rate ± 95% CI for each model.

Usage:
    PYTHONPATH=. python scripts/reeval_ship.py
"""
from src.mcts.evaluator_mp import evaluate_against_random_mp, mcts_agent_mp
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.model.network_mp import QuoridorModelMP
from src.env.quoridor_env_mp import QuoridorEnvMP
import json
import sys
import time
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


NUM_GAMES = 240
MCTS_SIMS = 100

# tensor_spec mirrors runs/MODELS.json — both ship.pt files predate the v2
# distance-plane divisor, so they must be fed v1 planes to score meaningfully.
MODELS = {
    "2p_vector": {
        "path": "runs/n2_5x5_v1/ship.pt",
        "board_size": 5, "num_players": 2, "in_channels": 9,
        "action_space_size": 44, "num_channels": 64, "num_res_blocks": 4,
        "tensor_spec": 1,
    },
    "4p_ship": {
        "path": "runs/n4_5x5_v3/ship.pt",
        "board_size": 5, "num_players": 4, "in_channels": 15,
        "action_space_size": 44, "num_channels": 64, "num_res_blocks": 4,
        "tensor_spec": 1,
    },
}


def wilson_ci(wins, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, center - margin, center + margin


def eval_model(name, spec):
    print(f"\n{'='*60}")
    print(f"  {name}: {spec['path']} ({NUM_GAMES} games)")
    print(f"{'='*60}")

    model = QuoridorModelMP(
        board_size=spec["board_size"],
        action_space_size=spec["action_space_size"],
        in_channels=spec["in_channels"],
        num_channels=spec["num_channels"],
        num_res_blocks=spec["num_res_blocks"],
        num_players=spec["num_players"],
        device="cpu",
    )
    model.load(spec["path"])
    env = QuoridorEnvMP(board_size=spec["board_size"],
                        num_players=spec["num_players"], max_turns=300,
                        spec_version=spec["tensor_spec"])
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=MCTS_SIMS, c_puct=1.41,
                          dirichlet_epsilon=0.0),
        evaluate_fn=lambda state, _env=env, _m=model: (
            _m.predict(_env.state_to_tensor(state))
        ),
        num_players=spec["num_players"],
    )
    agent = mcts_agent_mp(mcts, temperature=0.1)

    t0 = time.time()
    res = evaluate_against_random_mp(env, agent, num_games=NUM_GAMES,
                                     max_moves=300)
    elapsed = time.time() - t0

    wr = res.candidate_win_rate
    p_hat, ci_lo, ci_hi = wilson_ci(res.candidate_wins, NUM_GAMES)
    margin = (ci_hi - ci_lo) / 2

    print(f"  Result: {res.candidate_wins}/{NUM_GAMES} = {wr:.1%} "
          f"± {margin:.1%} (95% Wilson CI)")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  {res.summary()}")

    return {
        "model": name,
        "path": spec["path"],
        "num_games": NUM_GAMES,
        "wins": res.candidate_wins,
        "draws": res.draws,
        "win_rate": round(wr, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "margin_95": round(margin, 4),
        "elapsed_s": round(elapsed, 1),
        "by_seat": {str(s): res.seat_wins.get(s, 0)
                    for s in range(spec["num_players"])},
    }


if __name__ == "__main__":
    results = []
    # Only re-eval the model(s) specified on CLI, or all if none given
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    for name in targets:
        if name not in MODELS:
            print(f"Unknown model: {name}")
            continue
        results.append(eval_model(name, MODELS[name]))

    out_path = Path("outputs/reeval_ship_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "mcts_sims": MCTS_SIMS,
                   "results": results}, f, indent=2)
    print(f"\nResults saved: {out_path}")
