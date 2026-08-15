"""
Leaf-parallel fidelity test: does leaf_batch>1 (virtual loss) cost strength?

leaf_batch=8 + virtual loss is the ONLY knob in the self-play stack that makes the
search an *approximation* of exact sequential MCTS (concurrent tree walks are
steered apart, so the final visit distribution differs slightly from leaf_batch=1).
Every other speedup (work-stealing, Option B) produces bit-identical samples.

This script measures that approximation two ways on a FIXED checkpoint, so the
result is a clean comparison of the search only (same net, no Dirichlet noise):

  1. Search fidelity — on K sampled positions, run deterministic search with
     leaf_batch=1 and leaf_batch=B at the SAME sim budget, then compare the visit
     distributions: top-1 action agreement, mean total-variation distance, mean KL.
  2. Strength — win rate vs random for leaf_batch=1 and leaf_batch=B (Wilson 95% CI).

Verdict:
  - fidelity high (top-1 agree high, TV small) AND strength CIs overlap
        => virtual loss is harmless here; the work-stealing tail-fix is enough,
           Option B is optional (pure throughput, no strength gain).
  - fidelity low OR leaf_batch=B strength clearly below leaf_batch=1
        => the approximation is costing strength; Option B (which batches ACROSS
           games and lets each game run exact leaf_batch=1) pays off on STRENGTH.

Usage (local 5x5 default):
    PYTHONPATH=. python scripts/run_fidelity_test.py

Usage (9x9 run under test):
    PYTHONPATH=. python scripts/run_fidelity_test.py \
        --checkpoint runs/n2_9x9_v2/best.pt --board-size 9 --walls 10 \
        --num-channels 128 --num-res-blocks 8 --sims 800 --leaf-batch 8 \
        --positions 60 --games 120
"""
from src.mcts.evaluator_mp import evaluate_against_random_mp, mcts_agent_mp
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.model.network_mp import QuoridorModelMP
from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def wilson_ci(wins, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, center - margin, center + margin


def make_eval_single(model, env):
    """evaluate_fn for the sequential path: state -> (policy(A,), value(N,))."""
    return lambda s: model.predict(env.state_to_tensor(s))


def make_eval_many(model, env):
    """evaluate_fn for the leaf-parallel path: [state,...] -> [(policy, value), ...]."""
    def ev(states, model_id=0):
        stacked = torch.from_numpy(
            np.ascontiguousarray(
                np.stack([env.state_to_tensor(s).transpose(2, 0, 1)
                         for s in states]),
                dtype=np.float32)
        ).to(model.device)
        policies, values = model.predict_batch(stacked)
        policies = policies.cpu().numpy()
        values = values.cpu().numpy()
        return [(policies[i], values[i]) for i in range(len(states))]
    return ev


def make_mcts(model, env, sims, leaf_batch, virtual_loss, num_players):
    """Deterministic search (eps=0, no Dirichlet) at the given leaf_batch."""
    cfg = MCTSConfig(num_simulations=sims, dirichlet_epsilon=0.0,
                     max_rollout_depth=env.max_turns,
                     leaf_batch=leaf_batch, virtual_loss=virtual_loss)
    evaluate_fn = make_eval_single(model, env) if leaf_batch <= 1 \
        else make_eval_many(model, env)
    return MCTSMaxN(config=cfg, evaluate_fn=evaluate_fn, num_players=num_players)


def sample_positions(env, k, seed=0):
    """Collect k non-terminal states from random playouts at varied depths."""
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < k:
        state = env.reset()
        depth_target = int(rng.integers(2, max(3, env.max_turns // 3)))
        for _ in range(depth_target):
            valid = env.get_valid_actions(state)
            if len(valid) == 0:
                break
            state, _, done, _ = env.step(state, int(rng.choice(valid)))
            if done:
                break
        else:
            if len(env.get_valid_actions(state)) > 0:
                positions.append(env.clone_state(state))
    return positions


def fidelity(model, env, positions, sims, leaf_batch, virtual_loss, num_players):
    """Compare visit distributions of leaf_batch=1 vs leaf_batch=B on fixed positions."""
    mcts_seq = make_mcts(model, env, sims, 1, virtual_loss, num_players)
    mcts_par = make_mcts(model, env, sims, leaf_batch,
                         virtual_loss, num_players)
    eps = 1e-12
    top1_hits, tvs, kls = 0, [], []
    for st in positions:
        # temperature=1.0 => raw visit fractions (the quantity self-play samples).
        np.random.seed(0)
        p1 = mcts_seq.search(env, env.clone_state(st), temperature=1.0)
        np.random.seed(0)
        pB = mcts_par.search(env, env.clone_state(st), temperature=1.0)
        p1 = p1 / max(p1.sum(), eps)
        pB = pB / max(pB.sum(), eps)
        top1_hits += int(np.argmax(p1) == np.argmax(pB))
        tvs.append(0.5 * np.abs(p1 - pB).sum())
        kls.append(
            float(np.sum(np.where(p1 > 0, p1 * np.log((p1 + eps) / (pB + eps)), 0.0))))
    n = len(positions)
    return {
        "positions": n,
        "top1_agreement": top1_hits / n if n else 0.0,
        "mean_tv": float(np.mean(tvs)) if tvs else 0.0,
        "max_tv": float(np.max(tvs)) if tvs else 0.0,
        "mean_kl": float(np.mean(kls)) if kls else 0.0,
    }


def strength(model, env, sims, leaf_batch, virtual_loss, num_players, games, max_moves):
    """Win rate vs random for a given leaf_batch (deterministic search, temp=0.1)."""
    mcts = make_mcts(model, env, sims, leaf_batch, virtual_loss, num_players)
    agent = mcts_agent_mp(mcts, temperature=0.1)
    res = evaluate_against_random_mp(
        env, agent, num_games=games, max_moves=max_moves)
    p, lo, hi = wilson_ci(res.candidate_wins, res.num_games)
    return {"win_rate": p, "ci_lo": lo, "ci_hi": hi,
            "wins": res.candidate_wins, "games": res.num_games}


def main():
    ap = argparse.ArgumentParser(
        description="Leaf-parallel fidelity/strength test.")
    ap.add_argument("--checkpoint", default="runs/n2_5x5_v1/ship.pt")
    ap.add_argument("--board-size", type=int, default=5)
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--walls", type=int, default=3)
    ap.add_argument("--num-channels", type=int, default=64)
    ap.add_argument("--num-res-blocks", type=int, default=4)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--leaf-batch", type=int, default=8)
    ap.add_argument("--virtual-loss", type=float, default=1.0)
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--max-moves", type=int, default=150)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    N = args.num_players
    env = QuoridorEnvMP(board_size=args.board_size, num_players=N,
                        max_turns=args.max_moves, max_walls_per_player=args.walls)
    model = QuoridorModelMP(
        board_size=args.board_size,
        action_space_size=compute_action_space_size(args.board_size),
        in_channels=3 * N + 3, num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks, num_players=N, device=args.device,
    )
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    model.load(args.checkpoint)

    print("=" * 68)
    print(f"Fidelity test | {args.checkpoint}")
    print(f"  board={args.board_size} N={N} walls={args.walls} sims={args.sims} "
          f"leaf_batch={args.leaf_batch} vloss={args.virtual_loss}")
    print("=" * 68)

    positions = sample_positions(env, args.positions)
    fid = fidelity(model, env, positions, args.sims, args.leaf_batch,
                   args.virtual_loss, N)
    print("\n[1] Search fidelity (leaf_batch=1 vs "
          f"leaf_batch={args.leaf_batch}, {fid['positions']} positions)")
    print(f"    top-1 action agreement : {fid['top1_agreement']:.1%}")
    print(
        f"    mean total-variation   : {fid['mean_tv']:.4f}  (max {fid['max_tv']:.4f})")
    print(f"    mean KL(seq || par)    : {fid['mean_kl']:.4f}")

    print(f"\n[2] Strength vs random ({args.games} games each)")
    s1 = strength(model, env, args.sims, 1, args.virtual_loss,
                  N, args.games, args.max_moves)
    sB = strength(model, env, args.sims, args.leaf_batch, args.virtual_loss, N,
                  args.games, args.max_moves)
    print(f"    leaf_batch=1            : {s1['win_rate']:.1%} "
          f"[{s1['ci_lo']:.1%}, {s1['ci_hi']:.1%}]  ({s1['wins']}/{s1['games']})")
    print(f"    leaf_batch={args.leaf_batch:<11}: {sB['win_rate']:.1%} "
          f"[{sB['ci_lo']:.1%}, {sB['ci_hi']:.1%}]  ({sB['wins']}/{sB['games']})")

    ci_overlap = not (sB['ci_hi'] < s1['ci_lo'] or s1['ci_hi'] < sB['ci_lo'])
    fidelity_high = fid['top1_agreement'] >= 0.95 and fid['mean_tv'] <= 0.05

    print("\n" + "=" * 68)
    if fidelity_high and ci_overlap:
        print("VERDICT: virtual loss is HARMLESS on this net.")
        print("  -> Work-stealing tail-fix is sufficient. Option B would be pure")
        print("     throughput (no strength gain) — build it only if you need more speed.")
    elif not ci_overlap and sB['win_rate'] < s1['win_rate']:
        print(
            f"VERDICT: leaf_batch={args.leaf_batch} is measurably WEAKER than leaf_batch=1.")
        print("  -> Option B pays off on STRENGTH: it batches across games so each")
        print("     game can run exact leaf_batch=1 search AND fill the GPU.")
    else:
        print("VERDICT: MIXED — fidelity drop present but strength CIs overlap.")
        print("  -> Borderline. Consider lowering leaf_batch (e.g. 4) or building")
        print("     Option B if you want both exact search and full batches.")
    print("=" * 68)


if __name__ == "__main__":
    main()
