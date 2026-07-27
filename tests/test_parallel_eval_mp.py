"""
Correctness tests for GPU-batched parallel evaluation (`parallel_eval_mp.py`).

Validated against the existing 5×5 checkpoints for BOTH N=2 and N=4:

  T1  inference equivalence — model.predict_batch(stack) == model.predict(each)
      within tolerance (the batcher's only numerical assumption).
  T2  exact eval equivalence (the gate) — evaluate_parallel_mp reproduces, field
      for field, an in-process reference that uses the SAME predict_batch numerics
      (candidate vs champion, ε=0, deterministic). Catches any seat-rotation,
      tally, aggregation, or model-id routing bug — if the batcher served a
      candidate leaf with the champion net, the win tallies would diverge.
  T3  exact vs-random equivalence — same, for the candidate-vs-random path
      (per-game seeding makes the random opponent reproducible).
  T4  production-parity — evaluate_parallel_mp vs the real sequential evaluate_mp
      (predict-based, ε=0); win counts match within 1 game, absorbing the tiny
      predict-vs-predict_batch float gap measured in T1.

Eval uses ε=0 (no Dirichlet noise), so the candidate/champion are deterministic
and these comparisons are exact rather than statistical. Requires torch + the
5×5 run checkpoints; skipped automatically if either is missing.
"""
import os

import numpy as np
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP
from src.mcts.evaluator_mp import EvalResultMP, evaluate_mp, mcts_agent_mp
from src.mcts.mcts_maxn import MCTSConfig, MCTSMaxN
from src.mcts.parallel_eval_mp import (evaluate_against_random_parallel_mp,
                                       evaluate_parallel_mp)

torch = pytest.importorskip("torch")
from src.model.network_mp import QuoridorModelMP  # noqa: E402  (after importorskip)

# 5×5 checkpoint geometry (network shape lives in the train scripts, not config.json;
# see scripts/reeval_ship.py). in_channels auto-derives to 3*N+3 (9 for N=2, 15 for N=4).
SPECS = {
    "n2": dict(run="runs/n2_5x5_v1", num_players=2, walls=3),
    "n4": dict(run="runs/n4_5x5_v3", num_players=4, walls=4),
}
GEOM = dict(board_size=5, action_space_size=44, num_channels=64, num_res_blocks=4)

EVAL_SIMS = 12       # small for test speed; strength is irrelevant, only equivalence
MAX_MOVES = 40
NUM_GAMES = 8        # ≥ N so every candidate seat (g % N) is exercised
BASE_SEED = 12345


def _have(spec):
    return (os.path.exists(f"{spec['run']}/latest.pt")
            and os.path.exists(f"{spec['run']}/best.pt"))


def _load(spec, ckpt):
    m = QuoridorModelMP(board_size=GEOM["board_size"],
                        action_space_size=GEOM["action_space_size"],
                        num_channels=GEOM["num_channels"],
                        num_res_blocks=GEOM["num_res_blocks"],
                        num_players=spec["num_players"], device="cpu")
    m.load(f"{spec['run']}/{ckpt}")
    return m


def _make_env(spec):
    return QuoridorEnvMP(board_size=GEOM["board_size"], num_players=spec["num_players"],
                         max_turns=300, max_walls_per_player=spec["walls"])


def _config_dict(spec):
    return {"num_players": spec["num_players"], "board_size": GEOM["board_size"],
            "max_walls_per_player": spec["walls"], "max_turns": 300,
            "eval_simulations": EVAL_SIMS, "max_game_moves": MAX_MOVES}


def _predict_batch_evaluate_fn(model, env):
    """evaluate_fn that routes through predict_batch (batch of 1) so the in-process
    reference uses byte-identical inference numerics to the parallel batcher."""
    def f(state):
        t = (torch.from_numpy(env.state_to_tensor(state)).float()
             .permute(2, 0, 1).unsqueeze(0).to(model.device))
        pol, val = model.predict_batch(t)
        return pol[0].cpu().numpy(), val[0].cpu().numpy()
    return f


def _pb_agent(model, env, N):
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=EVAL_SIMS, dirichlet_epsilon=0.0,
                          max_rollout_depth=MAX_MOVES),
        evaluate_fn=_predict_batch_evaluate_fn(model, env), num_players=N)
    return mcts_agent_mp(mcts, temperature=0.1)


def _reference(env, spec, cand, champ_or_none, mode, base_seed):
    """In-process reference mirroring _eval_worker exactly (ε=0 predict_batch MCTS,
    seat rotation g%N, per-game seed, evaluate_mp tally)."""
    N = spec["num_players"]
    cand_agent = _pb_agent(cand, env, N)
    if mode == "vs_best":
        opp_agent = _pb_agent(champ_or_none, env, N)
    else:
        def opp_agent(env, state):
            return int(np.random.choice(env.get_valid_actions(state)))

    res = EvalResultMP(num_players=N)
    for g in range(NUM_GAMES):
        cand_seat = g % N
        np.random.seed(base_seed + g)
        agents = {s: (cand_agent if s == cand_seat else opp_agent) for s in range(N)}
        state = env.reset()
        winner = None
        for _ in range(MAX_MOVES):
            cp = env.get_current_player(state)
            action = agents[cp](env, state)
            state, _, done, info = env.step(state, action)
            if done:
                winner = info.get("winner")
                break
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


def _assert_same_result(a, b):
    assert a.num_games == b.num_games
    assert a.candidate_wins == b.candidate_wins
    assert a.opponent_wins == b.opponent_wins
    assert a.draws == b.draws
    assert dict(a.seat_wins) == dict(b.seat_wins)
    assert dict(a.games_per_seat) == dict(b.games_per_seat)


@pytest.mark.parametrize("key", ["n2", "n4"])
def test_inference_equivalence(key):
    """T1: predict_batch(stack) matches per-state predict, for candidate + champion."""
    spec = SPECS[key]
    if not _have(spec):
        pytest.skip(f"missing checkpoints for {spec['run']}")
    env = _make_env(spec)
    # Collect a handful of real states by walking a random game.
    states, state = [], env.reset()
    for _ in range(6):
        states.append(env.clone_state(state))
        valid = env.get_valid_actions(state)
        state, _, done, info = env.step(state, int(np.random.choice(valid)))
        if done:
            state = env.reset()

    for ckpt in ("latest.pt", "best.pt"):
        model = _load(spec, ckpt)
        batch = torch.stack([
            torch.from_numpy(env.state_to_tensor(s)).float().permute(2, 0, 1)
            for s in states]).to(model.device)
        bp, bv = model.predict_batch(batch)
        bp, bv = bp.cpu().numpy(), bv.cpu().numpy()
        for i, s in enumerate(states):
            sp, sv = model.predict(env.state_to_tensor(s))
            assert np.allclose(bp[i], sp, atol=1e-4), f"{ckpt} policy row {i}"
            assert np.allclose(bv[i], sv, atol=1e-4), f"{ckpt} value row {i}"


@pytest.mark.parametrize("key", ["n2", "n4"])
def test_parallel_vs_best_exact(key):
    """T2 (gate): parallel gating eval == in-process predict_batch reference, exactly."""
    spec = SPECS[key]
    if not _have(spec):
        pytest.skip(f"missing checkpoints for {spec['run']}")
    cand, champ = _load(spec, "latest.pt"), _load(spec, "best.pt")
    ref = _reference(_make_env(spec), spec, cand, champ, "vs_best", BASE_SEED)
    par = evaluate_parallel_mp(cand, champ, _config_dict(spec), num_games=NUM_GAMES,
                               num_workers=2, batch_size=8, base_seed=BASE_SEED,
                               log=lambda *a: None)
    _assert_same_result(par, ref)
    # Sanity: rotation actually covered every seat.
    assert set(par.games_per_seat) == set(range(spec["num_players"]))


@pytest.mark.parametrize("key", ["n2", "n4"])
def test_parallel_vs_random_exact(key):
    """T3: parallel candidate-vs-random == per-game-seeded reference, exactly."""
    spec = SPECS[key]
    if not _have(spec):
        pytest.skip(f"missing checkpoints for {spec['run']}")
    cand = _load(spec, "latest.pt")
    ref = _reference(_make_env(spec), spec, cand, None, "vs_random", BASE_SEED)
    par = evaluate_against_random_parallel_mp(cand, _config_dict(spec), num_games=NUM_GAMES,
                                              num_workers=2, batch_size=8, base_seed=BASE_SEED,
                                              log=lambda *a: None)
    _assert_same_result(par, ref)


def test_parallel_matches_sequential_evaluate_mp():
    """T4: parallel gating eval vs the REAL sequential evaluate_mp (predict-based, ε=0).
    Win counts match within 1 game — the only slack is the predict-vs-predict_batch
    float gap bounded by T1, which can flip a rare argmax tie."""
    spec = SPECS["n2"]
    if not _have(spec):
        pytest.skip(f"missing checkpoints for {spec['run']}")
    env = _make_env(spec)
    cand, champ = _load(spec, "latest.pt"), _load(spec, "best.pt")

    def _predict_agent(model):
        mcts = MCTSMaxN(
            config=MCTSConfig(num_simulations=EVAL_SIMS, dirichlet_epsilon=0.0,
                              max_rollout_depth=MAX_MOVES),
            evaluate_fn=lambda st: model.predict(env.state_to_tensor(st)),
            num_players=spec["num_players"])
        return mcts_agent_mp(mcts, temperature=0.1)

    seq = evaluate_mp(env, _predict_agent(cand), _predict_agent(champ),
                      num_games=NUM_GAMES, max_moves=MAX_MOVES)
    par = evaluate_parallel_mp(cand, champ, _config_dict(spec), num_games=NUM_GAMES,
                               num_workers=2, batch_size=8, base_seed=BASE_SEED,
                               log=lambda *a: None)
    assert abs(par.candidate_wins - seq.candidate_wins) <= 1
    assert par.num_games == seq.num_games == NUM_GAMES
