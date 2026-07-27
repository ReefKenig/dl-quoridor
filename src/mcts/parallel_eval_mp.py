"""
GPU-batched parallel evaluation for the vector-maxⁿ (_mp) stack.

Eval is otherwise the sequential bottleneck: `evaluate_mp` plays one game at a
time, each MCTS leaf a single `model.predict` call, so the GPU starves exactly
like sequential self-play did. This module runs eval games across worker
processes whose leaf evaluations are batched through the shared GPU batcher in
`batched_inference_mp.py` — the same machinery self-play uses — but serving TWO
models: the candidate (`model_id` 0) and the champion (`model_id` 1). The random
opponent never touches the GPU.

The observable behaviour is identical to `evaluate_mp`: the candidate rotates
through seats by game index (`cand_seat = g % N`), the same opponent fills the
other seats, and results are aggregated into the same `EvalResultMP` with the
same tally rules and `on_progress` cadence. Dirichlet noise is disabled
(`dirichlet_epsilon=0`) so strength is measured deterministically at true
best-play; combined with a per-game seed this makes eval reproducible and lets
the parallel path be validated against the sequential one exactly.
"""
import os
import sys
import traceback
from functools import partial

from src.mcts.batched_inference_mp import (make_batched_evaluate,
                                           run_batched_inference)
from src.mcts.evaluator_mp import EvalResultMP


def _eval_worker(worker_id, request_queue, response_queue, results_queue,
                 payload, project_dir, response_timeout=300.0):
    """Play the assigned eval games; candidate rotates seats by game index.

    `payload` is `(mode, game_indices, config_dict, base_seed)` with
    `mode in {"vs_best", "vs_random"}`. Emits one
    `("game", worker_id, g, cand_seat, winner)` per game, then `("done", worker_id)`.
    """
    mode, game_indices, config_dict, base_seed = payload
    try:
        os.chdir(project_dir)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        import numpy as np
        from src.env.quoridor_env_mp import QuoridorEnvMP
        from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig

        N = config_dict["num_players"]
        env = QuoridorEnvMP(
            board_size=config_dict["board_size"],
            num_players=N,
            max_turns=config_dict["max_turns"],
            max_walls_per_player=config_dict["max_walls_per_player"],
        )
        batched_evaluate = make_batched_evaluate(
            worker_id, request_queue, response_queue, env, response_timeout)

        def _make_mcts(model_id):
            return MCTSMaxN(
                config=MCTSConfig(
                    num_simulations=config_dict["eval_simulations"],
                    dirichlet_epsilon=0.0,  # deterministic eval — no exploration noise
                    max_rollout_depth=config_dict["max_game_moves"],
                ),
                evaluate_fn=partial(batched_evaluate, model_id=model_id),
                num_players=N,
            )

        # Agents mirror evaluator_mp.mcts_agent_mp / random_agent: argmax over the
        # search's visit-count probabilities (temperature=0.1). MCTS trees are
        # stateless per search, so build once per worker.
        cand_mcts = _make_mcts(0)

        def cand_agent(env, state):
            probs = cand_mcts.search(env, state, temperature=0.1)
            return int(np.argmax(probs))

        if mode == "vs_best":
            champ_mcts = _make_mcts(1)

            def opp_agent(env, state):
                probs = champ_mcts.search(env, state, temperature=0.1)
                return int(np.argmax(probs))
        else:  # vs_random
            def opp_agent(env, state):
                return int(np.random.choice(env.get_valid_actions(state)))

        for g in game_indices:
            cand_seat = g % N
            # Per-game seed → reproducible games (matters for the random opponent;
            # the eps=0 candidate/champion are already deterministic). Distribution
            # across workers therefore cannot change any game's outcome.
            np.random.seed(base_seed + g)
            agents = {s: (cand_agent if s == cand_seat else opp_agent)
                      for s in range(N)}
            state = env.reset()
            winner = None
            for _ in range(config_dict["max_game_moves"]):
                cp = env.get_current_player(state)
                action = agents[cp](env, state)
                state, _, done, info = env.step(state, action)
                if done:
                    winner = info.get("winner")
                    break
            results_queue.put(("game", worker_id, g, cand_seat, winner))

        results_queue.put(("done", worker_id))
    except Exception as e:
        print(f"[EVAL-WORKER {worker_id}] CRASHED: {e}\n{traceback.format_exc()}", flush=True)
        results_queue.put(("done", worker_id))


def _run_eval(models, mode, config_dict, num_games, num_workers, batch_size,
              on_progress, base_seed, log, response_timeout):
    """Shared driver for both eval modes; returns an EvalResultMP (see evaluate_mp)."""
    N = config_dict["num_players"]
    n_workers = max(1, min(num_workers, num_games))
    # Round-robin game INDICES across workers. Outcome depends only on the game
    # index (via cand_seat = g % N and the per-game seed), never on which worker
    # runs it, so the aggregate tally matches the sequential evaluate_mp exactly.
    buckets = [[] for _ in range(n_workers)]
    for g in range(num_games):
        buckets[g % n_workers].append(g)
    payloads = [(mode, buckets[i], config_dict, base_seed) for i in range(n_workers)]

    res = EvalResultMP(num_players=N)
    done = 0

    def on_result(msg):
        nonlocal done
        _, _wid, _g, cand_seat, winner = msg  # ("game", wid, g, cand_seat, winner)
        res.num_games += 1
        res.games_per_seat[cand_seat] = res.games_per_seat.get(cand_seat, 0) + 1
        if winner is None:
            res.draws += 1
        elif winner == cand_seat:
            res.candidate_wins += 1
            res.seat_wins[cand_seat] = res.seat_wins.get(cand_seat, 0) + 1
        else:
            res.opponent_wins += 1
        done += 1
        # Heartbeat every 5 games and on the last one, matching evaluate_mp.
        if on_progress is not None and (done % 5 == 0 or done == num_games):
            on_progress(done, num_games, res)

    queue_timeout = max(1800.0, num_games * 30.0 * N)
    run_batched_inference(
        models, _eval_worker, payloads, batch_size, on_result,
        log=log, response_timeout=response_timeout, queue_timeout=queue_timeout,
        label="EVAL-MP",
        spawn_detail=f" ({num_games} {mode} games, sims={config_dict['eval_simulations']})",
    )
    return res


def evaluate_parallel_mp(cand_model, champ_model, config_dict, num_games=24,
                         num_workers=8, batch_size=64, on_progress=None,
                         base_seed=0, log=print, response_timeout=300.0):
    """Parallel candidate-vs-champion gating eval. Drop-in for `evaluate_mp`.

    `config_dict` needs: num_players, board_size, max_walls_per_player, max_turns,
    eval_simulations, max_game_moves. Returns an `EvalResultMP`.
    """
    return _run_eval({0: cand_model, 1: champ_model}, "vs_best", config_dict,
                     num_games, num_workers, batch_size, on_progress, base_seed,
                     log, response_timeout)


def evaluate_against_random_parallel_mp(cand_model, config_dict, num_games=24,
                                        num_workers=8, batch_size=64,
                                        on_progress=None, base_seed=0, log=print,
                                        response_timeout=300.0):
    """Parallel candidate-vs-random eval. Drop-in for `evaluate_against_random_mp`."""
    return _run_eval({0: cand_model}, "vs_random", config_dict,
                     num_games, num_workers, batch_size, on_progress, base_seed,
                     log, response_timeout)
