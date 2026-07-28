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
other seats, and results are aggregated into the same `EvalResultMP`. Both paths
now share `play_eval_game`/`tally_game`/`eval_rng` from `evaluator_mp`, so the
game loop and bookkeeping exist once and the parallel path can be validated
against the sequential one exactly.

Dirichlet noise is disabled (`dirichlet_epsilon=0`) so strength is measured at
true best-play, but the first `eval_opening_plies` moves are sampled from the
search's own visit distribution using a per-game RNG. Without that, argmax over
an ε=0 search is a deterministic function of the position and every game sharing
a seat assignment replayed identically — a 40-game gating eval was measuring 2
distinct games at N=2 and 4 at N=4.
"""
import os
import sys
import traceback
from functools import partial

from src.mcts.batched_inference_mp import (DEFAULT_BATCH_WAIT_MS,
                                           make_batched_evaluate,
                                           make_batched_evaluate_many,
                                           run_batched_inference)
from src.mcts.evaluator_mp import EvalResultMP, tally_game


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

        from src.env.quoridor_env_mp import QuoridorEnvMP
        from src.mcts.evaluator_mp import (DEFAULT_EVAL_OPENING_PLIES, eval_rng,
                                           mcts_agent_mp, play_eval_game,
                                           random_agent)
        from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig

        N = config_dict["num_players"]
        env = QuoridorEnvMP(
            board_size=config_dict["board_size"],
            num_players=N,
            max_turns=config_dict["max_turns"],
            max_walls_per_player=config_dict["max_walls_per_player"],
        )
        # leaf_batch>1 => leaf-parallel eval (waves of leaves per GPU forward); the
        # candidate (model_id 0) and champion (model_id 1) share one batcher.
        leaf_batch = int(config_dict.get("leaf_batch", 1))
        virtual_loss = float(config_dict.get("virtual_loss", 1.0))
        if leaf_batch > 1:
            evaluate_fn = make_batched_evaluate_many(
                worker_id, request_queue, response_queue, env, response_timeout)
        else:
            evaluate_fn = make_batched_evaluate(
                worker_id, request_queue, response_queue, env, response_timeout)

        def _make_mcts(model_id):
            return MCTSMaxN(
                config=MCTSConfig(
                    num_simulations=config_dict["eval_simulations"],
                    dirichlet_epsilon=0.0,  # deterministic eval — no exploration noise
                    max_rollout_depth=config_dict["max_game_moves"],
                    leaf_batch=leaf_batch,
                    virtual_loss=virtual_loss,
                ),
                evaluate_fn=partial(evaluate_fn, model_id=model_id),
                num_players=N,
            )

        # Agents come from evaluator_mp so the sequential path, this worker and
        # the test reference cannot drift apart. MCTS trees are stateless per
        # search, so build once per worker.
        opening_plies = int(config_dict.get(
            "eval_opening_plies", DEFAULT_EVAL_OPENING_PLIES))
        cand_agent = mcts_agent_mp(_make_mcts(0), temperature=0.1,
                                   opening_plies=opening_plies)
        if mode == "vs_best":
            opp_agent = mcts_agent_mp(_make_mcts(1), temperature=0.1,
                                      opening_plies=opening_plies)
        else:  # vs_random
            opp_agent = random_agent()

        for g in game_indices:
            cand_seat = g % N
            # Per-game RNG → reproducible games that differ from one another. It
            # drives the random opponent and the sampled opening, and is keyed on
            # the game index alone, so distribution across workers cannot change
            # any game's outcome.
            agents = {s: (cand_agent if s == cand_seat else opp_agent)
                      for s in range(N)}
            winner = play_eval_game(env, agents, config_dict["max_game_moves"],
                                    rng=eval_rng(base_seed, g))
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
        tally_game(res, cand_seat, winner)
        done += 1
        # Heartbeat every 5 games and on the last one, matching evaluate_mp.
        if on_progress is not None and (done % 5 == 0 or done == num_games):
            on_progress(done, num_games, res)

    queue_timeout = max(1800.0, num_games * 30.0 * N)
    run_batched_inference(
        models, _eval_worker, payloads, batch_size, on_result,
        log=log, response_timeout=response_timeout, queue_timeout=queue_timeout,
        label="EVAL-MP", batch_wait_ms=config_dict.get(
            "batch_wait_ms", DEFAULT_BATCH_WAIT_MS),
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
