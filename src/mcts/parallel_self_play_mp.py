"""
Batched parallel self-play for the vector-maxⁿ (_mp) stack.

Motivation — GPU starvation
---------------------------
Sequential self-play evaluates every MCTS leaf with `model.predict(single_state)`,
i.e. one (1, C, H, W) forward pass at a time, leaving the GPU idle between
micro-calls while the CPU tree walk dominates. This module runs K worker
processes whose leaf evaluations are batched through a single GPU inference
thread, so the GPU sees fat batches and CPU tree-walks overlap with GPU compute.

The reusable machinery (multi-model batcher thread, worker↔batcher plumbing,
spawn/shutdown orchestration) lives in `batched_inference_mp.py`; this module
only provides the self-play worker game loop and the thin orchestrator. Self-play
uses a single model (`model_id` 0). Value-target assignment, augmentation and the
maxⁿ search are unchanged — workers reuse `play_one_game` from `self_play_mp.py`,
so samples are bit-for-bit identical to the sequential path.
"""
import os
import sys
import traceback

from src.mcts.batched_inference_mp import (make_batched_evaluate,
                                           run_batched_inference)


def _self_play_worker(worker_id, request_queue, response_queue, results_queue,
                      payload, project_dir, response_timeout=300.0):
    """Play the assigned self-play games, deferring every leaf eval to the GPU.

    `payload` is `(num_games, config_dict, seed)`.
    """
    num_games, config_dict, seed = payload
    try:
        os.chdir(project_dir)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        import random
        import numpy as np
        import torch
        from src.env.quoridor_env_mp import QuoridorEnvMP
        from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
        from src.mcts.self_play_mp import play_one_game

        # Deterministic, distinct per-worker seed so parallel runs are
        # reproducible and workers don't play identical games.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        N = config_dict["num_players"]
        env = QuoridorEnvMP(
            board_size=config_dict["board_size"],
            num_players=N,
            max_turns=config_dict["max_turns"],
            max_walls_per_player=config_dict["max_walls_per_player"],
        )
        batched_evaluate = make_batched_evaluate(
            worker_id, request_queue, response_queue, env, response_timeout)

        mcts = MCTSMaxN(
            config=MCTSConfig(
                num_simulations=config_dict["mcts_simulations"],
                dirichlet_epsilon=config_dict.get("mcts_dirichlet_epsilon", 0.25),
                max_rollout_depth=config_dict["max_game_moves"],
            ),
            evaluate_fn=batched_evaluate,  # model_id defaults to 0
            num_players=N,
        )

        produced = 0
        for _ in range(num_games):
            samples, winner = play_one_game(
                env, mcts, N,
                max_moves=config_dict["max_game_moves"],
                discount=config_dict["discount"],
                explore_moves=config_dict["explore_moves"],
            )
            results_queue.put(("game", worker_id, samples, winner))
            produced += len(samples)

        results_queue.put(("done", worker_id, produced))
    except Exception as e:
        print(f"[WORKER {worker_id}] CRASHED: {e}\n{traceback.format_exc()}", flush=True)
        results_queue.put(("done", worker_id, 0))


def generate_parallel_self_play_mp(model, cfg, num_workers=8, total_games=40,
                                   batch_size=64, on_games_complete=None, base_seed=0,
                                   worker_join_timeout=30.0, response_timeout=300.0,
                                   log=print):
    """Generate one iteration of self-play data with batched GPU inference.

    Args:
        model : QuoridorModelMP (the learner; stays in the main process).
        cfg   : TrainingConfigMP-like object. Reads num_players, board_size,
                max_walls_per_player, max_turns, mcts_simulations, discount,
                explore_moves, max_game_moves, mcts_dirichlet_epsilon.
        num_workers      : number of CPU game-worker processes.
        total_games      : EXACT number of games to play this iteration. Split
                           as evenly as possible across workers (the first
                           `total_games % num_workers` workers play one extra),
                           so the count matches the config regardless of how it
                           factors against num_workers.
        batch_size       : max leaves batched per GPU forward pass.
        on_games_complete: optional callback(games_done, total_games, wins_dict).
        base_seed        : worker w is seeded base_seed + w for reproducibility.
        worker_join_timeout: max seconds to wait for each worker to exit (default 30).
        response_timeout : max seconds a worker waits for a single GPU reply before
                           treating the inference batcher as dead and crashing out
                           (default 300). Prevents workers hanging indefinitely.

    Returns:
        (samples, wins) where samples is a flat list of (state_hwc, policy,
        value_vec) tuples (already augmented) and wins maps winner -> count
        (None key = draw/timeout).
    """
    # Distribute games exactly. If total < num_workers, spawn only as many
    # workers as there are games (idle workers would just add spawn overhead).
    n_workers = max(1, min(num_workers, total_games))
    base, rem = divmod(total_games, n_workers)
    games_for = [base + (1 if i < rem else 0) for i in range(n_workers)]

    config_dict = {
        "num_players": cfg.num_players,
        "board_size": getattr(cfg, "board_size", None) or model.board_size,
        "max_walls_per_player": getattr(cfg, "max_walls_per_player", 3),
        "max_turns": getattr(cfg, "max_turns", cfg.max_game_moves),
        "mcts_simulations": cfg.mcts_simulations,
        "mcts_dirichlet_epsilon": getattr(cfg, "mcts_dirichlet_epsilon", 0.25),
        "discount": cfg.discount,
        "explore_moves": cfg.explore_moves,
        "max_game_moves": cfg.max_game_moves,
    }
    payloads = [(games_for[i], config_dict, base_seed + i) for i in range(n_workers)]

    samples = []
    wins = {}
    games_done = 0

    def on_result(msg):
        nonlocal games_done
        _, _wid, game_samples, winner = msg  # ("game", worker_id, samples, winner)
        samples.extend(game_samples)
        wins[winner] = wins.get(winner, 0) + 1
        games_done += 1
        if on_games_complete:
            on_games_complete(games_done, total_games, wins)

    # Timeout per message: scales with players (N=4 games much longer than N=2)
    # and sims. Untrained N=4 9×9 at 800 sims can take 40+ min for first game.
    queue_timeout = max(1800.0, total_games * 30.0 * cfg.num_players)
    run_batched_inference(
        {0: model}, _self_play_worker, payloads, batch_size, on_result,
        log=log, response_timeout=response_timeout,
        worker_join_timeout=worker_join_timeout, queue_timeout=queue_timeout,
        label="PARALLEL-MP",
        spawn_detail=f" ({total_games} games, sims={cfg.mcts_simulations})",
    )

    if not samples:
        log("[PARALLEL-MP] WARNING: no samples generated — workers may have crashed.")

    return samples, wins
