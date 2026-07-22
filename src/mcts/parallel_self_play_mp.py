"""
Batched parallel self-play for the vector-maxⁿ (_mp) stack.

Motivation — GPU starvation
---------------------------
`training_mp.py` plays self-play games sequentially: every MCTS leaf calls
`model.predict(single_state)`, i.e. one (1, C, H, W) forward pass at a time.
On a GPU that leaves the device idle between micro-calls and the launch
overhead dominates — the CPU-side MCTS tree walk becomes the bottleneck while
the GPU sits ~empty.

This module fixes that the same way `parallel_self_play.py` does for the old
single-player stack, but for the N-player vector-value network:

  - N CPU processes ("game workers") each run their own MCTSMaxN game loop.
  - Whenever a worker needs a leaf evaluated it ships the (C, H, W) tensor to a
    single shared request queue and blocks on its private response queue.
  - One "inference worker" thread in the main process drains up to `batch_size`
    requests, stacks them into one (B, C, H, W) batch, runs a single
    `model.predict_batch(...)` forward pass, and scatters the per-row
    (policy, value_vec) back to each worker's response queue.

So the GPU sees fat batches instead of a stream of singletons, and the CPU
tree-walk of many workers overlaps with GPU compute.

Value-target assignment, augmentation and the max^n search are unchanged — the
workers reuse `play_one_game` from `self_play_mp.py`, so the samples produced
here are bit-for-bit the same shape/semantics as the sequential path.
"""
import multiprocessing as mp
import os
import sys
import threading
import traceback

import torch


def _inference_worker(model, request_queue, response_queues, batch_size, stop_flag, log=print):
    """Drain requests, batch them, run one GPU forward pass, scatter replies.

    Each request is (worker_id, chw_tensor). Each reply is
    (policy_np (A,), value_vec_np (num_players,)).
    Returns when it sees the "STOP" sentinel or on any unhandled exception.
    `log` is a callable(msg) used for progress (defaults to print; the training
    loop passes a disk-backed logger so GPU stats reach games.log too).
    """
    try:
        import time
        model.network.eval()
        batches_done = 0
        evals_done = 0
        t_start = time.time()

        while True:
            batch = []
            # Block for the first item so we don't busy-spin, then greedily drain
            # whatever else is already queued up to batch_size.
            try:
                first = request_queue.get(timeout=0.2)
            except Exception:
                if stop_flag.is_set():
                    return
                continue
            if first == "STOP":
                return
            batch.append(first)
            while len(batch) < batch_size:
                try:
                    req = request_queue.get_nowait()
                except Exception:
                    break
                if req == "STOP":
                    # Finish the batch we have, then stop.
                    stop_flag.set()
                    break
                batch.append(req)

            worker_ids = [r[0] for r in batch]
            tensors = [r[1] for r in batch]
            batch_tensor = torch.stack(tensors).to(model.device)

            policies, values = model.predict_batch(batch_tensor)  # (B, A), (B, N)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

            for i, w_id in enumerate(worker_ids):
                response_queues[w_id].put((policies[i], values[i]))

            batches_done += 1
            evals_done += len(batch)
            if evals_done % 50_000 < len(batch):
                elapsed = time.time() - t_start
                rate = evals_done / elapsed if elapsed > 0 else 0
                avg_batch = evals_done / batches_done
                log(f"  [GPU] {evals_done:,} evals ({batches_done} batches, "
                    f"avg {avg_batch:.0f}/batch, {rate:.0f} evals/s, {elapsed:.0f}s)")

            if stop_flag.is_set():
                return
    except Exception as e:
        log(f"[GPU INFERENCE] CRASHED: {e}\n{traceback.format_exc()}")
        stop_flag.set()


def _game_worker(worker_id, request_queue, response_queue, num_games,
                 config_dict, results_queue, project_dir, seed):
    """Play `num_games` self-play games, deferring every leaf eval to the GPU."""
    try:
        os.chdir(project_dir)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        import random
        import numpy as np
        from src.env.quoridor_env_mp import QuoridorEnvMP
        from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
        from src.mcts.self_play_mp import play_one_game

        # Deterministic per-worker seed so parallel runs are reproducible.
        # Distinct per worker so workers don't play identical games.
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

        def batched_evaluate(state):
            # HWC numpy -> CHW float tensor, hand off to the GPU batcher.
            tensor = torch.from_numpy(env.state_to_tensor(state)).float().permute(2, 0, 1)
            request_queue.put((worker_id, tensor))
            policy, value_vec = response_queue.get()
            return np.asarray(policy), np.asarray(value_vec)

        mcts = MCTSMaxN(
            config=MCTSConfig(
                num_simulations=config_dict["mcts_simulations"],
                dirichlet_epsilon=config_dict.get("mcts_dirichlet_epsilon", 0.25),
                max_rollout_depth=config_dict["max_game_moves"],
            ),
            evaluate_fn=batched_evaluate,
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
                                   worker_join_timeout=30.0, log=print):
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
    num_workers = n_workers
    project_dir = os.getcwd()
    existing = os.environ.get("PYTHONPATH", "")
    if project_dir not in existing:
        os.environ["PYTHONPATH"] = project_dir + (os.pathsep + existing if existing else "")

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

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    results_queue = ctx.Queue()
    response_queues = {i: ctx.Queue() for i in range(num_workers)}

    processes = []
    for i in range(num_workers):
        p = ctx.Process(
            target=_game_worker,
            args=(i, request_queue, response_queues[i], games_for[i],
                  config_dict, results_queue, project_dir, base_seed + i),
        )
        p.start()
        processes.append(p)

    log(f"[PARALLEL-MP] {num_workers} workers spawned "
        f"({total_games} games, sims={cfg.mcts_simulations}), GPU batcher starting...")

    stop_flag = threading.Event()
    inference_thread = threading.Thread(
        target=_inference_worker,
        args=(model, request_queue, response_queues, batch_size, stop_flag, log),
        daemon=True,
    )
    inference_thread.start()

    samples = []
    wins = {}
    games_done = 0
    workers_done = 0
    # Timeout per message: 9×9 with 800 sims can take 10+ min for the first game.
    queue_timeout = max(600.0, total_games * 30.0)
    try:
        while workers_done < num_workers:
            try:
                msg = results_queue.get(timeout=queue_timeout)
            except Exception:
                log(f"[PARALLEL-MP] WARNING: results queue timeout after {queue_timeout:.0f}s. "
                    f"Aborting ({workers_done}/{num_workers} workers done, "
                    f"{games_done}/{total_games} games).")
                break
            if msg[0] == "game":
                _, _wid, game_samples, winner = msg
                samples.extend(game_samples)
                wins[winner] = wins.get(winner, 0) + 1
                games_done += 1
                if on_games_complete:
                    on_games_complete(games_done, total_games, wins)
            elif msg[0] == "done":
                workers_done += 1
    finally:
        # Graceful shutdown: signal workers to stop.
        request_queue.put("STOP")
        stop_flag.set()

        # Wait for processes with timeout; terminate stragglers.
        for i, p in enumerate(processes):
            p.join(timeout=worker_join_timeout)
            if p.is_alive():
                log(f"[PARALLEL-MP] Worker {i} did not exit after {worker_join_timeout}s; terminating.")
                p.terminate()
                p.join(timeout=5.0)
                if p.is_alive():
                    log(f"[PARALLEL-MP] Worker {i} still alive after terminate; killing.")
                    p.kill()
                    p.join()

        # Wait for inference thread with timeout.
        inference_thread.join(timeout=5.0)
        if inference_thread.is_alive():
            log("[PARALLEL-MP] WARNING: Inference thread did not exit after 5s "
                "(daemon thread will be killed on shutdown).")

    if not samples:
        log("[PARALLEL-MP] WARNING: no samples generated — workers may have crashed.")

    return samples, wins

