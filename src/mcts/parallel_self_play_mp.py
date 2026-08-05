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
import multiprocessing as mp
import os
import sys
import traceback

from src.env.pathing import CURRENT_SPEC
from src.utils.schedule import game_is_masked, opponent_for_game
from src.mcts.batched_inference_mp import (make_batched_evaluate,
                                           make_batched_evaluate_many,
                                           run_batched_inference)


def _self_play_worker(worker_id, request_queue, response_queue, results_queue,
                      payload, project_dir, response_timeout=300.0):
    """Play self-play games claimed from a shared counter, deferring every leaf
    eval to the GPU.

    `payload` is `(game_counter, total_games, config_dict, base_seed)`. Instead of
    a fixed per-worker quota, every worker pulls the next game index from
    `game_counter` (an atomic shared Value) until it is exhausted. A fast worker
    keeps grabbing games rather than going idle while a slow worker finishes its
    last one, so GPU-batch concurrency stays high through the end of the iteration
    (no straggler tail — the failure mode where the last few games run alone at a
    fraction of steady-state throughput).
    """
    game_counter, total_games, config_dict, base_seed = payload
    try:
        os.chdir(project_dir)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        import random
        import numpy as np
        import torch
        # Each worker otherwise sizes its thread pool from the host's core count
        # (256), not the cgroup quota (16), so N workers fight over N*256 threads.
        torch.set_num_threads(1)
        from src.env.quoridor_env_mp import QuoridorEnvMP
        from src.mcts.evaluator_mp import greedy_agent
        from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
        from src.mcts.self_play_mp import play_one_game, game_seed

        N = config_dict["num_players"]
        env = QuoridorEnvMP(
            board_size=config_dict["board_size"],
            num_players=N,
            max_turns=config_dict["max_turns"],
            max_walls_per_player=config_dict["max_walls_per_player"],
            wall_budget=config_dict.get("wall_budget"),
            spec_version=config_dict.get("spec_version", CURRENT_SPEC),
        )
        iter_budget = config_dict.get("wall_budget")
        mask_fraction = float(config_dict.get("wall_mask_fraction", 0.0) or 0.0)
        greedy_share = float(config_dict.get("opponent_greedy_share", 0.0) or 0.0)
        past_share = float(config_dict.get("opponent_past_share", 0.0) or 0.0)
        # leaf_batch>1 => collect several leaves per MCTS wave and ship them in one
        # message (make_batched_evaluate_many); leaf_batch=1 keeps the one-leaf path.
        leaf_batch = int(config_dict.get("leaf_batch", 1))
        virtual_loss = float(config_dict.get("virtual_loss", 1.0))
        wall_candidates = int(config_dict.get("mcts_wall_candidates", 0) or 0)
        if leaf_batch > 1:
            evaluate_fn = make_batched_evaluate_many(
                worker_id, request_queue, response_queue, env, response_timeout)
        else:
            evaluate_fn = make_batched_evaluate(
                worker_id, request_queue, response_queue, env, response_timeout)

        mcts = MCTSMaxN(
            config=MCTSConfig(
                num_simulations=config_dict["mcts_simulations"],
                dirichlet_epsilon=config_dict.get(
                    "mcts_dirichlet_epsilon", 0.25),
                max_rollout_depth=config_dict["max_game_moves"],
                leaf_batch=leaf_batch,
                virtual_loss=virtual_loss,
                wall_candidates=wall_candidates,
            ),
            evaluate_fn=evaluate_fn,  # model_id defaults to 0
            num_players=N,
        )

        # The frozen champion is served by the same batcher under model_id 1,
        # exactly as the gate serves it. Built once; MCTS trees are per-search.
        past_agent = None
        if past_share:
            from functools import partial
            from src.mcts.evaluator_mp import mcts_agent_mp
            past_agent = mcts_agent_mp(
                MCTSMaxN(
                    config=MCTSConfig(
                        num_simulations=config_dict["mcts_simulations"],
                        dirichlet_epsilon=config_dict.get(
                            "mcts_dirichlet_epsilon", 0.25),
                        max_rollout_depth=config_dict["max_game_moves"],
                        leaf_batch=leaf_batch,
                        virtual_loss=virtual_loss,
                        wall_candidates=wall_candidates,
                    ),
                    evaluate_fn=partial(evaluate_fn, model_id=1),
                    num_players=N,
                ),
                temperature=0.3)

        produced = 0
        while True:
            # Claim the next game index atomically. game_index is unique across all
            # workers, so its seed is deterministic regardless of who plays it.
            with game_counter.get_lock():
                remaining = game_counter.value
                if remaining <= 0:
                    break
                game_counter.value = remaining - 1
            game_index = total_games - remaining

            # Seed per game, not per worker: ties the sample stream to the game
            # index rather than to whichever worker claimed it.
            seed = game_seed(base_seed, game_index)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Mixed curriculum: this game races wall-free, the rest play normally.
            # Keyed on game_index so the mix is identical however work is split.
            if mask_fraction and game_is_masked(game_index, mask_fraction):
                env.wall_budget = 0
            else:
                env.wall_budget = iter_budget

            # Opponent pool: an anchored game gives every seat but one to a
            # scripted racer. The model's seat rotates as it does in eval, so no
            # seat is trained on more than it is scored on.
            seat_agents = None
            opponent = opponent_for_game(game_index, greedy_share, past_share)
            if opponent in ("greedy", "past"):
                model_seat = game_index % N
                other = greedy_agent() if opponent == "greedy" else past_agent
                seat_agents = {s: other for s in range(N) if s != model_seat}

            samples, winner = play_one_game(
                env, mcts, N,
                max_moves=config_dict["max_game_moves"],
                discount=config_dict["discount"],
                # .get: workers re-import from disk, so an older run keeps its unit.
                discount_unit=config_dict.get("discount_unit", "round"),
                explore_moves=config_dict["explore_moves"],
                seat_agents=seat_agents,
            )
            results_queue.put(("game", worker_id, samples, winner, opponent))
            produced += len(samples)

        results_queue.put(("done", worker_id, produced))
    except Exception as e:
        print(
            f"[WORKER {worker_id}] CRASHED: {e}\n{traceback.format_exc()}", flush=True)
        results_queue.put(("done", worker_id, 0))


def generate_parallel_self_play_mp(model, cfg, num_workers=8, total_games=40,
                                   batch_size=64, on_games_complete=None, base_seed=0,
                                   worker_join_timeout=30.0, response_timeout=300.0,
                                   log=print, past_model=None):
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
        base_seed        : game g (0-based, across all workers) is seeded
                           game_seed(base_seed, g), so the data is reproducible
                           even though game-to-worker assignment is dynamic
                           (work-stealing).
        worker_join_timeout: max seconds to wait for each worker to exit (default 30).
        response_timeout : max seconds a worker waits for a single GPU reply before
                           treating the inference batcher as dead and crashing out
                           (default 300). Prevents workers hanging indefinitely.

    Returns:
        (samples, wins) where samples is a flat list of (state_hwc, policy,
        value_vec) tuples (already augmented) and wins maps winner -> count
        (None key = draw/timeout).
    """
    # Spawn at most one worker per game (idle workers would only add spawn
    # overhead). Games are NOT pre-divided: workers pull from a shared atomic
    # counter (work-stealing), so a fast worker keeps claiming games instead of
    # sitting idle while a slow worker finishes — this keeps the GPU batch full
    # right up to the last game and removes the end-of-iteration throughput tail.
    n_workers = max(1, min(num_workers, total_games))

    config_dict = {
        "num_players": cfg.num_players,
        "board_size": getattr(cfg, "board_size", None) or model.board_size,
        "max_walls_per_player": getattr(cfg, "max_walls_per_player", 3),
        "wall_budget": getattr(cfg, "wall_budget", None),
        "wall_mask_fraction": getattr(cfg, "wall_mask_fraction", 0.0),
        "opponent_greedy_share": getattr(cfg, "opponent_greedy_share", 0.0),
        "opponent_past_share": getattr(cfg, "opponent_past_share", 0.0),
        "mcts_wall_candidates": getattr(cfg, "mcts_wall_candidates", 0),
        "spec_version": getattr(cfg, "spec_version", CURRENT_SPEC),
        "max_turns": getattr(cfg, "max_turns", cfg.max_game_moves),
        "mcts_simulations": cfg.mcts_simulations,
        "mcts_dirichlet_epsilon": getattr(cfg, "mcts_dirichlet_epsilon", 0.25),
        "discount": cfg.discount,
        "discount_unit": getattr(cfg, "discount_unit", "round"),
        "explore_moves": cfg.explore_moves,
        "max_game_moves": cfg.max_game_moves,
        "leaf_batch": getattr(cfg, "leaf_batch", 1),
        "virtual_loss": getattr(cfg, "virtual_loss", 1.0),
    }
    # Shared work-stealing counter (spawn context matches run_batched_inference,
    # which returns the same singleton SpawnContext). All workers share this one
    # Value; each atomically decrements it to claim the next game index.
    game_counter = mp.get_context("spawn").Value("i", total_games)
    payloads = [(game_counter, total_games, config_dict, base_seed)
                for _ in range(n_workers)]

    samples = []
    sources = []
    wins = {}
    opponent_mix = {}
    samples_by_source = {}
    games_done = 0

    def on_result(msg):
        nonlocal games_done
        # ("game", worker_id, samples, winner, opponent)
        _, _wid, game_samples, winner, opponent = msg
        samples.extend(game_samples)
        # Index-aligned with samples so the buffer can draw a target mix.
        sources.extend([opponent] * len(game_samples))
        wins[winner] = wins.get(winner, 0) + 1
        # Realised, not configured: a sampler that silently stopped anchoring
        # would otherwise be invisible in the run record.
        opponent_mix[opponent] = opponent_mix.get(opponent, 0) + 1
        samples_by_source[opponent] = (samples_by_source.get(opponent, 0)
                                       + len(game_samples))
        games_done += 1
        if on_games_complete:
            on_games_complete(games_done, total_games, wins)

    # Timeout per message: scales with players (N=4 games much longer than N=2)
    # and sims. Untrained N=4 9×9 at 800 sims can take 40+ min for first game.
    queue_timeout = max(1800.0, total_games * 30.0 * cfg.num_players)
    run_batched_inference(
        ({0: model, 1: past_model} if past_model is not None else {0: model}),
        _self_play_worker, payloads, batch_size, on_result,
        log=log, response_timeout=response_timeout,
        worker_join_timeout=worker_join_timeout, queue_timeout=queue_timeout,
        label="PARALLEL-MP",
        spawn_detail=f" ({total_games} games, sims={cfg.mcts_simulations})",
    )

    if not samples:
        log("[PARALLEL-MP] WARNING: no samples generated — workers may have crashed.")

    # The buffer indexes one against the other; a drift here would mislabel the
    # source of every sample after the first mismatch instead of failing.
    if len(sources) != len(samples):
        raise RuntimeError(
            f"[PARALLEL-MP] {len(sources)} sources for {len(samples)} samples — "
            f"the two lists must stay index-aligned.")

    return samples, wins, {"opponent_mix": opponent_mix,
                           "samples_by_source": samples_by_source,
                           "sources": sources}
