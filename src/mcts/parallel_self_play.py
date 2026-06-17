import multiprocessing as mp
import os
import sys
import threading
import traceback

import numpy as np
import torch


def inference_worker(model, request_queue, response_queues, batch_size=64):
    """
    Batches requests from MCTS workers and processes them on the GPU.
    Exits when it receives a "STOP" sentinel.
    """
    model.eval()
    print("[INF_THREAD] Inference worker started, waiting for requests...", flush=True)
    batches_processed = 0

    while True:
        batch_requests = []

        while len(batch_requests) < batch_size:
            try:
                req = request_queue.get(timeout=0.1)
                if req == "STOP":
                    print(
                        f"[INF_THREAD] Got STOP after {batches_processed} batches", flush=True)
                    return
                batch_requests.append(req)
            except Exception:
                break

        if not batch_requests:
            continue

        worker_ids = [req[0] for req in batch_requests]
        tensors = [req[1] for req in batch_requests]

        batch_tensor = torch.stack(tensors).to(model.device)

        with torch.no_grad():
            policies, values = model.predict_batch(batch_tensor)

        for i, w_id in enumerate(worker_ids):
            response_queues[w_id].put(
                (policies[i].cpu().numpy(), values[i].item()))

        batches_processed += 1
        if batches_processed == 1:
            print(
                f"[INF_THREAD] First batch processed ({len(batch_requests)} requests)", flush=True)
        elif batches_processed % 100 == 0:
            print(
                f"[INF_THREAD] {batches_processed} batches processed", flush=True)


def game_worker(
    worker_id, request_queue, response_queue, num_games, config_dict, results_queue, project_dir
):
    """
    Plays Quoridor games using MCTS, pausing to ask the GPU for predictions.
    Runs in a spawned process — imports are resolved here.
    """
    try:
        print(
            f"[WORKER {worker_id}] Starting, project_dir={project_dir}", flush=True)

        # Ensure src is importable in spawned processes
        os.chdir(project_dir)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        print(f"[WORKER {worker_id}] sys.path set, importing...", flush=True)

        from src.env.quoridor_env import QuoridorEnv
        from src.mcts.mcts import MCTS, MCTSConfig

        print(
            f"[WORKER {worker_id}] Imports OK, starting {num_games} games", flush=True)

        board_size = config_dict["board_size"]
        max_walls = config_dict["max_walls_per_player"]
        mcts_cfg = config_dict["mcts"]
        gamma = config_dict["reward_decay"]

        def batched_evaluate(state):
            tensor = torch.from_numpy(
                env.state_to_tensor(state)).float().permute(2, 0, 1)
            request_queue.put((worker_id, tensor))
            policy, value = response_queue.get()
            return policy, value

        env = QuoridorEnv(board_size=board_size,
                          max_walls_per_player=max_walls)
        mcts = MCTS(
            config=MCTSConfig(
                num_simulations=mcts_cfg.get("num_simulations", 150),
                c_puct=mcts_cfg.get("c_puct", 1.41),
                temperature=mcts_cfg.get("temperature", 1.0),
                dirichlet_alpha=mcts_cfg.get("dirichlet_alpha", 0.3),
                dirichlet_epsilon=mcts_cfg.get("dirichlet_epsilon", 0.25),
                max_rollout_depth=mcts_cfg.get("max_rollout_depth", 100),
            ),
            evaluate_fn=batched_evaluate,
        )

        worker_history = []

        for _ in range(num_games):
            state = env.reset()
            game_history = []

            while not state.game_over:
                action_probs = mcts.search(env, state, temperature=1.0)

                action = np.random.choice(len(action_probs), p=action_probs)
                next_state, reward, done, _ = env.step(state, action)

                game_history.append(
                    (env.state_to_tensor(state), action_probs, state.current_player)
                )
                state = next_state

            final_reward = -1.0 if state.winner is None else 1.0

            for i, (tensor, probs, player) in enumerate(game_history):
                perspective_reward = (
                    final_reward if player == state.winner else -final_reward
                )

                steps_to_end = len(game_history) - 1 - i
                discounted_reward = perspective_reward * (gamma**steps_to_end)

                worker_history.append((tensor, probs, discounted_reward))

        print(
            f"[WORKER {worker_id}] Done, generated {len(worker_history)} samples", flush=True)
        results_queue.put(worker_history)
    except Exception as e:
        print(f"[WORKER {worker_id}] CRASHED: {e}", flush=True)
        traceback.print_exc()
        results_queue.put([])  # empty so join doesn't hang


def generate_parallel_self_play_data(
    model,
    config,
    num_workers: int = 8,
    games_per_worker: int = 5,
    batch_size: int = 64,
):
    """
    Generate self-play training data using CPU game workers + batched GPU inference.

    Returns:
        list of tuples: (state_hwc, policy_probs, discounted_value)
    """
    # Ensure spawned children can find `src` package via PYTHONPATH
    project_dir = os.getcwd()
    existing = os.environ.get("PYTHONPATH", "")
    if project_dir not in existing:
        os.environ["PYTHONPATH"] = project_dir + \
            (":" + existing if existing else "")

    print(f"[PARALLEL] project_dir={project_dir}", flush=True)
    print(
        f"[PARALLEL] PYTHONPATH={os.environ.get('PYTHONPATH', '')}", flush=True)

    ctx = mp.get_context("spawn")

    request_queue = ctx.Queue()
    results_queue = ctx.Queue()

    # Per-worker response queues (no Manager needed)
    response_queues = {i: ctx.Queue() for i in range(num_workers)}

    # Serialize config to a plain dict so it pickles cleanly across spawn
    config_dict = {
        "board_size": config.board_size,
        "max_walls_per_player": config.max_walls_per_player,
        "reward_decay": config.reward_decay,
        "mcts": config.raw.get("mcts", {}),
    }

    print(f"[PARALLEL] Spawning {num_workers} workers...", flush=True)

    processes = []
    for i in range(num_workers):
        p = ctx.Process(
            target=game_worker,
            args=(
                i,
                request_queue,
                response_queues[i],
                games_per_worker,
                config_dict,
                results_queue,
                project_dir,
            ),
        )
        p.start()
        processes.append(p)

    print(
        f"[PARALLEL] All {num_workers} workers started, launching inference thread...", flush=True)

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(model, request_queue, response_queues, batch_size),
    )
    inference_thread.start()

    print("[PARALLEL] Waiting for workers to finish...", flush=True)

    for i, p in enumerate(processes):
        p.join()
        print(
            f"[PARALLEL] Worker {i} joined (exit code: {p.exitcode})", flush=True)

    request_queue.put("STOP")
    inference_thread.join()
    print("[PARALLEL] Inference thread joined", flush=True)

    global_training_buffer = []
    while not results_queue.empty():
        worker_data = results_queue.get()
        global_training_buffer.extend(worker_data)

    return global_training_buffer


if __name__ == "__main__":
    from src.model.network import QuoridorModel
    from src.utils.config import load_config

    mp.set_start_method("spawn")

    cfg = load_config("configs/config_5x5.json")
    NUM_WORKERS = 32
    GAMES_PER_WORKER = 10

    # Load model (device handled internally by QuoridorModel)
    model = QuoridorModel(
        board_size=cfg.board_size,
        action_space_size=44,
        in_channels=11,
        device="cuda",
    )

    global_training_buffer = generate_parallel_self_play_data(
        model=model,
        config=cfg,
        num_workers=NUM_WORKERS,
        games_per_worker=GAMES_PER_WORKER,
        batch_size=64,
    )

    print(
        f"Parallel Self-Play Complete! Generated {len(global_training_buffer)} states."
    )
