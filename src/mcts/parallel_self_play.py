import multiprocessing as mp
import os
import sys
import threading

import numpy as np
import torch

from src.env.quoridor_env import QuoridorEnv
from src.mcts.mcts import MCTS
from src.model.network import QuoridorModel
from src.utils.config import load_config


def inference_worker(model, request_queue, response_queues, batch_size=64):
    """
    Batches requests from MCTS workers and processes them on the GPU.
    Exits when it receives a "STOP" sentinel.
    """
    model.eval()

    while True:
        batch_requests = []

        while len(batch_requests) < batch_size:
            try:
                req = request_queue.get(timeout=0.01)
                if req == "STOP":
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


def game_worker(
    worker_id, request_queue, response_queue, num_games, config_dict, results_queue
):
    """
    Plays Quoridor games using MCTS, pausing to ask the GPU for predictions.
    Runs in a spawned process — imports are resolved here.
    """
    # Ensure src is importable in spawned processes
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    from src.env.quoridor_env import QuoridorEnv
    from src.mcts.mcts import MCTS, MCTSConfig

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

    env = QuoridorEnv(board_size=board_size, max_walls_per_player=max_walls)
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

            # state_to_tensor returns (H, W, C) numpy — matches train_step's expected input
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

    results_queue.put(worker_history)


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
            ),
        )
        p.start()
        processes.append(p)

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(model, request_queue, response_queues, batch_size),
    )
    inference_thread.start()

    for p in processes:
        p.join()

    request_queue.put("STOP")
    inference_thread.join()

    global_training_buffer = []
    while not results_queue.empty():
        worker_data = results_queue.get()
        global_training_buffer.extend(worker_data)

    return global_training_buffer


if __name__ == "__main__":
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
