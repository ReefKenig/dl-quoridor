import multiprocessing as mp

import numpy as np
import torch

from src.env.quoridor_env import QuoridorEnv
from src.mcts.mcts import MCTS
from src.model.network import QuoridorModel
from src.utils.config import load_config


def inference_worker(model, request_queue, response_dicts, batch_size=64):
    """
    Batches requests from MCTS workers and processes them on the GPU.
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
            except mp.queues.Empty:
                break

        if not batch_requests:
            continue

        worker_ids = [req[0] for req in batch_requests]
        tensors = [req[1] for req in batch_requests]

        batch_tensor = torch.stack(tensors).to("cuda")

        with torch.no_grad():
            policies, values = model.predict_batch(batch_tensor)

        for i, w_id in enumerate(worker_ids):
            response_dicts[w_id]["policy"] = policies[i].cpu().numpy()
            response_dicts[w_id]["value"] = values[i].item()
            response_dicts[w_id]["event"].set()


def game_worker(
    worker_id, request_queue, response_dict, num_games, config, results_queue
):
    """
    Plays Quoridor games using MCTS, pausing to ask the GPU for predictions.
    """

    def batched_evaluate(state_tensor):
        response_dict["event"].clear()
        request_queue.put((worker_id, state_tensor))
        response_dict["event"].wait()
        return response_dict["policy"], response_dict["value"]

    env = QuoridorEnv(board_size=config.board_size)
    mcts = MCTS(config=config.mcts_config(), evaluate_fn=batched_evaluate)

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

        gamma = getattr(config, "reward_decay", 0.97)

        final_reward = -1.0 if state.winner is None else 1.0

        for i, (tensor, probs, player) in enumerate(game_history):
            perspective_reward = (
                final_reward if player == state.winner else -final_reward
            )

            steps_to_end = len(game_history) - 1 - i
            discounted_reward = perspective_reward * (gamma**steps_to_end)

            worker_history.append((tensor, probs, discounted_reward))

    results_queue.put(worker_history)

    return worker_history


if __name__ == "__main__":
    mp.set_start_method("spawn")

    cfg = load_config("configs/config_5x5.json")
    NUM_WORKERS = 32

    request_queue = mp.Queue()
    results_queue = mp.Queue()
    manager = mp.Manager()

    response_dicts = {
        i: manager.dict({"policy": None, "value": None, "event": manager.Event()})
        for i in range(NUM_WORKERS)
    }

    # Start CPU workers
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=game_worker, args=(i, request_queue, response_dicts[i], 10, cfg)
        )
        p.start()
        processes.append(p)

    # Start the GPU worker
    model = QuoridorModel(board_size=cfg.board_size, action_space_size=44).to("cuda")
    inference_worker(model, request_queue, response_dicts)

    for p in processes:
        p.join()

    global_training_buffer = []
    while not results_queue.empty():
        worker_data = results_queue.get()
        global_training_buffer.extend(worker_data)

    print(
        f"Parallel Self-Play Complete! Generated {len(global_training_buffer)} states."
    )
