"""
Shared batched GPU inference for the vector-maxⁿ (_mp) stack.

Both parallel self-play (`parallel_self_play_mp.py`) and parallel evaluation
(`parallel_eval_mp.py`) follow the same pattern to avoid GPU starvation:

  - K CPU worker processes each run their own MCTSMaxN game loop.
  - Each leaf eval ships a (C, H, W) tensor + a `model_id` to a shared request
    queue and blocks on the worker's private response queue.
  - One inference thread in the main process drains up to `batch_size` requests,
    buckets them by `model_id`, runs one `model.predict_batch(...)` forward pass
    per model, and scatters the per-row (policy, value_vec) back to each worker.

Self-play uses a single model (`model_id` 0); evaluation uses two (candidate=0,
champion=1). This module owns the reusable machinery — the multi-model batcher
thread, the worker↔batcher plumbing (`make_batched_evaluate`), process spawning,
results draining, and graceful shutdown (`run_batched_inference`). The game-loop
bodies (what a worker actually plays) live in the caller modules.

Failure handling: a worker whose GPU reply never arrives raises after
`response_timeout` instead of hanging forever; a dead batcher pushes an
`("error", …)` message so the main loop aborts immediately instead of waiting out
`queue_timeout` (worker processes cannot see the main-process `stop_flag`).
"""
import multiprocessing as mp
import os
import threading
import time
import traceback

import numpy as np
import torch


def _inference_worker(models, request_queue, response_queues, batch_size,
                      stop_flag, results_queue=None, log=print):
    """Multi-model GPU batcher thread (runs in the main process).

    `models` is a dict {model_id: QuoridorModelMP}. Each request is
    `(worker_id, model_id, chw_tensor)`; each reply is
    `(policy_np (A,), value_vec_np (num_players,))`. A drained batch may mix
    model ids, so rows are bucketed by `model_id` and each model runs one
    `predict_batch`. Returns on the "STOP" sentinel or any unhandled exception;
    on failure also pushes `("error", msg)` onto `results_queue` so the main loop
    aborts immediately (worker processes cannot observe `stop_flag`).
    """
    try:
        for m in models.values():
            m.network.eval()
        batches_done = 0
        evals_done = 0
        messages_done = 0
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

            # Each request carries EITHER a single (C,H,W) tensor or a stacked
            # (b,C,H,W) batch of leaves from one worker (leaf-parallel). Expand every
            # request into individual rows, run one forward per model_id, then regroup
            # replies per request so a stacked request gets a single list reply.
            rows = []                 # (model_id, tensor_chw)
            n_rows_per_req = []       # leaves contributed by each request
            for (_w_id, model_id, tensor) in batch:
                if tensor.dim() == 4:
                    for k in range(tensor.shape[0]):
                        rows.append((model_id, tensor[k]))
                    n_rows_per_req.append(tensor.shape[0])
                else:
                    rows.append((model_id, tensor))
                    n_rows_per_req.append(1)

            by_model = {}
            for pos, (model_id, t) in enumerate(rows):
                by_model.setdefault(model_id, []).append((pos, t))
            row_replies = [None] * len(rows)
            for model_id, mrows in by_model.items():
                stacked = torch.stack([t for (_pos, t) in mrows]).to(
                    models[model_id].device)
                policies, values = models[model_id].predict_batch(
                    stacked)  # (b,A),(b,N)
                policies = policies.cpu().numpy()
                values = values.cpu().numpy()
                for j, (pos, _t) in enumerate(mrows):
                    row_replies[pos] = (policies[j], values[j])

            # Regroup rows back to their originating request; a stacked (4-dim)
            # request gets a single LIST reply, a single request gets one tuple.
            cursor = 0
            for req_i, (w_id, _model_id, tensor) in enumerate(batch):
                n = n_rows_per_req[req_i]
                if tensor.dim() == 4:
                    response_queues[w_id].put(row_replies[cursor:cursor + n])
                else:
                    response_queues[w_id].put(row_replies[cursor])
                cursor += n

            batches_done += 1
            n_leaves = len(rows)
            evals_done += n_leaves
            messages_done += len(batch)
            if evals_done % 50_000 < n_leaves:
                elapsed = time.time() - t_start
                rate = evals_done / elapsed if elapsed > 0 else 0
                msg_rate = messages_done / elapsed if elapsed > 0 else 0
                avg_batch = evals_done / batches_done
                leaves_per_msg = evals_done / messages_done if messages_done else 0
                log(f"  [GPU] {evals_done:,} evals ({batches_done} batches, "
                    f"avg {avg_batch:.0f}/batch, {rate:.0f} evals/s, "
                    f"{msg_rate:.0f} msg/s, {leaves_per_msg:.1f} leaves/msg, {elapsed:.0f}s)")

            if stop_flag.is_set():
                return
    except Exception as e:
        log(f"[GPU INFERENCE] CRASHED: {e}\n{traceback.format_exc()}")
        stop_flag.set()
        # Unblock the main loop right away — it is waiting on results_queue and
        # would otherwise sit until the (multi-thousand-second) timeout, since
        # the worker processes cannot observe stop_flag.
        if results_queue is not None:
            try:
                results_queue.put(("error", f"GPU inference thread died: {e}"))
            except Exception:
                pass


def make_batched_evaluate(worker_id, request_queue, response_queue, env,
                          response_timeout=300.0):
    """Build the `evaluate_fn(state, model_id=0)` a worker hands to its MCTS.

    Converts HWC numpy → CHW float tensor, ships `(worker_id, model_id, tensor)`
    to the batcher, and blocks (bounded) on the worker's response queue. A single
    batched forward pass returns in well under a second, so a multi-minute stall
    means the batcher died — raise so the worker exits instead of hanging forever
    and burning the whole results-queue timeout.
    """
    def batched_evaluate(state, model_id=0):
        tensor = torch.from_numpy(
            env.state_to_tensor(state)).float().permute(2, 0, 1)
        request_queue.put((worker_id, model_id, tensor))
        try:
            policy, value_vec = response_queue.get(timeout=response_timeout)
        except Exception:
            raise RuntimeError(
                f"worker {worker_id}: no GPU response within {response_timeout:.0f}s "
                f"— inference batcher likely crashed (check games.log).")
        return np.asarray(policy), np.asarray(value_vec)
    return batched_evaluate


def make_batched_evaluate_many(worker_id, request_queue, response_queue, env,
                               response_timeout=300.0):
    """Build the leaf-parallel `evaluate_many(states, model_id=0)` a worker hands to
    its MCTS. Stacks a LIST of states into one (b, C, H, W) request, ships it as a
    single queue message, and blocks (bounded) for the batcher's list reply. Used
    when leaf_batch > 1: each MCTS wave submits its leaves in one round-trip instead
    of one per leaf, which is what actually fills the GPU.
    """
    def evaluate_many(states, model_id=0):
        chw = [torch.from_numpy(env.state_to_tensor(s)).float().permute(2, 0, 1)
               for s in states]
        stacked = torch.stack(chw)   # (b, C, H, W)
        request_queue.put((worker_id, model_id, stacked))
        try:
            replies = response_queue.get(
                timeout=response_timeout)  # list of (p, v)
        except Exception:
            raise RuntimeError(
                f"worker {worker_id}: no GPU response within {response_timeout:.0f}s "
                f"— inference batcher likely crashed (check games.log).")
        return [(np.asarray(p), np.asarray(v)) for (p, v) in replies]
    return evaluate_many


def run_batched_inference(models, worker_target, per_worker_payloads, batch_size,
                          on_result, log=print, response_timeout=300.0,
                          worker_join_timeout=30.0, queue_timeout=1800.0,
                          label="PARALLEL-MP", spawn_detail=""):
    """Spawn game workers + one multi-model GPU batcher, then drain results.

    Args:
        models : {model_id: QuoridorModelMP} held in the main process (the batcher
                 thread shares them in-memory; nothing is serialized to workers).
        worker_target : a module-level (picklable) function with signature
                 ``fn(worker_id, request_queue, response_queue, results_queue,
                 payload, project_dir, response_timeout)``. It must emit a
                 ``("done", worker_id, ...)`` message when finished and may emit
                 any number of other result messages consumed by ``on_result``.
        per_worker_payloads : list, one opaque payload per worker
                 (``len`` == number of workers spawned).
        batch_size : max leaves per GPU forward pass.
        on_result(msg) : called for every results message that is not
                 ``("done", …)`` or ``("error", …)``.
        queue_timeout : max seconds to wait for the next results message before
                 declaring the batcher wedged and aborting.

    Raises RuntimeError if the GPU batcher thread dies. Returns None (all
    aggregation happens in the caller's ``on_result`` closure).
    """
    num_workers = len(per_worker_payloads)
    project_dir = os.getcwd()
    existing = os.environ.get("PYTHONPATH", "")
    if project_dir not in existing:
        os.environ["PYTHONPATH"] = project_dir + \
            (os.pathsep + existing if existing else "")

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    results_queue = ctx.Queue()
    response_queues = {i: ctx.Queue() for i in range(num_workers)}

    processes = []
    for i in range(num_workers):
        p = ctx.Process(
            target=worker_target,
            args=(i, request_queue, response_queues[i], results_queue,
                  per_worker_payloads[i], project_dir, response_timeout),
        )
        p.start()
        processes.append(p)

    log(f"[{label}] {num_workers} workers spawned{spawn_detail}, GPU batcher starting...")

    stop_flag = threading.Event()
    inference_thread = threading.Thread(
        target=_inference_worker,
        args=(models, request_queue, response_queues, batch_size, stop_flag,
              results_queue, log),
        daemon=True,
    )
    inference_thread.start()

    workers_done = 0
    inference_error = None
    try:
        while workers_done < num_workers:
            try:
                msg = results_queue.get(timeout=queue_timeout)
            except Exception:
                log(f"[{label}] WARNING: results queue timeout after {queue_timeout:.0f}s "
                    f"({workers_done}/{num_workers} workers done). The GPU inference "
                    f"thread is likely wedged (no exception, no progress) — check "
                    f"nvidia-smi for orphaned worker processes from a hard-killed run.")
                break
            tag = msg[0]
            if tag == "done":
                workers_done += 1
            elif tag == "error":
                # The batcher crashed and told us directly, so we don't wait out
                # queue_timeout. Abort now and re-raise below (surfaces to caller).
                inference_error = msg[1]
                log(f"[{label}] ABORTING: {inference_error}")
                break
            else:
                on_result(msg)
    finally:
        # Graceful shutdown: signal workers + batcher to stop.
        request_queue.put("STOP")
        stop_flag.set()

        for i, p in enumerate(processes):
            p.join(timeout=worker_join_timeout)
            if p.is_alive():
                log(f"[{label}] Worker {i} did not exit after {worker_join_timeout}s; terminating.")
                p.terminate()
                p.join(timeout=5.0)
                if p.is_alive():
                    log(f"[{label}] Worker {i} still alive after terminate; killing.")
                    p.kill()
                    p.join()

        inference_thread.join(timeout=5.0)
        if inference_thread.is_alive():
            log(f"[{label}] WARNING: Inference thread did not exit after 5s "
                f"(daemon thread will be killed on shutdown).")

    if inference_error is not None:
        raise RuntimeError(f"parallel run aborted: {inference_error}")
