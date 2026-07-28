"""
In-process vectorized self-play (Option B) for the vector-maxⁿ (_mp) stack.

Where `parallel_self_play_mp` spawns K worker processes that each block on a
shared GPU batcher (throughput ≈ workers ÷ round-trip latency, and the batch
collapses at the tail of an iteration when stragglers finish), this driver runs
G games concurrently in ONE process. Each game owns a `VectorizedSearch`
coordinator (exact sequential MCTS, one pending leaf per tree); every step, the
driver gathers one leaf from each active game, runs a SINGLE `model.predict_batch`
over all of them, scatters the results back, and advances any game whose search
finished. No multiprocessing, no queues, no per-worker blocking — the GPU batch is
G-wide for the bulk of the iteration, because a finished game is immediately
refilled from the remaining quota. It does narrow over the final G games, once
the quota is exhausted and slots start retiring without replacement; that tail is
bounded by G rather than by the whole straggler spread, which is the improvement
over the worker-pool path.

Correctness: because each game runs `VectorizedSearch` (bit-identical to
`MCTSMaxN.search` with leaf_batch=1), the per-move visit distributions are exact
sequential MCTS — no virtual-loss approximation. Value-target assignment and
augmentation reuse `assign_vector_targets` / `augment_mp`, so samples match the
sequential/parallel paths. The return contract `(samples, wins)` is identical to
`generate_parallel_self_play_mp` for a drop-in swap.

Randomness: every slot owns a `np.random.Generator` seeded from (base_seed,
game_index), used for BOTH its Dirichlet noise and its action sampling. Nothing
here touches the global `np.random` stream. That is what makes a game's data a
function of its index alone — run the same iteration at vec_games=1 or 64 and
game 7 is the same game either way. Sharing the global stream would instead make
every game's noise depend on how many others happened to be in flight.
(On CUDA, invariance is up to `predict_batch`'s batch-composition nondeterminism:
different batch shapes can differ in the last ulp. Exact on CPU.)
"""
import numpy as np
import torch

from src.env.quoridor_env_mp import QuoridorEnvMP
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig, VectorizedSearch
from src.mcts.self_play_mp import (assign_vector_targets, augment_mp, game_seed,
                                   normalize_action_probs)


class _GameSlot:
    """Per-game state for one concurrent self-play game in the driver."""
    __slots__ = ["game_index", "state", "trajectory", "move_count", "search",
                 "rng"]

    def __init__(self, game_index, state):
        self.game_index = game_index
        self.state = state
        self.trajectory = []
        self.move_count = 0
        self.search = None
        self.rng = None


def _submit(model, env, chunk):
    """Launch one forward over `chunk`'s leaves WITHOUT reading the result back.

    On CUDA the kernels are queued asynchronously, so this returns while the GPU
    is still working and the caller can descend more trees on the CPU meanwhile.
    Returns an opaque handle for `_apply`.
    """
    stacked = torch.from_numpy(
        np.ascontiguousarray(
            np.stack([env.state_to_tensor(leaf).transpose(2, 0, 1)
                      for _slot, leaf in chunk]),
            dtype=np.float32)
    ).to(model.device)
    policies, values = model.predict_batch(stacked)
    return chunk, policies, values


def _apply(inflight):
    """Read back a `_submit` handle (this is the GPU sync point) and scatter the
    policies/values into the searches that asked for them."""
    chunk, policies, values = inflight
    pols = policies.cpu().numpy()
    vals = values.cpu().numpy()
    for j, (slot, _leaf) in enumerate(chunk):
        slot.search.apply(pols[j], vals[j])


def generate_vectorized_self_play_mp(model, cfg, total_games=40, vec_games=None,
                                     batch_size=None, on_games_complete=None,
                                     base_seed=0, explore_temp=1.0, final_temp=0.3,
                                     pipeline_groups=None, log=print):
    """Generate one iteration of self-play data via in-process vectorized MCTS.

    Args:
        model : QuoridorModelMP (the learner; stays in this process).
        cfg   : TrainingConfigMP-like. Reads num_players, board_size,
                max_walls_per_player, max_turns, mcts_simulations, discount,
                explore_moves, max_game_moves, mcts_dirichlet_epsilon.
        total_games : EXACT number of games to play this iteration.
        vec_games   : number of games run concurrently (the GPU batch width).
                      Defaults to min(total_games, 64).
        batch_size  : cap on leaves per predict_batch (defaults to vec_games; a
                      round never has more than vec_games leaves anyway).
        on_games_complete : optional callback(games_done, total_games, wins).
        base_seed   : game g draws from its own Generator seeded
                      game_seed(base_seed, g) — Dirichlet noise + action sampling.
                      A game's data therefore depends on its index, never on
                      vec_games or on which games ran alongside it.
        explore_temp/final_temp : action-sampling temperature before/after
                      cfg.explore_moves (defaults match play_one_game: 1.0 then 0.3).
        pipeline_groups : how many sub-batches to split each round into so CPU
                      tree-descent overlaps GPU compute (see the loop below).
                      Defaults to 2 on CUDA, 1 elsewhere (on CPU the forward is
                      synchronous, so splitting only shrinks the batch).

    Returns:
        (samples, wins) — samples is a flat list of (state_hwc, policy, value_vec)
        tuples (already augmented), wins maps winner -> count (None = draw/timeout).
    """
    N = cfg.num_players
    env = QuoridorEnvMP(
        board_size=getattr(cfg, "board_size", None) or model.board_size,
        num_players=N,
        max_turns=getattr(cfg, "max_turns", cfg.max_game_moves),
        max_walls_per_player=getattr(cfg, "max_walls_per_player", 3),
    )
    eps = getattr(cfg, "mcts_dirichlet_epsilon", 0.25)
    discount = cfg.discount
    explore_moves = cfg.explore_moves
    max_moves = cfg.max_game_moves

    G = vec_games or min(total_games, 64)
    G = max(1, min(G, total_games))
    if batch_size is None:
        batch_size = G
    if pipeline_groups is None:
        pipeline_groups = 2 if model.device.type == "cuda" else 1
    pipeline_groups = max(1, int(pipeline_groups))

    def _new_mcts():
        return MCTSMaxN(
            config=MCTSConfig(num_simulations=cfg.mcts_simulations,
                              dirichlet_epsilon=eps,
                              max_rollout_depth=max_moves),
            evaluate_fn=None, num_players=N)

    mcts = _new_mcts()   # stateless across searches; shared by all slots

    by_game = {}          # game_index -> that game's samples (see the flush below)
    wins = {}
    games_done = 0
    next_game = 0

    def _start_slot(slot_or_none):
        nonlocal next_game
        gi = next_game
        next_game += 1
        state = env.reset()
        if slot_or_none is None:
            slot = _GameSlot(gi, state)
        else:
            slot = slot_or_none
            slot.game_index = gi
            slot.state = state
            slot.trajectory = []
            slot.move_count = 0
        # Private stream per game — never the global one, which the other G-1
        # in-flight games would interleave their draws into.
        slot.rng = np.random.default_rng(game_seed(base_seed, gi))
        slot.search = VectorizedSearch(mcts, env, state, rng=slot.rng)
        return slot

    slots = [_start_slot(None) for _ in range(min(G, total_games))]

    while slots:
        # 1+2. Descend trees and evaluate leaves, pipelined: split the active
        # slots into `pipeline_groups` and keep one forward in flight while the
        # next group's tree-walks run. Without this the round is strictly
        # serial — GPU idle during the (Python, single-core, and at 9x9/800 sims
        # dominant) descents, CPU idle during the forward. With k groups, k-1 of
        # the k forwards overlap CPU work. k=1 restores the plain behaviour.
        per_group = -(-len(slots) // pipeline_groups)   # ceil
        inflight = None
        for gs in range(0, len(slots), per_group):
            group = slots[gs:gs + per_group]
            pending = [(slot, slot.search.collect()) for slot in group]
            pending = [(slot, leaf) for slot, leaf in pending if leaf is not None]
            for i in range(0, len(pending), batch_size):
                if inflight is not None:
                    _apply(inflight)
                inflight = _submit(model, env, pending[i:i + batch_size])
        if inflight is not None:
            _apply(inflight)

        # 3. Advance any game whose search is complete: pick a move, step the env,
        #    and finalize + refill (or retire) on game end.
        surviving = []
        for slot in slots:
            if not slot.search.done():
                surviving.append(slot)
                continue

            temp = explore_temp if slot.move_count < explore_moves else final_temp
            probs = slot.search.action_probs(temp)
            probs = normalize_action_probs(probs, env, slot.state)
            mover = env.get_current_player(slot.state)
            slot.trajectory.append(
                (env.state_to_tensor(slot.state), probs, mover))
            action = int(slot.rng.choice(len(probs), p=probs))
            slot.state, _, done, info = env.step(slot.state, action)
            slot.move_count += 1

            game_over = done or slot.move_count >= max_moves
            if not game_over:
                slot.search = VectorizedSearch(
                    mcts, env, slot.state, rng=slot.rng)
                surviving.append(slot)
                continue

            # Finalize the game -> training samples (+ mirror augmentation).
            # Keyed by game index, not appended: games finish out of order and
            # the order depends on vec_games, so appending would make the output
            # sequence a function of the concurrency width. Flushed in index
            # order below.
            winner = info.get("winner") if done else None
            game_samples = assign_vector_targets(
                slot.trajectory, winner, N, discount)
            aug = [augment_mp(t, p, v, N, env.board_size)
                   for (t, p, v) in game_samples]
            by_game[slot.game_index] = game_samples + aug
            wins[winner] = wins.get(winner, 0) + 1
            games_done += 1
            if on_games_complete:
                on_games_complete(games_done, total_games, wins)

            if next_game < total_games:
                # refill from remaining quota
                surviving.append(_start_slot(slot))
            # else: retire this slot (drop it)

        slots = surviving

    # Flush in game-index order so the returned sequence is a function of
    # base_seed and total_games alone — same data whatever vec_games was.
    samples = [s for gi in sorted(by_game) for s in by_game[gi]]
    if not samples:
        log("[VECTORIZED-MP] WARNING: no samples generated.")
    return samples, wins
