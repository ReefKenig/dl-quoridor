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
naturally G-wide and stays full because finished games are refilled from the
iteration's remaining quota.

Correctness: because each game runs `VectorizedSearch` (bit-identical to
`MCTSMaxN.search` with leaf_batch=1), the per-move visit distributions are exact
sequential MCTS — no virtual-loss approximation. Value-target assignment and
augmentation reuse `assign_vector_targets` / `augment_mp`, so samples match the
sequential/parallel paths. The return contract `(samples, wins)` is identical to
`generate_parallel_self_play_mp` for a drop-in swap.
"""
import numpy as np
import torch

from src.env.quoridor_env_mp import QuoridorEnvMP
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig, VectorizedSearch
from src.mcts.self_play_mp import assign_vector_targets, augment_mp


class _GameSlot:
    """Per-game state for one concurrent self-play game in the driver."""
    __slots__ = ["game_index", "state", "trajectory", "move_count", "search"]

    def __init__(self, game_index, state):
        self.game_index = game_index
        self.state = state
        self.trajectory = []
        self.move_count = 0
        self.search = None


def _batched_eval(model, env, leaf_states):
    """One GPU forward over all pending leaves; returns list of (policy, value)."""
    stacked = torch.from_numpy(
        np.ascontiguousarray(
            np.stack([env.state_to_tensor(s).transpose(2, 0, 1)
                      for s in leaf_states]),
            dtype=np.float32)
    ).to(model.device)
    policies, values = model.predict_batch(stacked)
    return policies.cpu().numpy(), values.cpu().numpy()


def generate_vectorized_self_play_mp(model, cfg, total_games=40, vec_games=None,
                                     batch_size=None, on_games_complete=None,
                                     base_seed=0, explore_temp=1.0, final_temp=0.3,
                                     log=print):
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
        base_seed   : game g is seeded base_seed + g (Dirichlet + action sampling).
        explore_temp/final_temp : action-sampling temperature before/after
                      cfg.explore_moves (defaults match play_one_game: 1.0 then 0.3).

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

    def _new_mcts():
        return MCTSMaxN(
            config=MCTSConfig(num_simulations=cfg.mcts_simulations,
                              dirichlet_epsilon=eps,
                              max_rollout_depth=max_moves),
            evaluate_fn=None, num_players=N)

    mcts = _new_mcts()   # stateless across searches; shared by all slots

    samples = []
    wins = {}
    games_done = 0
    next_game = 0

    def _start_slot(slot_or_none):
        nonlocal next_game
        gi = next_game
        next_game += 1
        np.random.seed(base_seed + gi)   # per-game reproducibility
        state = env.reset()
        if slot_or_none is None:
            slot = _GameSlot(gi, state)
        else:
            slot = slot_or_none
            slot.game_index = gi
            slot.state = state
            slot.trajectory = []
            slot.move_count = 0
        slot.search = VectorizedSearch(mcts, env, state)
        return slot

    slots = [_start_slot(None) for _ in range(min(G, total_games))]

    while slots:
        # 1. Gather one pending leaf from every active game.
        pending = [(slot, slot.search.collect()) for slot in slots]
        pending = [(slot, leaf) for slot, leaf in pending if leaf is not None]

        # 2. One (chunked) GPU forward over all pending leaves; scatter back.
        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            pols, vals = _batched_eval(model, env, [leaf for _, leaf in chunk])
            for j, (slot, _leaf) in enumerate(chunk):
                slot.search.apply(pols[j], vals[j])

        # 3. Advance any game whose search is complete: pick a move, step the env,
        #    and finalize + refill (or retire) on game end.
        surviving = []
        for slot in slots:
            if not slot.search.done():
                surviving.append(slot)
                continue

            temp = explore_temp if slot.move_count < explore_moves else final_temp
            probs = slot.search.action_probs(temp)
            probs = probs / probs.sum()
            mover = env.get_current_player(slot.state)
            slot.trajectory.append(
                (env.state_to_tensor(slot.state), probs, mover))
            action = int(np.random.choice(len(probs), p=probs))
            slot.state, _, done, info = env.step(slot.state, action)
            slot.move_count += 1

            game_over = done or slot.move_count >= max_moves
            if not game_over:
                slot.search = VectorizedSearch(mcts, env, slot.state)
                surviving.append(slot)
                continue

            # Finalize the game -> training samples (+ mirror augmentation).
            winner = info.get("winner") if done else None
            game_samples = assign_vector_targets(
                slot.trajectory, winner, N, discount)
            aug = [augment_mp(t, p, v, N, env.board_size)
                   for (t, p, v) in game_samples]
            samples.extend(game_samples + aug)
            wins[winner] = wins.get(winner, 0) + 1
            games_done += 1
            if on_games_complete:
                on_games_complete(games_done, total_games, wins)

            if next_game < total_games:
                # refill from remaining quota
                surviving.append(_start_slot(slot))
            # else: retire this slot (drop it)

        slots = surviving

    if not samples:
        log("[VECTORIZED-MP] WARNING: no samples generated.")
    return samples, wins
