"""Greedy-imitation pretraining data, in the exact self-play sample format.

The warm start's whole premise is that RL continues from these weights on the
same tensors and targets self-play produces - so the generator lives here,
beside `assign_vector_targets`/`augment_mp`, not in a script nobody greps when
the sample format changes.
"""
import os
import time

import numpy as np

from src.mcts.evaluator_mp import greedy_agent
from src.mcts.self_play_mp import assign_vector_targets, augment_mp, game_seed


def pretrain_report_path(checkpoint_path):
    """The report JSON that scripts/pretrain_greedy.py writes next to the .pt."""
    return os.path.splitext(checkpoint_path)[0] + "_report.json"


def generate_games(env, num_games, opening_max, max_moves, discount,
                   discount_unit, base_seed, log=print):
    """Per-game sample lists (kept separate so the holdout split is by game).

    A random opening (mostly walls - they are 98% of legal actions) diversifies
    the states greedy then races through; only greedy's own plies become policy
    targets. Timeouts are dropped by assign_vector_targets, same as self-play.
    """
    greedy = greedy_agent()
    games, n_samples, t0 = [], 0, time.time()
    for g in range(num_games):
        rng = np.random.RandomState(game_seed(base_seed, g))
        opening = rng.randint(0, opening_max + 1)
        state = env.reset()
        trajectory, plies = [], []
        move_count, winner = 0, None
        while move_count < max_moves:
            if move_count < opening:
                action = int(rng.choice(env.get_valid_actions(state)))
            else:
                action = int(greedy(env, state, move_count, rng))
                onehot = np.zeros(env.action_space_size, dtype=np.float32)
                onehot[action] = 1.0
                trajectory.append((env.state_to_tensor(state), onehot,
                                   env.get_current_player(state)))
                plies.append(move_count)
            state, _, done, info = env.step(state, action)
            move_count += 1
            if done:
                winner = info.get("winner")
                break
        samples = assign_vector_targets(
            trajectory, winner, env.num_players, discount,
            discount_unit=discount_unit, plies=plies, total_plies=move_count)
        aug = [augment_mp(t, p, v, env.num_players, env.board_size)
               for (t, p, v) in samples]
        if samples:
            games.append(samples + aug)
            n_samples += len(samples) + len(aug)
        if (g + 1) % 200 == 0:
            log(f"  {g + 1}/{num_games} games, "
                f"{n_samples} samples ({time.time() - t0:.0f}s)")
    return games


def to_arrays(games):
    flat = [s for game in games for s in game]
    if not flat:
        raise ValueError(
            "no samples from greedy game generation - every game timed out or "
            "none were played; check --games/--opening-max/--max-moves")
    S = np.stack([s[0] for s in flat]).astype(np.float32, copy=False)
    P = np.stack([s[1] for s in flat]).astype(np.float32, copy=False)
    V = np.stack([s[2] for s in flat]).astype(np.float32, copy=False)
    return S, P, V
