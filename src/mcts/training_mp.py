"""
N-player AlphaZero training loop (vector value + maxⁿ).

Self-contained: self-play with the current model's maxⁿ → train on vector targets
→ accept/reject vs the best model (candidate rotated through all seats) → eval vs
random → checkpoint best/latest. Reduces to the standard duel at N=2.
"""
import os
import copy
import time
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.mcts.self_play_mp import play_one_game
from src.mcts.evaluator_mp import (evaluate_mp, evaluate_against_random_mp,
                                   mcts_agent_mp, random_agent)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfigMP:
    num_players: int = 4
    num_iterations: int = 20
    games_per_iteration: int = 40
    batch_size: int = 64
    train_steps_per_iter: int = 200
    mcts_simulations: int = 100
    replay_buffer_size: int = 50_000
    max_game_moves: int = 300
    eval_games: int = 80
    eval_random_games: int = 24
    accept_margin: float = 0.05          # accept if win_rate > fair_share + margin
    discount: float = 0.97
    explore_moves: int = 15


class ReplayBufferMP:
    def __init__(self, max_size=50_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, samples):
        # each sample: (state_hwc, policy, value_vec)
        self.buffer.extend(samples)

    def sample_batch(self, batch_size):
        n = min(batch_size, len(self.buffer))
        idx = np.random.choice(len(self.buffer), n, replace=False)
        b = [self.buffer[i] for i in idx]
        S = np.array([x[0] for x in b], np.float32)
        P = np.array([x[1] for x in b], np.float32)
        V = np.array([x[2] for x in b], np.float32)
        return S, P, V

    def __len__(self):
        return len(self.buffer)


def _mcts(model, env, cfg):
    return MCTSMaxN(
        config=MCTSConfig(num_simulations=cfg.mcts_simulations,
                          dirichlet_epsilon=0.25, max_rollout_depth=cfg.max_game_moves),
        evaluate_fn=lambda st: model.predict(env.state_to_tensor(st)),
        num_players=cfg.num_players,
    )


def training_loop_mp(env, model, make_model, cfg: TrainingConfigMP,
                     checkpoint_dir="checkpoints_mp", log_every=1):
    """
    env   : QuoridorEnvMP (num_players == cfg.num_players)
    model : QuoridorModelMP (the learner)
    make_model : zero-arg callable returning a fresh QuoridorModelMP (for `best`)
    """
    assert env.num_players == cfg.num_players
    os.makedirs(checkpoint_dir, exist_ok=True)
    buffer = ReplayBufferMP(cfg.replay_buffer_size)
    best = make_model()
    best.copy_weights_from(model)
    fair = 1.0 / cfg.num_players
    threshold = fair + cfg.accept_margin
    history = []

    for it in range(cfg.num_iterations):
        t0 = time.time()
        # --- 1. self-play ---
        sp_mcts = _mcts(model, env, cfg)
        wins = {}
        for _ in range(cfg.games_per_iteration):
            samples, w = play_one_game(env, sp_mcts, cfg.num_players,
                                       max_moves=cfg.max_game_moves,
                                       discount=cfg.discount,
                                       explore_moves=cfg.explore_moves)
            buffer.add(samples)
            wins[w] = wins.get(w, 0) + 1

        # --- 2. train ---
        lp = lv = 0.0
        steps = cfg.train_steps_per_iter if len(
            buffer) >= cfg.batch_size else 0
        for _ in range(steps):
            S, P, V = buffer.sample_batch(cfg.batch_size)
            a, b = model.train_step(S, P, V)
            lp += a
            lv += b
        lp /= max(steps, 1)
        lv /= max(steps, 1)

        # --- 3. accept/reject vs best (candidate rotates seats) ---
        cand = mcts_agent_mp(_mcts(model, env, cfg), temperature=0.1)
        champ = mcts_agent_mp(_mcts(best, env, cfg), temperature=0.1)
        ev = evaluate_mp(env, cand, champ, num_games=cfg.eval_games,
                         max_moves=cfg.max_game_moves)
        accepted = ev.should_accept(threshold)
        if accepted:
            best.copy_weights_from(model)

        # --- 4. eval vs random ---
        evr = evaluate_against_random_mp(env, cand, num_games=cfg.eval_random_games,
                                         max_moves=cfg.max_game_moves)

        # --- 5. checkpoint ---
        model.save(os.path.join(checkpoint_dir, "latest.pt"))
        if accepted:
            best.save(os.path.join(checkpoint_dir, "best.pt"))

        row = dict(iter=it + 1, loss_p=lp, loss_v=lv,
                   win_vs_best=ev.candidate_win_rate, accepted=accepted,
                   win_vs_random=evr.candidate_win_rate, fair=fair,
                   secs=time.time() - t0, buffer=len(buffer))
        history.append(row)
        if (it % log_every) == 0:
            logger.info(
                "iter %d | loss_p=%.3f loss_v=%.3f | vs_best=%.1f%% (acc>%.1f%%) %s | "
                "vs_rand=%.1f%% (fair=%.1f%%) | buf=%d | %.0fs",
                it + 1, lp, lv, 100 * ev.candidate_win_rate, 100 * threshold,
                "ACCEPT" if accepted else "reject",
                100 * evr.candidate_win_rate, 100 * fair, len(buffer), row["secs"])
    return history
