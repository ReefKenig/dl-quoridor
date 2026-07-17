"""
N-player AlphaZero training loop (vector value + maxⁿ).

Self-contained: self-play with the current model's maxⁿ → train on vector targets
→ accept/reject vs the best model (candidate rotated through all seats) → eval vs
random → checkpoint best/latest. Reduces to the standard duel at N=2.
"""
import json
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
    mcts_dirichlet_epsilon: float = 0.25


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
                          dirichlet_epsilon=getattr(cfg, 'mcts_dirichlet_epsilon', 0.25),
                          max_rollout_depth=cfg.max_game_moves),
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

    # --- Resume from checkpoint if available ---
    start_iter = 0
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(meta_path) and os.path.exists(latest_path):
        with open(meta_path) as f:
            meta = json.load(f)
        start_iter = meta.get("completed_iterations", 0)
        model.load(latest_path)
        if os.path.exists(best_path):
            best.load(best_path)
        else:
            best.copy_weights_from(model)
        history = meta.get("history", [])
        logger.info("Resumed from iteration %d (checkpoint: %s)",
                    start_iter, checkpoint_dir)
    else:
        print(f"Starting N={cfg.num_players} training: {cfg.num_iterations} iterations, "
              f"{cfg.games_per_iteration} games/iter, {cfg.mcts_simulations} sims",
              flush=True)

    for it in range(start_iter, cfg.num_iterations):
        t0 = time.time()
        # --- 1. self-play ---
        print(f"[iter {it+1}/{cfg.num_iterations}] self-play: 0/{cfg.games_per_iteration} games...",
              end="", flush=True)
        sp_mcts = _mcts(model, env, cfg)
        wins = {}
        for g in range(cfg.games_per_iteration):
            samples, w = play_one_game(env, sp_mcts, cfg.num_players,
                                       max_moves=cfg.max_game_moves,
                                       discount=cfg.discount,
                                       explore_moves=cfg.explore_moves)
            buffer.add(samples)
            wins[w] = wins.get(w, 0) + 1
            if (g + 1) % 10 == 0:
                print(f"\r[iter {it+1}/{cfg.num_iterations}] self-play: {g+1}/{cfg.games_per_iteration} games...",
                      end="", flush=True)
        sp_secs = time.time() - t0
        print(f"\r[iter {it+1}/{cfg.num_iterations}] self-play: {cfg.games_per_iteration}/{cfg.games_per_iteration} done ({sp_secs:.0f}s)",
              flush=True)

        # --- 2. train ---
        print(f"[iter {it+1}/{cfg.num_iterations}] training...",
              end="", flush=True)
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
        print(f" loss_p={lp:.3f} loss_v={lv:.3f}", flush=True)

        # --- 3. accept/reject vs best (candidate rotates seats) ---
        print(f"[iter {it+1}/{cfg.num_iterations}] eval vs best ({cfg.eval_games} games)...",
              end="", flush=True)
        cand = mcts_agent_mp(_mcts(model, env, cfg), temperature=0.1)
        champ = mcts_agent_mp(_mcts(best, env, cfg), temperature=0.1)
        ev = evaluate_mp(env, cand, champ, num_games=cfg.eval_games,
                         max_moves=cfg.max_game_moves)
        accepted = ev.should_accept(threshold)
        if accepted:
            best.copy_weights_from(model)
        print(f" {100*ev.candidate_win_rate:.1f}% {'ACCEPT' if accepted else 'reject'}",
              flush=True)

        # --- 4. eval vs random ---
        print(f"[iter {it+1}/{cfg.num_iterations}] eval vs random ({cfg.eval_random_games} games)...",
              end="", flush=True)
        evr = evaluate_against_random_mp(env, cand, num_games=cfg.eval_random_games,
                                         max_moves=cfg.max_game_moves)
        print(f" {100*evr.candidate_win_rate:.1f}%", flush=True)

        # --- 5. checkpoint ---
        model.save(os.path.join(checkpoint_dir, "latest.pt"))
        if accepted:
            best.save(os.path.join(checkpoint_dir, "best.pt"))

        row = dict(iter=it + 1, loss_p=lp, loss_v=lv,
                   win_vs_best=ev.candidate_win_rate, accepted=accepted,
                   win_vs_random=evr.candidate_win_rate, fair=fair,
                   secs=time.time() - t0, buffer=len(buffer))
        history.append(row)

        # Save resume metadata
        with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
            json.dump({"completed_iterations": it + 1, "history": history}, f)

        print(f">>> iter {it+1} | loss_p={lp:.3f} loss_v={lv:.3f} | "
              f"vs_best={100*ev.candidate_win_rate:.1f}% {'ACCEPT' if accepted else 'reject'} | "
              f"vs_rand={100*evr.candidate_win_rate:.1f}% | buf={len(buffer)} | {row['secs']:.0f}s",
              flush=True)
    return history
