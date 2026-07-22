"""
N-player AlphaZero training loop (vector value + maxⁿ).

Self-contained: self-play with the current model's maxⁿ → train on vector targets
→ accept/reject vs the best model (candidate rotated through all seats) → eval vs
random → checkpoint best/latest. Reduces to the standard duel at N=2.
"""
import json
import os
import time
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.mcts.self_play_mp import play_one_game
from src.mcts.parallel_self_play_mp import generate_parallel_self_play_mp
from src.mcts.evaluator_mp import (evaluate_mp, evaluate_against_random_mp,
                                   mcts_agent_mp)
from src.utils.logger import make_progress_logger

logger = logging.getLogger(__name__)


def _make_progress_logger(log_path):
    """Return a log(*parts) fn that prints to console AND appends to log_path on
    disk. The disk copy keeps recording even if the Jupyter UI disconnects, so
    progress can be tailed from a terminal.

    Accepts one or more strings; multi-line messages (embedded "\\n" or extra
    positional args) are timestamped per line so the on-disk log stays aligned.
    """
    def _log(*parts):
        msg = "\n".join(str(p) for p in parts)
        print(msg, flush=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            for line in msg.splitlines() or [""]:
                f.write(f"{ts} {line}\n")
    return _log


@dataclass
class TrainingConfigMP:
    num_players: int = 4
    num_iterations: int = 20
    games_per_iteration: int = 40
    batch_size: int = 64
    train_steps_per_iter: int = 200
    mcts_simulations: int = 100
    # 0 => use mcts_simulations; else lower for faster eval
    eval_simulations: int = 0
    replay_buffer_size: int = 50_000
    max_game_moves: int = 300
    eval_games: int = 80
    eval_random_games: int = 24
    accept_margin: float = 0.05          # accept if win_rate > fair_share + margin
    discount: float = 0.97
    explore_moves: int = 15
    mcts_dirichlet_epsilon: float = 0.25
    # --- parallel self-play ---
    parallel_self_play: bool = False
    num_workers: int = 8
    inference_batch_size: int = 64
    # --- env geometry (needed by parallel workers to rebuild env) ---
    board_size: int = 5
    max_walls_per_player: int = 3
    max_turns: int = 300


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


def _mcts(model, env, cfg, sims=None):
    return MCTSMaxN(
        config=MCTSConfig(num_simulations=sims or cfg.mcts_simulations,
                          dirichlet_epsilon=getattr(
                              cfg, 'mcts_dirichlet_epsilon', 0.25),
                          max_rollout_depth=cfg.max_game_moves),
        evaluate_fn=lambda st: model.predict(env.state_to_tensor(st)),
        num_players=cfg.num_players,
    )


def training_loop_mp(env, model, make_model, cfg: TrainingConfigMP,
                     checkpoint_dir="checkpoints_mp"):
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

    # Disk log: keeps recording progress even if the Jupyter UI disconnects.
    _log = make_progress_logger(os.path.join(checkpoint_dir, "games.log"))

    _log(
        "=" * 70,
        f"training_loop_mp launched | N={cfg.num_players} board={cfg.board_size}x{cfg.board_size} "
        f"| sims={cfg.mcts_simulations} games/iter={cfg.games_per_iteration} "
        f"| parallel={cfg.parallel_self_play} workers={cfg.num_workers}",
        f"checkpoint_dir={checkpoint_dir} | eval={cfg.eval_games}+{cfg.eval_random_games} "
        f"| accept_margin={cfg.accept_margin} | buffer={cfg.replay_buffer_size}",
        "=" * 70,
    )

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
        _log(
            f"Resumed from iteration {start_iter} (checkpoint: {checkpoint_dir})")
    else:
        _log(f"Starting N={cfg.num_players} training: {cfg.num_iterations} iterations, "
             f"{cfg.games_per_iteration} games/iter, {cfg.mcts_simulations} sims")

    for it in range(start_iter, cfg.num_iterations):
        t0 = time.time()
        # --- 1. self-play ---
        _log(f"[iter {it+1}/{cfg.num_iterations}] self-play starting "
             f"({cfg.games_per_iteration} games, {cfg.mcts_simulations} sims)...")

        if cfg.parallel_self_play:
            # GPU-batched parallel self-play
            def _on_progress(done, total, w):
                if done % 5 == 0 or done == total:
                    _log(
                        f"[iter {it+1}/{cfg.num_iterations}] self-play: {done}/{total} games...")
            sp_samples, wins = generate_parallel_self_play_mp(
                model, cfg,
                num_workers=cfg.num_workers,
                total_games=cfg.games_per_iteration,
                batch_size=cfg.inference_batch_size,
                on_games_complete=_on_progress,
                base_seed=it * cfg.num_workers,
                log=_log,
            )
            buffer.add(sp_samples)
        else:
            # Sequential self-play (original path)
            sp_mcts = _mcts(model, env, cfg)
            wins = {}
            for g in range(cfg.games_per_iteration):
                samples, w = play_one_game(env, sp_mcts, cfg.num_players,
                                           max_moves=cfg.max_game_moves,
                                           discount=cfg.discount,
                                           explore_moves=cfg.explore_moves)
                buffer.add(samples)
                wins[w] = wins.get(w, 0) + 1
                # Log every 5 games so long iterations don't look stuck.
                if (g + 1) % 5 == 0 or (g + 1) == cfg.games_per_iteration:
                    elapsed = time.time() - t0
                    rate = (g + 1) / elapsed if elapsed > 0 else 0
                    _log(f"[iter {it+1}/{cfg.num_iterations}] self-play: "
                         f"{g+1}/{cfg.games_per_iteration} games "
                         f"({elapsed:.0f}s, {rate*60:.1f} games/min)")

        sp_secs = time.time() - t0
        _log(f"[iter {it+1}/{cfg.num_iterations}] self-play done: "
             f"{cfg.games_per_iteration} games ({sp_secs:.0f}s)")

        # --- 2. train ---
        t_train = time.time()
        _log(
            f"[iter {it+1}/{cfg.num_iterations}] training ({cfg.train_steps_per_iter} steps)...")
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
        train_secs = time.time() - t_train
        _log(f"[iter {it+1}/{cfg.num_iterations}] training done: "
             f"loss_p={lp:.3f} loss_v={lv:.3f} ({train_secs:.0f}s)")

        # --- 3. accept/reject vs best (candidate rotates seats) ---
        # Eval uses fewer sims than self-play when eval_simulations is set:
        # eval only measures relative strength, so it doesn't need full search
        # depth. This does NOT weaken the trained model (self-play keeps full sims).
        eval_sims = cfg.eval_simulations or cfg.mcts_simulations
        t_eval_best = time.time()
        _log(
            f"[iter {it+1}/{cfg.num_iterations}] eval vs best ({cfg.eval_games} games, {eval_sims} sims)...")
        cand = mcts_agent_mp(
            _mcts(model, env, cfg, sims=eval_sims), temperature=0.1)
        champ = mcts_agent_mp(
            _mcts(best, env, cfg, sims=eval_sims), temperature=0.1)

        def _eval_progress(done, total, r):
            elapsed = time.time() - t_eval_best
            _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs best: "
                 f"{done}/{total} games ({elapsed:.0f}s, cand {r.candidate_win_rate:.0%})")

        ev = evaluate_mp(env, cand, champ, num_games=cfg.eval_games,
                         max_moves=cfg.max_game_moves, on_progress=_eval_progress)
        accepted = ev.should_accept(threshold)
        if accepted:
            best.copy_weights_from(model)
        eval_best_secs = time.time() - t_eval_best
        _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs best done: "
             f"{100*ev.candidate_win_rate:.1f}% {'ACCEPT' if accepted else 'reject'} ({eval_best_secs:.0f}s)")

        # --- 4. eval vs random ---
        t_eval_rand = time.time()
        _log(
            f"[iter {it+1}/{cfg.num_iterations}] eval vs random ({cfg.eval_random_games} games, {eval_sims} sims)...")

        def _eval_rand_progress(done, total, r):
            elapsed = time.time() - t_eval_rand
            _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs random: "
                 f"{done}/{total} games ({elapsed:.0f}s, cand {r.candidate_win_rate:.0%})")

        evr = evaluate_against_random_mp(env, cand, num_games=cfg.eval_random_games,
                                         max_moves=cfg.max_game_moves,
                                         on_progress=_eval_rand_progress)
        eval_rand_secs = time.time() - t_eval_rand
        _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs random done: "
             f"{100*evr.candidate_win_rate:.1f}% ({eval_rand_secs:.0f}s)")

        # --- 5. checkpoint ---
        model.save(os.path.join(checkpoint_dir, "latest.pt"))
        if accepted:
            best.save(os.path.join(checkpoint_dir, "best.pt"))

        row = dict(iter=it + 1, loss_p=lp, loss_v=lv,
                   win_vs_best=ev.candidate_win_rate, accepted=accepted,
                   win_vs_random=evr.candidate_win_rate, fair=fair,
                   secs=time.time() - t0, buffer=len(buffer),
                   sp_secs=sp_secs, train_secs=train_secs,
                   eval_best_secs=eval_best_secs, eval_rand_secs=eval_rand_secs)
        history.append(row)

        # Save resume metadata
        with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
            json.dump({"completed_iterations": it + 1, "history": history}, f)

        _log(f">>> iter {it+1} | loss_p={lp:.3f} loss_v={lv:.3f} | "
             f"vs_best={100*ev.candidate_win_rate:.1f}% {'ACCEPT' if accepted else 'reject'} | "
             f"vs_rand={100*evr.candidate_win_rate:.1f}% | buf={len(buffer)} | "
             f"sp={sp_secs:.0f}s train={train_secs:.0f}s "
             f"eval_best={eval_best_secs:.0f}s eval_rand={eval_rand_secs:.0f}s | "
             f"total={row['secs']:.0f}s")
    return history
