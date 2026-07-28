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
from src.mcts.vectorized_self_play_mp import generate_vectorized_self_play_mp
from src.mcts.evaluator_mp import (DEFAULT_EVAL_OPENING_PLIES, evaluate_mp,
                                   evaluate_against_random_mp, mcts_agent_mp)
from src.mcts.parallel_eval_mp import (evaluate_parallel_mp,
                                       evaluate_against_random_parallel_mp)
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
    # Skip training until the buffer holds at least this many samples (0 = off).
    # Guards against overfitting train_steps to a tiny early, mostly-draw buffer.
    warmup_min_samples: int = 0
    mcts_simulations: int = 100
    # 0 => use mcts_simulations; else lower for faster eval
    eval_simulations: int = 0
    replay_buffer_size: int = 50_000
    max_game_moves: int = 300
    eval_games: int = 80
    eval_random_games: int = 24
    accept_margin: float = 0.05          # accept if win_rate > fair_share + margin
    # Opening moves sampled from the visit distribution during eval. 0 makes
    # every eval game with the same seat assignment a replay of the same game,
    # which is how a 40-game gate came to measure 2 distinct games at N=2.
    eval_opening_plies: int = DEFAULT_EVAL_OPENING_PLIES
    # run eval every N iterations (1 = every iter)
    eval_every: int = 1
    discount: float = 0.97
    explore_moves: int = 15
    mcts_dirichlet_epsilon: float = 0.25
    # --- self-play engine selector ---
    # "auto" (default) => derive from parallel_self_play (back-compat: parallel if
    # True else sequential). Explicit values override:
    #   "sequential" — one game at a time, no batching.
    #   "parallel"   — K worker processes + shared GPU batcher (leaf-parallel).
    #   "vectorized" — in-process, G games share one predict_batch (Option B;
    #                  exact sequential MCTS per game, no straggler tail).
    self_play_mode: str = "auto"
    # Games run concurrently by the vectorized engine (GPU batch width).
    # 0 => driver default (min(games_per_iteration, 64)).
    vec_games: int = 0
    # --- parallel self-play ---
    parallel_self_play: bool = False
    num_workers: int = 8
    inference_batch_size: int = 64
    # Leaf-parallel MCTS in the spawned self-play/eval workers: leaves collected per
    # GPU forward (>1 breaks the batch<=num_workers ceiling) + virtual loss to
    # diversify the concurrent tree walks. leaf_batch=1 keeps the one-leaf path.
    leaf_batch: int = 1
    virtual_loss: float = 1.0
    # GPU-batched parallel evaluation (candidate/champion served by one batcher).
    # Reuses num_workers / inference_batch_size. Opt-in; sequential eval otherwise.
    parallel_eval: bool = False
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


def _mcts(model, env, cfg, sims=None, dirichlet_epsilon=None):
    # dirichlet_epsilon override: pass 0.0 for eval (deterministic best-play, no
    # exploration noise); leave None for self-play to use the cfg default.
    eps = (dirichlet_epsilon if dirichlet_epsilon is not None
           else getattr(cfg, 'mcts_dirichlet_epsilon', 0.25))
    return MCTSMaxN(
        config=MCTSConfig(num_simulations=sims or cfg.mcts_simulations,
                          dirichlet_epsilon=eps,
                          max_rollout_depth=cfg.max_game_moves),
        evaluate_fn=lambda st: model.predict(env.state_to_tensor(st)),
        num_players=cfg.num_players,
    )


SELF_PLAY_MODES = ("sequential", "parallel", "vectorized")


def resolve_self_play_mode(cfg):
    """Resolve cfg.self_play_mode to a concrete engine name, raising on typos
    rather than silently falling through to the sequential path."""
    mode = getattr(cfg, "self_play_mode", "auto")
    if mode == "auto":
        return "parallel" if cfg.parallel_self_play else "sequential"
    if mode not in SELF_PLAY_MODES:
        raise ValueError(
            f"self_play_mode={mode!r} is not recognised — expected 'auto' or one "
            f"of {SELF_PLAY_MODES}.")
    return mode


def init_champion(best, model, checkpoint_dir, log=print):
    """Establish the gating champion at run start and make it durable on disk.

    `best.pt` used to be written only on acceptance. A run that never accepted
    therefore never created it, and `load_champion` below would then silently
    re-anchor the champion to the current learner on every restart — candidate vs
    an identical copy of itself, which splits seats exactly and scores 1/N, below
    any `fair + margin` threshold. That is a closed loop: never accept -> never
    write best.pt -> next restart resets the champion -> never accept. Writing it
    up front breaks the loop at its only entry point.
    """
    best.copy_weights_from(model)
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(best_path):
        best.save(best_path)
        log(f"Champion initialised from the starting model -> {best_path}")
    return best_path


def load_champion(best, model, checkpoint_dir, log=print):
    """Restore the gating champion on resume. Returns True if loaded from disk.

    A missing `best.pt` is not a normal state once `init_champion` has run, so it
    is reported loudly rather than papered over: falling back to the current
    learner makes the gate compare the candidate against itself, which looks like
    a working eval while measuring nothing.
    """
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
        best.load(best_path)
        return True
    best.copy_weights_from(model)
    best.save(best_path)
    log(f"WARNING: {best_path} is missing on resume — the gating champion has "
        f"been re-seeded from the current model and saved. Until a candidate is "
        f"accepted, eval-vs-best compares the model against an identical copy of "
        f"itself and cannot exceed the accept threshold.")
    return False


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
    fair = 1.0 / cfg.num_players
    threshold = fair + cfg.accept_margin
    history = []

    # Disk log: keeps recording progress even if the Jupyter UI disconnects.
    _log = make_progress_logger(os.path.join(checkpoint_dir, "games.log"))

    # Resolved once, up front: validates the config before any work is done, and
    # the banner then reports exactly what the loop will run.
    sp_mode = resolve_self_play_mode(cfg)
    _log(
        "=" * 70,
        f"training_loop_mp launched | N={cfg.num_players} board={cfg.board_size}x{cfg.board_size} "
        f"| sims={cfg.mcts_simulations} games/iter={cfg.games_per_iteration} "
        f"| self_play={sp_mode} workers={cfg.num_workers} vec_games={cfg.vec_games}",
        f"checkpoint_dir={checkpoint_dir} | eval={cfg.eval_games}+{cfg.eval_random_games} "
        f"| accept_margin={cfg.accept_margin} | buffer={cfg.replay_buffer_size}",
        f"train_steps={cfg.train_steps_per_iter} max_moves={cfg.max_game_moves} "
        f"explore_moves={cfg.explore_moves} warmup={cfg.warmup_min_samples} "
        f"leaf_batch={cfg.leaf_batch} vloss={cfg.virtual_loss}",
        "=" * 70,
    )

    # --- Resume from checkpoint if available ---
    start_iter = 0
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(meta_path) and os.path.exists(latest_path):
        with open(meta_path) as f:
            meta = json.load(f)
        start_iter = meta.get("completed_iterations", 0)
        model.load(latest_path)
        load_champion(best, model, checkpoint_dir, log=_log)
        history = meta.get("history", [])
        _log(
            f"Resumed from iteration {start_iter} (checkpoint: {checkpoint_dir})")
    else:
        # Durable champion from iteration 0, so the gate has a real opponent even
        # if nothing is ever accepted.
        init_champion(best, model, checkpoint_dir, log=_log)
        _log(f"Starting N={cfg.num_players} training: {cfg.num_iterations} iterations, "
             f"{cfg.games_per_iteration} games/iter, {cfg.mcts_simulations} sims")

    for it in range(start_iter, cfg.num_iterations):
        t0 = time.time()
        # --- 1. self-play ---
        _log(f"[iter {it+1}/{cfg.num_iterations}] self-play starting "
             f"({cfg.games_per_iteration} games, {cfg.mcts_simulations} sims)...")

        def _on_progress(done, total, w):
            if done % 5 == 0 or done == total:
                _log(
                    f"[iter {it+1}/{cfg.num_iterations}] self-play: {done}/{total} games...")

        if sp_mode == "vectorized":
            # In-process vectorized self-play (Option B): G games share one
            # predict_batch; exact sequential MCTS per game.
            sp_samples, wins = generate_vectorized_self_play_mp(
                model, cfg,
                total_games=cfg.games_per_iteration,
                vec_games=(cfg.vec_games or None),
                batch_size=cfg.inference_batch_size,
                on_games_complete=_on_progress,
                base_seed=it * cfg.games_per_iteration,
                log=_log,
            )
            if not sp_samples:
                raise RuntimeError(
                    f"[iter {it+1}] vectorized self-play produced 0 samples — "
                    f"aborting before empty training.")
            buffer.add(sp_samples)
            n_new_samples = len(sp_samples)
        elif sp_mode == "parallel":
            # GPU-batched parallel self-play
            sp_samples, wins = generate_parallel_self_play_mp(
                model, cfg,
                num_workers=cfg.num_workers,
                total_games=cfg.games_per_iteration,
                batch_size=cfg.inference_batch_size,
                on_games_complete=_on_progress,
                # Seeds are per-game (base_seed + game_index, index in
                # [0, games_per_iteration)); stride by games_per_iteration so
                # iterations never reuse each other's seeds.
                base_seed=it * cfg.games_per_iteration,
                log=_log,
            )
            if not sp_samples:
                # A zero-sample iteration means the parallel self-play stalled
                # (almost always: the GPU inference thread died and workers hung
                # until the queue timeout). Abort loudly instead of "training" on
                # an empty buffer and silently advancing completed_iterations —
                # the run can then be resumed from the last good checkpoint.
                raise RuntimeError(
                    f"[iter {it+1}] parallel self-play produced 0 samples — aborting "
                    f"before empty training. Check {os.path.join(checkpoint_dir, 'games.log')} "
                    f"for '[GPU INFERENCE] CRASHED' or '[WORKER … ] CRASHED'.")
            buffer.add(sp_samples)
            n_new_samples = len(sp_samples)
        else:
            # Sequential self-play (original path)
            sp_mcts = _mcts(model, env, cfg)
            wins = {}
            n_new_samples = 0
            for g in range(cfg.games_per_iteration):
                samples, w = play_one_game(env, sp_mcts, cfg.num_players,
                                           max_moves=cfg.max_game_moves,
                                           discount=cfg.discount,
                                           explore_moves=cfg.explore_moves)
                buffer.add(samples)
                n_new_samples += len(samples)
                wins[w] = wins.get(w, 0) + 1
                # Log every 5 games so long iterations don't look stuck.
                if (g + 1) % 5 == 0 or (g + 1) == cfg.games_per_iteration:
                    elapsed = time.time() - t0
                    rate = (g + 1) / elapsed if elapsed > 0 else 0
                    _log(f"[iter {it+1}/{cfg.num_iterations}] self-play: "
                         f"{g+1}/{cfg.games_per_iteration} games "
                         f"({elapsed:.0f}s, {rate*60:.1f} games/min)")

        sp_secs = time.time() - t0
        # Win distribution across seats (None = draw/timeout). A healthy self-play
        # iteration is roughly balanced; a lopsided split or all-draws is an early
        # warning of seat bias or a degenerate policy.
        win_dist = ", ".join(
            f"{'draw' if w is None else f'P{w}'}={wins[w]}"
            for w in sorted(wins, key=lambda k: (k is None, k)))
        # Quoridor has no true draws — a None winner is a timeout at max_game_moves,
        # which yields an all-zero value target (weak learning signal). Track the rate
        # as a watchdog. avg_len divides by 2 because augment_mp doubles each game's
        # samples (original + mirror).
        draws = wins.get(None, 0)
        draw_rate = draws / cfg.games_per_iteration if cfg.games_per_iteration else 0.0
        avg_len = (n_new_samples / 2.0 / cfg.games_per_iteration
                   if cfg.games_per_iteration else 0.0)
        _log(f"[iter {it+1}/{cfg.num_iterations}] self-play done: "
             f"{cfg.games_per_iteration} games ({sp_secs:.0f}s) | wins: {win_dist} "
             f"| draw_rate={100*draw_rate:.0f}% avg_len~{avg_len:.0f}")
        if draw_rate > 0.20:
            _log(f"[iter {it+1}/{cfg.num_iterations}] WARNING: draw_rate "
                 f"{100*draw_rate:.0f}% > 20% — games timing out at "
                 f"max_game_moves={cfg.max_game_moves} (weak value signal).")

        # --- 2. train ---
        t_train = time.time()
        _log(
            f"[iter {it+1}/{cfg.num_iterations}] training ({cfg.train_steps_per_iter} steps)...")
        lp = lv = 0.0
        warmup = max(cfg.batch_size, cfg.warmup_min_samples)
        if len(buffer) >= warmup:
            steps = cfg.train_steps_per_iter
        else:
            steps = 0
            _log(f"[iter {it+1}/{cfg.num_iterations}] training skipped — "
                 f"buffer {len(buffer)} < warmup {warmup} (filling)")
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

        # --- 2b. Checkpoint immediately after training, BEFORE eval. ---
        # Eval is a long (~hours), sequential, best-effort phase. If the process
        # dies during it we must NOT lose the self-play + training work, so we
        # persist the trained weights and advance completed_iterations here. The
        # model is not modified during eval, so latest.pt saved now == saved after
        # eval. Eval only affects best.pt (the gating champion) and this row's eval
        # columns, both updated in place below as eval progresses. An eval-phase
        # interruption therefore costs only that iteration's eval — the ~1h of
        # self-play + training is already durable and resume skips to the next iter.
        run_eval = (it + 1) % cfg.eval_every == 0 or (it +
                                                      1) == cfg.num_iterations
        eval_sims = cfg.eval_simulations or cfg.mcts_simulations
        accepted = False
        eval_best_secs = 0.0
        eval_rand_secs = 0.0
        # None, not 0.0: on a skipped-eval iteration nothing was measured. Zeros
        # here are indistinguishable from a real 0% and were persisted into
        # meta.json for 44 of the 51 rows across the two 9x9 runs, then plotted
        # as if they were results.
        ev_wr = None
        evr_wr = None

        model.save(os.path.join(checkpoint_dir, "latest.pt"))
        row = dict(iter=it + 1, loss_p=lp, loss_v=lv,
                   win_vs_best=ev_wr, accepted=accepted,
                   win_vs_random=evr_wr, fair=fair, draw_rate=draw_rate,
                   secs=time.time() - t0, buffer=len(buffer),
                   sp_secs=sp_secs, train_secs=train_secs,
                   eval_best_secs=eval_best_secs, eval_rand_secs=eval_rand_secs,
                   eval_ran=run_eval)
        history.append(row)

        def _write_meta():
            with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
                json.dump({"completed_iterations": it +
                          1, "history": history}, f)

        _write_meta()  # durable resume point: self-play + training now survive a crash

        # --- 3. accept/reject vs best (candidate rotates seats) — best-effort ---
        # Eval uses fewer sims than self-play when eval_simulations is set:
        # eval only measures relative strength, so it doesn't need full search
        # depth. This does NOT weaken the trained model (self-play keeps full sims).
        # eval_every: skip eval on most iterations to save time (eval is expensive).
        if run_eval:
            # Geometry + sims for the parallel eval workers (spawned processes).
            eval_config_dict = {
                "num_players": cfg.num_players,
                "board_size": getattr(cfg, "board_size", None) or model.board_size,
                "max_walls_per_player": getattr(cfg, "max_walls_per_player", 3),
                "max_turns": getattr(cfg, "max_turns", cfg.max_game_moves),
                "eval_simulations": eval_sims,
                "max_game_moves": cfg.max_game_moves,
                "leaf_batch": cfg.leaf_batch,
                "virtual_loss": cfg.virtual_loss,
                "eval_opening_plies": cfg.eval_opening_plies,
            }
            t_eval_best = time.time()
            _log(
                f"[iter {it+1}/{cfg.num_iterations}] eval vs best ({cfg.eval_games} games, {eval_sims} sims)...")

            def _eval_progress(done, total, r):
                elapsed = time.time() - t_eval_best
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs best: "
                     f"{done}/{total} games ({elapsed:.0f}s, cand {r.candidate_win_rate:.0%})")

            if cfg.parallel_eval:
                ev = evaluate_parallel_mp(
                    model, best, eval_config_dict, num_games=cfg.eval_games,
                    num_workers=cfg.num_workers, batch_size=cfg.inference_batch_size,
                    on_progress=_eval_progress, base_seed=it * 100_003, log=_log)
            else:
                # dirichlet_epsilon=0 → best-play eval (matches the parallel path,
                # which also disables exploration noise). Game diversity comes from
                # the sampled opening, not from search noise.
                cand = mcts_agent_mp(
                    _mcts(model, env, cfg, sims=eval_sims, dirichlet_epsilon=0.0),
                    temperature=0.1, opening_plies=cfg.eval_opening_plies)
                champ = mcts_agent_mp(
                    _mcts(best, env, cfg, sims=eval_sims, dirichlet_epsilon=0.0),
                    temperature=0.1, opening_plies=cfg.eval_opening_plies)
                ev = evaluate_mp(env, cand, champ, num_games=cfg.eval_games,
                                 max_moves=cfg.max_game_moves, on_progress=_eval_progress,
                                 base_seed=it * 100_003)
            accepted = ev.should_accept(threshold)
            eval_best_secs = time.time() - t_eval_best
            ev_wr = ev.candidate_win_rate
            if accepted:
                best.copy_weights_from(model)
                best.save(os.path.join(checkpoint_dir, "best.pt"))
            # Distinguish "lost the gate" from "the gate had nothing to judge":
            # a mostly-timed-out eval rejects on insufficient evidence, which is a
            # game-length problem, not a strength problem.
            if not accepted and ev.decided_games < ev.num_games * ev.MIN_DECIDED_FRACTION:
                verdict = (f"reject (only {ev.decided_games}/{ev.num_games} games "
                           f"decided — too few to gate on)")
            else:
                verdict = "ACCEPT" if accepted else "reject"
            _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs best done: "
                 f"{100*ev_wr:.1f}% of {ev.decided_games} decided {verdict} "
                 f"({eval_best_secs:.0f}s)")
            # Persist the accept/reject before eval-vs-random, so best.pt and the
            # row's `accepted` stay consistent even if the next phase is interrupted.
            row.update(win_vs_best=ev_wr, accepted=accepted,
                       eval_best_secs=eval_best_secs, secs=time.time() - t0)
            _write_meta()

            # --- 4. eval vs random ---
            t_eval_rand = time.time()
            _log(
                f"[iter {it+1}/{cfg.num_iterations}] eval vs random ({cfg.eval_random_games} games, {eval_sims} sims)...")

            def _eval_rand_progress(done, total, r):
                elapsed = time.time() - t_eval_rand
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs random: "
                     f"{done}/{total} games ({elapsed:.0f}s, cand {r.candidate_win_rate:.0%})")

            if cfg.parallel_eval:
                evr = evaluate_against_random_parallel_mp(
                    model, eval_config_dict, num_games=cfg.eval_random_games,
                    num_workers=cfg.num_workers, batch_size=cfg.inference_batch_size,
                    on_progress=_eval_rand_progress, base_seed=it * 100_003 + 50_000,
                    log=_log)
            else:
                evr = evaluate_against_random_mp(env, cand, num_games=cfg.eval_random_games,
                                                 max_moves=cfg.max_game_moves,
                                                 on_progress=_eval_rand_progress,
                                                 base_seed=it * 100_003 + 50_000)
            eval_rand_secs = time.time() - t_eval_rand
            evr_wr = evr.candidate_win_rate
            _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs random done: "
                 f"{100*evr_wr:.1f}% ({eval_rand_secs:.0f}s)")
            row.update(win_vs_random=evr_wr, eval_rand_secs=eval_rand_secs,
                       secs=time.time() - t0, eval_ran=True)
            _write_meta()
        else:
            _log(
                f"[iter {it+1}/{cfg.num_iterations}] eval skipped (eval_every={cfg.eval_every})")

        # "n/a" rather than 0.0%: the summary line is the thing most often read
        # at a glance, and a skipped eval printing "vs_best=0.0% reject" is what
        # made both 9x9 runs look like total failures for 44 of 51 iterations.
        vs_best_txt = (f"{100*ev_wr:.1f}% {'ACCEPT' if accepted else 'reject'}"
                       if ev_wr is not None else "n/a (not evaluated)")
        vs_rand_txt = f"{100*evr_wr:.1f}%" if evr_wr is not None else "n/a"
        _log(f">>> iter {it+1} | loss_p={lp:.3f} loss_v={lv:.3f} | "
             f"vs_best={vs_best_txt} | "
             f"vs_rand={vs_rand_txt} | draw={100*draw_rate:.0f}% | buf={len(buffer)} | "
             f"sp={sp_secs:.0f}s train={train_secs:.0f}s "
             f"eval_best={eval_best_secs:.0f}s eval_rand={eval_rand_secs:.0f}s | "
             f"total={row['secs']:.0f}s")
    return history
