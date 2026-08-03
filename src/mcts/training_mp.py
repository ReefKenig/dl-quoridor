"""
N-player AlphaZero training loop (vector value + maxⁿ).

Self-contained: self-play with the current model's maxⁿ → train on vector targets
→ accept/reject vs the best model (candidate rotated through all seats) → eval vs
random → checkpoint best/latest. Reduces to the standard duel at N=2.
"""
import json
import math
import os
import time
import logging
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from src.env.pathing import CURRENT_SPEC
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.mcts.self_play_mp import play_one_game
from src.utils.schedule import lr_at
from src.mcts.batched_inference_mp import DEFAULT_BATCH_WAIT_MS
from src.mcts.parallel_self_play_mp import generate_parallel_self_play_mp
from src.mcts.vectorized_self_play_mp import generate_vectorized_self_play_mp
from src.mcts.evaluator_mp import (DEFAULT_EVAL_OPENING_PLIES, evaluate_mp,
                                   evaluate_against_random_mp, greedy_agent,
                                   mcts_agent_mp)
from src.mcts.parallel_eval_mp import (evaluate_parallel_mp,
                                       evaluate_against_greedy_parallel_mp,
                                       evaluate_against_random_parallel_mp)
from src.utils.config import read_frozen_config
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
    # Absolute yardstick that does not saturate the way vs-random does (9x9 N=2
    # hit 100% vs random at iter 5 and stayed there). 0 disables it.
    eval_greedy_games: int = 0
    accept_margin: float = 0.05          # accept if win_rate > fair_share + margin
    # "constant" keeps existing scripts unchanged; 9x9 opts into cosine.
    lr_schedule: str = "constant"
    lr_final_frac: float = 0.1           # cosine end point, as a fraction of base_lr
    # Batcher accumulation window; 0 restores the old no-wait draining.
    batch_wait_ms: float = DEFAULT_BATCH_WAIT_MS
    # Sampled opening plies in eval; 0 makes same-seat games identical replays.
    eval_opening_plies: int = DEFAULT_EVAL_OPENING_PLIES
    # run eval every N iterations (1 = every iter)
    eval_every: int = 1
    discount: float = 0.97
    # "round" = decay per the mover's own turns; "ply" = per move by
    # anybody. Per-variant: measured better at N=2, worse at N=4.
    discount_unit: str = "round"
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
    # Wall curriculum for SELF-PLAY only; eval always plays the full game so the
    # gate and the greedy baseline stay comparable across the whole run.
    # wall_mask_iters opening iterations at 0 walls (a pure race), then
    # one more wall every wall_ramp_hold iterations. 0/0 = off.
    # Handing back all the walls at once relapsed the policy in one iteration at
    # both player counts; see academic_experiences.md 8.6.
    wall_mask_iters: int = 0
    wall_ramp_hold: int = 0
    # Set per iteration onto cfg.wall_budget, which the workers read.
    wall_budget: int = None
    # Input-plane spec the run trains under; frozen into the run dir's config.json
    # so the resulting checkpoint can be replayed on the planes it actually saw.
    spec_version: int = CURRENT_SPEC
    # Early stop on racing decay. Greedy always has one seat a pure racer wins
    # outright (seat 1 at N=2, seat 0 at N=4 — whoever the head-on pawn jump
    # favours), so the BEST per-seat greedy rate is a hard-ceiling probe: it
    # cannot drift up, and a sustained fall means the policy stopped racing.
    # The gate cannot see this — in local_9x9_v6 it kept accepting (69%, 60%)
    # while greedy went 60% -> 10%. Stop after this many consecutive evals at
    # least greedy_stop_drop below the best rate seen so far. 0 = disabled.
    greedy_stop_patience: int = 0
    greedy_stop_drop: float = 0.20
    # Sigma the drop must also clear, so noise on a 20-game seat cannot strike.
    greedy_stop_z: float = 2.0
    # Companion watch: stop if the best per-seat rate never REACHES this once the
    # masked curriculum is over. Decay cannot catch a run that never climbed.
    # 0 = off; shares greedy_stop_patience.
    greedy_min_seat: float = 0.0


BUFFER_FILE = "replay_buffer.npz"


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

    def save(self, checkpoint_dir):
        """Persist the buffer so a resume keeps its samples. tmp+rename so a
        kill mid-write leaves the previous buffer rather than a truncated one."""
        path = os.path.join(checkpoint_dir, BUFFER_FILE)
        tmp = path + ".tmp"
        S = np.array([x[0] for x in self.buffer], np.float32)
        P = np.array([x[1] for x in self.buffer], np.float32)
        V = np.array([x[2] for x in self.buffer], np.float32)
        with open(tmp, "wb") as f:
            np.savez(f, S=S, P=P, V=V)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    def load(self, checkpoint_dir, log=print):
        """Samples restored from disk. Missing or unreadable leaves it empty,
        which is the old resume behaviour rather than a failed run."""
        path = os.path.join(checkpoint_dir, BUFFER_FILE)
        if not os.path.exists(path):
            return 0
        try:
            with np.load(path) as data:
                S, P, V = data["S"], data["P"], data["V"]
        except (OSError, ValueError, KeyError) as exc:
            log(f"WARNING: could not read {path} ({exc}) — starting with an "
                f"empty buffer, as if it had not been persisted.")
            return 0
        # .copy() because iterating a stacked array yields views: one surviving
        # view keeps the entire ~500 MB base alive. maxlen trims the oldest if
        # the run resumes at a smaller buffer size.
        self.add((s.copy(), p.copy(), v.copy()) for s, p, v in zip(S, P, V))
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


def drop_is_significant(peak, now, n_per_seat, z_min):
    """Two-proportion z: is the fall bigger than n-game sampling noise?

    The peak is a max over seats AND over evals, so it overshoots the true rate;
    without this the threshold lands near the mean and a stable model strikes.
    """
    if not n_per_seat:
        return True
    pooled = (peak + now) / 2
    se = math.sqrt(2 * pooled * (1 - pooled) / n_per_seat)
    return se == 0 or (peak - now) >= z_min * se


def stalled_below_floor(best_seat, floor, below):
    """Advance the never-acquired watch. Returns consecutive evals under floor.

    racing_decay_strike can only fire on a fall from a peak, so a run that never
    climbs is invisible to it: probe_n4_ramp peaked at 0.10 against a 0.20 drop,
    making a strike arithmetically impossible, and it ran 10 hours past the point
    the answer was known.
    """
    return below + 1 if best_seat < floor else 0


def racing_decay_strike(best_seat, peak, below, drop, n_per_seat=0, z_min=0.0):
    """Advance the racing-decay watch by one greedy eval.

    Compares against the best rate ever seen, not the previous eval, so a slow
    slide cannot hide. A strike needs the drop to clear `drop` AND, when
    n_per_seat/z_min are given, to be significant at that many sigma.
    Returns (peak, consecutive_strikes).
    """
    if peak is None or best_seat > peak:
        return best_seat, 0
    if best_seat > peak - drop:
        return peak, 0
    if not drop_is_significant(peak, best_seat, n_per_seat, z_min):
        return peak, 0
    return peak, below + 1


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


def zero_sample_reason(it, engine, wins, games_per_iteration, checkpoint_dir):
    """Diagnose a zero-sample iteration: all-timeout vs a stalled batcher."""
    if wins.get(None, 0) >= games_per_iteration:
        return (
            f"[iter {it}] all {games_per_iteration} games timed out at "
            f"max_game_moves, so self-play produced no training samples. This is "
            f"a game-length problem, not a crash: raise max_game_moves for this "
            f"variant. It counts plies rather than rounds, so N=4 needs roughly "
            f"twice the N=2 value to give each player the same budget.")
    return (
        f"[iter {it}] {engine} self-play produced 0 samples — aborting before "
        f"empty training. Check {os.path.join(checkpoint_dir, 'games.log')} for "
        f"'[GPU INFERENCE] CRASHED' or '[WORKER … ] CRASHED'.")


def _rss_gb():
    """Resident memory of this process, in GB (0.0 if psutil is unavailable)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return 0.0


def _host_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def _read_first(*paths):
    for path in paths:
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def _cgroup_mem_limit_gb():
    """Container memory ceiling in GB, or None when unlimited. psutil reports the
    host's RAM, which on a shared or MIG-partitioned box is not what we may use."""
    raw = _read_first("/sys/fs/cgroup/memory.max",                  # cgroup v2
                      "/sys/fs/cgroup/memory/memory.limit_in_bytes")  # cgroup v1
    if raw is None or raw == "max":
        return None
    try:
        limit = int(raw)
    except ValueError:
        return None
    # v1 reports a sentinel near 2**63 when unlimited.
    return None if limit >= 2**62 else limit / 1e9


def _cpu_budget():
    """(usable_cpus, source): cgroup quota -> affinity -> host cores."""
    raw = _read_first("/sys/fs/cgroup/cpu.max")                     # cgroup v2
    if raw and not raw.startswith("max"):
        try:
            quota, period = raw.split()
            return int(quota) / int(period), "cgroup quota"
        except (ValueError, ZeroDivisionError):
            pass
    quota = _read_first("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")      # cgroup v1
    period = _read_first("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period), "cgroup quota"
    try:
        return float(len(os.sched_getaffinity(0))), "affinity"
    except AttributeError:
        return float(os.cpu_count() or 0), "host cores"


def _gpu_desc():
    """Device name and memory as this process sees it — a MIG slice, not the card."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "no CUDA"
        props = torch.cuda.get_device_properties(0)
        return f"{props.name}, {props.total_memory / 1e9:.0f} GB"
    except Exception:
        return "unknown"


def resource_banner(cfg):
    """One line describing what this process may actually use, plus a warning
    when num_workers oversubscribes the CPU budget."""
    cpus, source = _cpu_budget()
    mem_limit = _cgroup_mem_limit_gb()
    mem = (f"{mem_limit:.0f} GB limit (host {_host_ram_gb():.0f} GB)"
           if mem_limit else f"{_host_ram_gb():.0f} GB, no cgroup limit")
    lines = [f"resources: {cpus:.0f} usable cpus ({source}), {mem}, "
             f"gpu: {_gpu_desc()}, rss={_rss_gb():.2f} GB at launch"]
    if cpus and cfg.num_workers > cpus:
        lines.append(
            f"WARNING: num_workers={cfg.num_workers} exceeds the {cpus:.0f} usable "
            f"cpus — expect contention, and slowdowns that end in a killed kernel.")
    return lines


def wall_budget_at(it, mask_iters, ramp_hold, max_walls):
    """Walls each player starts self-play with at iteration `it` (0-based).

    0 while masked, then one more wall every `ramp_hold` iterations. A step
    lasted a single iteration before, which was too fast: probe_n2_ramp held
    the race floor at 1 wall and collapsed at 2. 0 = full allowance at once.
    """
    if it < mask_iters:
        return 0
    if ramp_hold <= 0:
        return max_walls
    return min(max_walls, 1 + (it - mask_iters) // ramp_hold)


@contextmanager
def _wall_budget(cfg, env, budget):
    """Apply the curriculum's wall budget to self-play only.

    Restores on the way out even if self-play raises — eval, gating and the
    greedy baseline must always play the full game, and the caller may keep
    using this env after a failed iteration.
    """
    cfg.wall_budget = env.wall_budget = budget
    try:
        yield
    finally:
        cfg.wall_budget = env.wall_budget = None


def assert_resume_spec_matches(cfg, checkpoint_dir):
    """Refuse to resume a run onto input planes it was not trained on. Run dirs
    predating the versioning have no spec_version key — those are all v1."""
    frozen = read_frozen_config(checkpoint_dir)
    if frozen is None:
        return
    frozen_spec = frozen["spec_version"]
    if frozen_spec != cfg.spec_version:
        raise ValueError(
            f"{checkpoint_dir} was trained under tensor spec v{frozen_spec} but "
            f"this launch is configured for v{cfg.spec_version}. Resuming would "
            f"feed the checkpoint planes on a scale it never saw. Start a fresh "
            f"run dir, or set spec_version={frozen_spec} to continue the old one.")


def freeze_config(cfg, checkpoint_dir, log=print):
    """Write the resolved config next to the checkpoints, so a run dir records
    what it actually ran rather than relying on the shared config file.

    First launch writes it; later launches only compare, so the record stays
    the run's rather than the last relaunch's."""
    path = os.path.join(checkpoint_dir, "config.json")
    resolved = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    frozen = read_frozen_config(checkpoint_dir)
    if frozen is not None:
        changed = {k: (frozen[k], v) for k, v in resolved.items()
                   if k in frozen and frozen[k] != v}
        added = sorted(set(resolved) - set(frozen))
        if changed or added:
            log(f"WARNING: config differs from {path}, which records what the "
                f"earlier iterations ran. Keeping the frozen copy; this launch "
                f"uses:")
            for k, (was, now) in sorted(changed.items()):
                log(f"  {k}: frozen={was!r} -> now={now!r}")
            if added:
                log(f"  new keys: {', '.join(added)}")
        else:
            log(f"Config matches the frozen copy -> {path}")
        return path
    with open(path, "w") as f:
        json.dump(resolved, f, indent=2, default=str, sort_keys=True)
    log(f"Config frozen -> {path}")
    return path


def init_champion(best, model, checkpoint_dir, log=print):
    """Establish the gating champion and make it durable from iteration 0.

    Written up front so a run that never accepts still has a real opponent.
    """
    best.copy_weights_from(model)
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(best_path):
        best.save(best_path)
        log(f"Champion initialised from the starting model -> {best_path}")
    return best_path


def load_champion(best, model, checkpoint_dir, log=print):
    """Restore the gating champion on resume. Returns True if loaded from disk.

    A missing best.pt warns loudly: falling back to the learner makes the gate
    compare the model against itself and measure nothing.
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
    # The workers rebuild the env from cfg, so a disagreement here would have the
    # parent and the workers training one model on two different plane scales.
    if env.spec_version != cfg.spec_version:
        raise ValueError(
            f"env tensor spec v{env.spec_version} != cfg spec v{cfg.spec_version}; "
            "resuming a run started under an older spec needs the env built with "
            "that run's config.json spec_version")
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
        f"checkpoint_dir={checkpoint_dir} | eval={cfg.eval_games}+{cfg.eval_random_games}"
        f"{'+' + str(cfg.eval_greedy_games) + ' greedy' if cfg.eval_greedy_games else ' (no greedy)'} "
        f"| accept_margin={cfg.accept_margin} | buffer={cfg.replay_buffer_size}",
        f"train_steps={cfg.train_steps_per_iter} max_moves={cfg.max_game_moves} "
        f"explore_moves={cfg.explore_moves} warmup={cfg.warmup_min_samples} "
        f"discount={cfg.discount}/{cfg.discount_unit} "
        f"leaf_batch={cfg.leaf_batch} vloss={cfg.virtual_loss}",
        *resource_banner(cfg),
        "=" * 70,
    )
    freeze_config(cfg, checkpoint_dir, log=_log)

    # --- Resume from checkpoint if available ---
    start_iter = 0
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(meta_path) and os.path.exists(latest_path):
        assert_resume_spec_matches(cfg, checkpoint_dir)
        with open(meta_path) as f:
            meta = json.load(f)
        start_iter = meta.get("completed_iterations", 0)
        model.load(latest_path)
        load_champion(best, model, checkpoint_dir, log=_log)
        history = meta.get("history", [])
        n_buffered = buffer.load(checkpoint_dir, log=_log)
        _log(
            f"Resumed from iteration {start_iter} (checkpoint: {checkpoint_dir})",
            f"Replay buffer restored: {n_buffered} samples" if n_buffered else
            "Replay buffer not on disk — refilling from scratch")
    else:
        init_champion(best, model, checkpoint_dir, log=_log)
        _log(f"Starting N={cfg.num_players} training: {cfg.num_iterations} iterations, "
             f"{cfg.games_per_iteration} games/iter, {cfg.mcts_simulations} sims")

    # Racing-decay watch (see cfg.greedy_stop_patience).
    greedy_peak = None
    greedy_below_peak = 0
    greedy_below_floor = 0
    stop_reason = None

    for it in range(start_iter, cfg.num_iterations):
        t0 = time.time()
        # --- 1. self-play ---
        budget = wall_budget_at(it, cfg.wall_mask_iters, cfg.wall_ramp_hold,
                                cfg.max_walls_per_player)
        with _wall_budget(cfg, env, budget):
            curriculum = ("" if budget == cfg.max_walls_per_player else
                          f", WALLS MASKED" if budget == 0 else
                          f", {budget}/{cfg.max_walls_per_player} WALLS")
            _log(f"[iter {it+1}/{cfg.num_iterations}] self-play starting "
                 f"({cfg.games_per_iteration} games, {cfg.mcts_simulations} sims"
                 f"{curriculum})...")

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
                    raise RuntimeError(zero_sample_reason(
                        it + 1, "vectorized", wins, cfg.games_per_iteration,
                        checkpoint_dir))
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
                    # Either self-play stalled (usually: the GPU inference thread died
                    # and workers hung until the queue timeout) or every game timed
                    # out. Abort loudly instead of "training" on an empty buffer and
                    # silently advancing completed_iterations — the run can then be
                    # resumed from the last good checkpoint.
                    raise RuntimeError(zero_sample_reason(
                        it + 1, "parallel", wins, cfg.games_per_iteration,
                        checkpoint_dir))
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
        # Pure function of the iteration - nothing to persist across resume.
        cur_lr = lr_at(cfg.lr_schedule, model.base_lr, it, cfg.num_iterations,
                       cfg.lr_final_frac)
        model.set_lr(cur_lr)
        _log(
            f"[iter {it+1}/{cfg.num_iterations}] training ({cfg.train_steps_per_iter} steps, "
            f"lr={cur_lr:.2e})...")
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
        # None, not 0.0 - a skipped eval measured nothing, and zeros here are
        # indistinguishable from a real 0%.
        ev_wr = None
        evr_wr = None

        model.save(os.path.join(checkpoint_dir, "latest.pt"))
        # Before _write_meta, so meta.json is never newer than the buffer.
        t_buf = time.time()
        buffer.save(checkpoint_dir)
        buf_secs = time.time() - t_buf
        row = dict(iter=it + 1, loss_p=lp, loss_v=lv,
                   win_vs_best=ev_wr, accepted=accepted,
                   win_vs_random=evr_wr, fair=fair, draw_rate=draw_rate,
                   # Racing evidence: under the wall mask this should fall
                   # toward a pure race. Was only ever in games.log.
                   avg_len=avg_len,
                   # First iteration after a relaunch, so a reader of the loss
                   # curve can tell a resume dip from learning dynamics.
                   resumed=(it == start_iter and start_iter > 0),
                   # Walls self-play ran with, so the curriculum is legible in
                   # the record rather than only in games.log.
                   wall_budget=budget,
                   # Denominators, so a rate in meta.json is readable on its own.
                   decided_games=None, eval_timeouts=None,
                   rand_decided_games=None, greedy_decided_games=None,
                   win_vs_greedy=None, greedy_by_seat=None,
                   secs=time.time() - t0, buffer=len(buffer),
                   sp_secs=sp_secs, train_secs=train_secs, buf_secs=buf_secs,
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
                # Without this a resumed v1 run would self-play on v1 planes and
                # gate on v2 ones, comparing both models on a scale neither saw.
                "spec_version": getattr(cfg, "spec_version", CURRENT_SPEC),
                "eval_simulations": eval_sims,
                "max_game_moves": cfg.max_game_moves,
                "leaf_batch": cfg.leaf_batch,
                "virtual_loss": cfg.virtual_loss,
                "eval_opening_plies": cfg.eval_opening_plies,
                "batch_wait_ms": cfg.batch_wait_ms,
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
            # Distinguish losing the gate from having too little evidence.
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
                       decided_games=ev.decided_games,
                       eval_timeouts=ev.num_games - ev.decided_games,
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
                       rand_decided_games=evr.decided_games,
                       secs=time.time() - t0, eval_ran=True)
            _write_meta()

            # --- 5. eval vs greedy (absolute yardstick; opt-in) ---
            if cfg.eval_greedy_games:
                t_eval_greedy = time.time()
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs greedy "
                     f"({cfg.eval_greedy_games} games, {eval_sims} sims)...")

                def _eval_greedy_progress(done, total, r):
                    elapsed = time.time() - t_eval_greedy
                    _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs greedy: "
                         f"{done}/{total} games ({elapsed:.0f}s, cand {r.candidate_win_rate:.0%})")

                if cfg.parallel_eval:
                    evg = evaluate_against_greedy_parallel_mp(
                        model, eval_config_dict, num_games=cfg.eval_greedy_games,
                        num_workers=cfg.num_workers, batch_size=cfg.inference_batch_size,
                        on_progress=_eval_greedy_progress,
                        base_seed=it * 100_003 + 70_000, log=_log)
                else:
                    greedy = greedy_agent()
                    evg = evaluate_mp(env, cand, greedy, num_games=cfg.eval_greedy_games,
                                      max_moves=cfg.max_game_moves,
                                      on_progress=_eval_greedy_progress,
                                      base_seed=it * 100_003 + 70_000)
                eval_greedy_secs = time.time() - t_eval_greedy
                evg_wr = evg.candidate_win_rate
                # Per seat, because the seats are not symmetric: two racers meeting
                # head-on let the SECOND one jump the first, so seat 0 must spend a
                # wall to win while seat 1 wins by racing. A pooled number hides
                # which of the two the model has failed to learn.
                # str keys: json.dump stringifies int keys anyway, so building
                # them that way keeps the in-memory row and the reloaded
                # meta.json the same shape.
                greedy_by_seat = {str(s): [evg.seat_wins.get(s, 0), n]
                                  for s, n in sorted(evg.games_per_seat.items())}
                seat_str = " ".join(f"seat{s}:{w}/{n}"
                                    for s, (w, n) in greedy_by_seat.items())
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs greedy done: "
                     f"{100*evg_wr:.1f}% of {evg.decided_games} decided "
                     f"[{seat_str}] ({eval_greedy_secs:.0f}s)")
                row.update(win_vs_greedy=evg_wr, eval_greedy_secs=eval_greedy_secs,
                           greedy_decided_games=evg.decided_games,
                           greedy_by_seat=greedy_by_seat,
                           secs=time.time() - t0)

                # Racing decay: best per-seat rate against its own running peak.
                if cfg.greedy_stop_patience:
                    seated = [(w / n, n) for w, n in greedy_by_seat.values() if n]
                    best_seat, best_seat_n = max(seated) if seated else (0.0, 0)
                    row["greedy_best_seat"] = best_seat
                    prev_below = greedy_below_peak
                    greedy_peak, greedy_below_peak = racing_decay_strike(
                        best_seat, greedy_peak, greedy_below_peak,
                        cfg.greedy_stop_drop, best_seat_n, cfg.greedy_stop_z)
                    if cfg.greedy_min_seat and it >= cfg.wall_mask_iters:
                        # Armed only once the curriculum has had its chance —
                        # every run is below the floor while still untrained.
                        greedy_below_floor = stalled_below_floor(
                            best_seat, cfg.greedy_min_seat, greedy_below_floor)
                        if greedy_below_floor >= cfg.greedy_stop_patience:
                            stop_reason = (
                                f"never acquired racing: best per-seat greedy rate "
                                f"{100*best_seat:.0f}% stayed under the "
                                f"{100*cfg.greedy_min_seat:.0f}% floor for "
                                f"{greedy_below_floor} consecutive evals after the "
                                f"masked phase. The curriculum is not delivering at "
                                f"this setting; more iterations will not fix it.")
                    if greedy_below_peak > prev_below:
                        _log(f"[iter {it+1}/{cfg.num_iterations}] WARNING: racing decay — "
                             f"best seat {100*best_seat:.0f}% is {100*(greedy_peak-best_seat):.0f} "
                             f"pts below the peak {100*greedy_peak:.0f}% "
                             f"({greedy_below_peak}/{cfg.greedy_stop_patience} evals)")
                        if greedy_below_peak >= cfg.greedy_stop_patience:
                            stop_reason = (
                                f"racing decay: best per-seat greedy rate {100*best_seat:.0f}% "
                                f"stayed >={100*cfg.greedy_stop_drop:.0f} pts below its peak "
                                f"{100*greedy_peak:.0f}% for {greedy_below_peak} consecutive "
                                f"evals. best.pt holds the peak; resume by raising "
                                f"greedy_stop_patience if this was noise.")
                _write_meta()
        else:
            _log(
                f"[iter {it+1}/{cfg.num_iterations}] eval skipped (eval_every={cfg.eval_every})")

        vs_best_txt = (f"{100*ev_wr:.1f}% {'ACCEPT' if accepted else 'reject'}"
                       if ev_wr is not None else "n/a (not evaluated)")
        vs_rand_txt = f"{100*evr_wr:.1f}%" if evr_wr is not None else "n/a"
        # Measured at iteration end, so it needs its own write.
        row["rss_gb"] = _rss_gb()
        _write_meta()
        _log(f">>> iter {it+1} | loss_p={lp:.3f} loss_v={lv:.3f} | "
             f"vs_best={vs_best_txt} | "
             f"vs_rand={vs_rand_txt} | draw={100*draw_rate:.0f}% | buf={len(buffer)} | "
             f"rss={row['rss_gb']:.2f}GB | "
             f"sp={sp_secs:.0f}s train={train_secs:.0f}s "
             f"eval_best={eval_best_secs:.0f}s eval_rand={eval_rand_secs:.0f}s | "
             f"total={row['secs']:.0f}s")

        if stop_reason:
            # meta.json and both checkpoints are already written for this
            # iteration, so the run resumes from here untouched if wanted.
            _log(f"STOPPING EARLY at iteration {it+1}/{cfg.num_iterations} — {stop_reason}")
            break
    return history
